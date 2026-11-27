//! Small host buffers placed in heaps made ahead.
//!
//! A new part of the world brings hundreds of host copies in a frame,
//! each a committed buffer of its own; for textures that cost most of a
//! frame (`crate::placement`: 255 ms committed, 3 ms placed). Buffers up
//! to [`LARGEST`] go into heaps of [`HEAP`] bytes, one set of heaps for
//! video memory the CPU writes and one for system memory both sides use,
//! by runs of 64 KiB pages ([`super::pages`]).
//!
//! When a buffer's pages are free again. A placed buffer leaves in one of
//! two ways:
//! - It could not be mapped or watched after it was made: no list ever
//!   saw it, and [`place`] gives its pages back at once.
//! - D3D12 destroys it, its last reference gone (the copy's, lists',
//!   descriptor tables' and deferred releases' clones): [`give_back`]
//!   notes its pages with the submission sent last ([`super::sent`]), and
//!   they are free once that submission has run ([`super::done`]). A
//!   buffer whose holders all let go while the GPU still runs lists that
//!   name it is thus not handed out, zeroed and written under them.
//!
//! There is no third way: the heaps are never released, and nothing else
//! clears a page. A buffer handed out is zeroed, as a committed one is: a
//! copy's pages read zeros until they are refreshed.
//!
//! Heaps in video memory count whole against the budget host copies in
//! GPU upload heaps share (`crate::host::charge_upload`), so the buffers
//! placed in them are not charged one by one.
//! shortcut: empty heaps are kept; release them if the card runs short.
//! shortcut: at most [`MOST`] heaps of each kind; past that, committed.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use windows::Win32::Graphics::Direct3D::ID3DDestructionNotifier;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;

use super::pages::Pages;

/// Bytes of one heap.
const HEAP: u64 = 64 << 20;
/// Bytes of a page (the placement alignment of buffers).
const PAGE: u64 = 64 << 10;
/// The largest buffer placed: larger ones are few, and their own.
const LARGEST: u64 = 4 << 20;
/// Heaps of each kind at most (4 GiB; those in video memory stop at the
/// budget before).
const MOST: usize = 64;

struct Arena {
    heap: ID3D12Heap,
    pages: Pages,
}

// SAFETY: D3D12 heaps are free-threaded; used under the lock.
unsafe impl Send for Arena {}

struct Pools {
    /// The heaps in system memory (0) and in video memory (1). An arena
    /// keeps its index: a buffer's pages name it when they come back.
    arenas: [Vec<Arena>; 2],
    /// The arena of each kind that had room last: looked at first.
    hint: [usize; 2],
    /// Pages of destroyed buffers (packed), each with the submission
    /// sent last when its buffer went.
    pending: Vec<(usize, u64)>,
}

static POOLS: Mutex<Pools> =
    Mutex::new(Pools { arenas: [Vec::new(), Vec::new()], hint: [0, 0], pending: Vec::new() });

/// The host refused a heap or a placed buffer of this kind: its buffers
/// are committed from then on, without asking again.
static REFUSED: [AtomicBool; 2] = [const { AtomicBool::new(false) }; 2];

/// A buffer's pages as the data of its destruction callback: kind,
/// arena, first page, pages.
fn pack(kind: usize, arena: usize, at: usize, n: usize) -> usize {
    (kind << 63) | (arena << 48) | (n << 32) | at
}

fn unpack(packed: usize) -> (usize, usize, usize, usize) {
    (packed >> 63, (packed >> 48) & 0x7fff, packed & 0xffff_ffff, (packed >> 32) & 0xffff)
}

/// Takes out of `pending` the pages whose submission has run (`done`:
/// the last one that has) and calls `free` with each.
fn due(pending: &mut Vec<(usize, u64)>, done: u64, mut free: impl FnMut(usize)) {
    let mut i = 0;
    while i < pending.len() {
        if pending[i].1 <= done {
            free(pending.swap_remove(i).0);
        } else {
            i += 1;
        }
    }
}

impl Pools {
    fn give(arenas: &mut [Vec<Arena>; 2], packed: usize) {
        let (kind, arena, at, n) = unpack(packed);
        if let Some(a) = arenas[kind].get_mut(arena) {
            a.pages.give(at, n);
        }
    }

    /// Frees the pages of destroyed buffers the GPU is done with.
    fn reap(&mut self) {
        let Pools { arenas, pending, .. } = self;
        due(pending, super::done(), |packed| Self::give(arenas, packed));
    }
}

/// Called by D3D12 when a placed buffer is destroyed (its last reference
/// gone): its pages are free once the submission sent last has run.
unsafe extern "system" fn give_back(packed: *mut std::ffi::c_void) {
    let mut pools = POOLS.lock().unwrap_or_else(|e| e.into_inner());
    pools.pending.push((packed as usize, super::sent()));
}

/// `n` pages in a heap of `kind` with room, else in a new heap: the heap,
/// the first page, and the pages packed.
fn take(device: &ID3D12Device, kind: usize, n: usize) -> Option<(ID3D12Heap, usize, usize)> {
    let mut pools = POOLS.lock().unwrap_or_else(|e| e.into_inner());
    pools.reap();
    let count = pools.arenas[kind].len();
    let hint = pools.hint[kind];
    for k in 0..count {
        let i = (hint + k) % count;
        let a = &mut pools.arenas[kind][i];
        // `Pages::take` refuses as fast, but most heaps are full.
        if a.pages.free() < n {
            continue;
        }
        if let Some(at) = a.pages.take(n) {
            let heap = a.heap.clone();
            pools.hint[kind] = i;
            return Some((heap, at, pack(kind, i, at, n)));
        }
    }
    if count >= MOST {
        return None;
    }
    // A heap in video memory is the host copies' budget's, whole.
    if kind == 1 && !crate::host::charge_upload(HEAP) {
        return None;
    }
    let properties = if kind == 1 {
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() }
    } else {
        D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        }
    };
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: HEAP,
        Properties: properties,
        Alignment: PAGE,
        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
    };
    let mut heap: Option<ID3D12Heap> = None;
    // SAFETY: plain creation, our out pointer.
    let made = unsafe { device.CreateHeap(&desc, &mut heap) };
    let heap = match (made, heap) {
        (Ok(()), Some(heap)) => heap,
        (made, _) => {
            if kind == 1 {
                crate::host::uncharge_upload(HEAP);
            }
            REFUSED[kind].store(true, Ordering::Relaxed);
            tracing::warn!(error = ?made.err(), kind, "no heap for host buffers: they stay committed");
            return None;
        }
    };
    crate::host::name(&heap, || "buffer heap".into());
    let mut pages = Pages::new((HEAP / PAGE) as usize);
    let at = pages.take(n)?;
    pools.arenas[kind].push(Arena { heap: heap.clone(), pages });
    pools.hint[kind] = count;
    Some((heap, at, pack(kind, count, at, n)))
}

/// A mapped, zeroed buffer of `size` bytes shaders may write, in the
/// common state, placed in a heap: in video memory the CPU writes
/// (`upload`, a GPU upload heap) or in system memory; with its pages
/// packed. None for a buffer past [`LARGEST`], or when it cannot be
/// placed (the caller then makes a committed one).
fn place_packed(device: &ID3D12Device, upload: bool, size: u64) -> Option<(ID3D12Resource, *mut u8, usize)> {
    let size = size.max(1);
    let kind = usize::from(upload);
    if size > LARGEST || REFUSED[kind].load(Ordering::Relaxed) {
        return None;
    }
    let n = size.div_ceil(PAGE) as usize;
    let Some((heap, at, packed)) = take(device, kind, n) else {
        super::add(super::N::PoolFull, 1);
        return None;
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    let mut data = std::ptr::null_mut();
    // SAFETY: the pages are ours alone until given back; plain creation
    // and mapping of a CPU-visible buffer.
    let made = unsafe {
        device
            .CreatePlacedResource(&heap, at as u64 * PAGE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
            .and_then(|()| {
                let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                b.Map(0, None, Some(&mut data))
            })
    };
    // The callback last: from then on the buffer's end gives the pages
    // back, before that we do.
    // SAFETY: a live buffer; the callback's data is the packed pages.
    let watched = made.and_then(|()| unsafe {
        let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
        b.cast::<ID3DDestructionNotifier>()?.RegisterDestructionCallback(Some(give_back), packed as *const _)
    });
    match watched {
        Ok(_) => {
            let data: *mut u8 = data.cast();
            // SAFETY: `size` mapped bytes of ours.
            unsafe { data.write_bytes(0, size as usize) };
            super::add(super::N::Pooled, 1);
            super::add(super::N::PooledBytes, size);
            Some((buffer?, data, packed))
        }
        Err(e) => {
            // No list saw it and nothing calls back for it: the pages are
            // free at once. The buffer goes first, outside the lock.
            drop(buffer);
            let mut pools = POOLS.lock().unwrap_or_else(|e| e.into_inner());
            Pools::give(&mut pools.arenas, packed);
            drop(pools);
            REFUSED[kind].store(true, Ordering::Relaxed);
            tracing::warn!(error = %e, upload, size, "a host buffer was not placed: they stay committed");
            None
        }
    }
}

/// [`place_packed`], where the pools are on ([`super::pooling`]).
pub(crate) fn buffer(device: &ID3D12Device, upload: bool, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    if !super::pooling() {
        return None;
    }
    place_packed(device, upload, size).map(|(buffer, data, _)| (buffer, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_survive_packing() {
        for &(kind, arena, at, n) in &[(0, 0, 0, 1), (1, 63, 1023, 64), (1, 0, 5, 3), (0, 17, 960, 64)] {
            assert_eq!(unpack(pack(kind, arena, at, n)), (kind, arena, at, n));
        }
    }

    /// Pages of a destroyed buffer are free once the submission sent
    /// last when it went has run, not before.
    #[test]
    fn pages_wait_for_the_submission_sent_when_their_buffer_went() {
        let mut pending = vec![(10, 5), (11, 7), (12, 5), (13, 6)];
        let mut freed = Vec::new();
        due(&mut pending, 4, |p| freed.push(p));
        assert!(freed.is_empty() && pending.len() == 4, "submission 5 still runs");
        due(&mut pending, 5, |p| freed.push(p));
        freed.sort_unstable();
        assert_eq!(freed, [10, 12]);
        due(&mut pending, 6, |p| freed.push(p));
        assert_eq!((freed.last(), pending.as_slice()), (Some(&13), &[(11, 7)][..]));
        due(&mut pending, u64::MAX, |p| freed.push(p));
        assert!(pending.is_empty() && freed.len() == 4);
    }

    /// A placed buffer's pages stay taken while anything holds it, are
    /// free again once it is destroyed and the GPU is done, and a buffer
    /// handed out reads zeros. Its own pages only: other tests may place
    /// buffers meanwhile.
    #[test]
    fn a_pooled_buffer_is_zeroed_and_its_range_comes_back() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let len = 3 * PAGE as usize;
        let Some((a, data, packed)) = place_packed(&host.device, false, len as u64) else {
            return; // this host places no such buffers
        };
        let (kind, arena, at, n) = unpack(packed);
        assert_eq!((kind, n), (0, 3));
        let taken = || {
            let mut pools = POOLS.lock().unwrap();
            pools.reap();
            let pages = &pools.arenas[kind][arena].pages;
            (at..at + n).filter(|&p| pages.is_used(p)).count()
        };
        // SAFETY: (both) `len` mapped bytes of the buffer.
        unsafe { data.write_bytes(0xee, len) };
        let held = a.clone();
        drop(a);
        assert_eq!(taken(), 3, "still held: not free");
        drop(held);
        // Free once the submissions other tests have on the GPU have run.
        let mut left = 3;
        for _ in 0..500 {
            left = taken();
            if left == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(left, 0, "destroyed and the GPU done: free again");
        let (b, data, _) = place_packed(&host.device, false, len as u64).unwrap();
        // SAFETY: as above.
        assert!(unsafe { std::slice::from_raw_parts(data, len) }.iter().all(|&x| x == 0));
        assert!(place_packed(&host.device, false, LARGEST + 1).is_none(), "too large to place");
        drop(b);
    }
}
