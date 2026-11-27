//! Small host buffers placed in heaps made ahead.
//!
//! A new part of the world brings hundreds of host copies in a frame,
//! each a committed buffer of its own; for textures that cost most of a
//! frame (`crate::placement`: 255 ms committed, 3 ms placed). Buffers up
//! to [`LARGEST`] go into heaps of [`HEAP`] bytes, one set of heaps for
//! video memory the CPU writes and one for system memory both sides use,
//! by runs of 64 KiB pages ([`super::pages`]).
//!
//! A buffer's pages are free again when D3D12 destroys it, not when its
//! copy drops: lists and descriptor tables may hold it longer. A buffer
//! handed out is zeroed, as a committed one is: a copy's pages read zeros
//! until they are refreshed.
//! shortcut: empty heaps are kept; release them if the card runs short.
//! shortcut: at most [`MOST`] heaps of each kind; past that, committed.

use std::sync::Mutex;

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
/// Heaps of each kind at most (4 GiB).
const MOST: usize = 64;

struct Arena {
    heap: ID3D12Heap,
    pages: Pages,
}

// SAFETY: D3D12 heaps are free-threaded; used under the lock.
unsafe impl Send for Arena {}

/// The heaps in system memory (0) and in video memory (1). An arena
/// keeps its index: a buffer's pages name it when they come back.
static POOLS: Mutex<[Vec<Arena>; 2]> = Mutex::new([Vec::new(), Vec::new()]);

/// A buffer's pages as the data of its destruction callback: kind,
/// arena, first page, pages.
fn pack(kind: usize, arena: usize, at: usize, n: usize) -> usize {
    (kind << 63) | (arena << 48) | (n << 32) | at
}

fn unpack(packed: usize) -> (usize, usize, usize, usize) {
    (packed >> 63, (packed >> 48) & 0x7fff, packed & 0xffff_ffff, (packed >> 32) & 0xffff)
}

fn give(packed: usize) {
    let (kind, arena, at, n) = unpack(packed);
    let mut pools = POOLS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(a) = pools[kind].get_mut(arena) {
        a.pages.give(at, n);
    }
}

/// Called by D3D12 when a placed buffer is destroyed (its last reference
/// gone): its pages are free again.
unsafe extern "system" fn give_back(packed: *mut std::ffi::c_void) {
    give(packed as usize);
}

/// `n` pages in a heap of `kind` with room, else in a new heap: the heap,
/// and the pages packed for [`give`].
fn take(device: &ID3D12Device, kind: usize, n: usize) -> Option<(ID3D12Heap, usize, usize)> {
    let mut pools = POOLS.lock().unwrap_or_else(|e| e.into_inner());
    let arenas = &mut pools[kind];
    for (i, a) in arenas.iter_mut().enumerate() {
        if let Some(at) = a.pages.take(n) {
            return Some((a.heap.clone(), at, pack(kind, i, at, n)));
        }
    }
    if arenas.len() >= MOST {
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
    if let Err(e) = unsafe { device.CreateHeap(&desc, &mut heap) } {
        static REFUSED: std::sync::Once = std::sync::Once::new();
        REFUSED.call_once(|| tracing::warn!(error = %e, kind, "no heap for host buffers: they stay committed"));
        return None;
    }
    let heap = heap?;
    crate::host::name(&heap, || "buffer heap".into());
    let mut pages = Pages::new((HEAP / PAGE) as usize);
    let at = pages.take(n)?;
    arenas.push(Arena { heap: heap.clone(), pages });
    Some((heap, at, pack(kind, arenas.len() - 1, at, n)))
}

/// A mapped, zeroed buffer of `size` bytes shaders may write, in the
/// common state, placed in a heap: in video memory the CPU writes
/// (`upload`, a GPU upload heap) or in system memory. None for a buffer
/// past [`LARGEST`], or when it cannot be placed (the caller then makes
/// a committed one).
fn place(device: &ID3D12Device, upload: bool, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    let size = size.max(1);
    if size > LARGEST {
        return None;
    }
    let n = size.div_ceil(PAGE) as usize;
    let Some((heap, at, packed)) = take(device, usize::from(upload), n) else {
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
            Some((buffer?, data))
        }
        Err(e) => {
            // Outside the lock: nothing of the buffer is left to call back.
            drop(buffer);
            give(packed);
            static REFUSED: std::sync::Once = std::sync::Once::new();
            REFUSED.call_once(|| tracing::warn!(error = %e, upload, size, "a host buffer was not placed: committed"));
            None
        }
    }
}

/// [`place`], where the pools are on ([`super::pooling`]).
pub(crate) fn buffer(device: &ID3D12Device, upload: bool, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    if !super::pooling() {
        return None;
    }
    place(device, upload, size)
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

    /// A placed buffer's pages are free again once the buffer is
    /// destroyed, and the next buffer there reads zeros.
    #[test]
    fn a_pooled_buffer_is_zeroed_and_its_range_comes_back() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let free = || POOLS.lock().unwrap().iter().flatten().map(|a| a.pages.free()).sum::<usize>();
        let len = 3 * PAGE as usize;
        // One first, so that the heap exists before pages are counted.
        let Some(keep) = place(&host.device, false, PAGE) else {
            return; // this host places no such buffers
        };
        let before = free();
        let (a, data) = place(&host.device, false, len as u64).unwrap();
        assert_eq!(free(), before - 3);
        // SAFETY: (both) `len` mapped bytes of the buffer.
        unsafe { data.write_bytes(0xee, len) };
        let held = a.clone();
        drop(a);
        assert_eq!(free(), before - 3, "still held: not free");
        drop(held);
        assert_eq!(free(), before, "destroyed: free again");
        let (b, data) = place(&host.device, false, len as u64).unwrap();
        // SAFETY: as above.
        assert!(unsafe { std::slice::from_raw_parts(data, len) }.iter().all(|&x| x == 0));
        assert!(place(&host.device, false, LARGEST + 1).is_none(), "too large to place");
        drop((b, keep));
    }
}
