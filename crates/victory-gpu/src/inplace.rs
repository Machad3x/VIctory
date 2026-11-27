//! Title memory in place (decision 0005's unified-memory item): a host heap
//! opened over memory the process already has
//! (`ID3D12Device13::OpenExistingHeapFromAddress1`), and one buffer placed
//! over all of it, so that shaders read and write those bytes themselves
//! and no host copy stands between.
//!
//! On a discrete card such a heap is system memory: shaders reach it over
//! the bus, far slower than a copy in video memory. So it is not a home for
//! data shaders read a lot, only for memory that would otherwise have no
//! copy at all (reads of zeros).
//!
//! What the probe below answers on a given PC and driver (run it by hand:
//! `cargo test -p victory-gpu --lib inplace -- --ignored --nocapture`):
//! which openings the driver takes (private memory, a view of a section
//! reserved and committed through another view, as `xmem` makes the page
//! pool), what alignment and size it asks, what an opened heap forbids
//! (protecting, freeing, unmapping), and whether the GPU reads and writes
//! the right bytes.
//!
//! The backstop (`VICTORY_INPLACE_BACKSTOP=1`, off by default): title
//! memory that no host copy holds, which shaders read as zeros and whose
//! writes they drop (`crate::memory`'s misses, and the withholding of
//! whole frames' results that follows), is given to the page table in
//! place instead. Only pages of physical memory (`xmem`'s sections) are
//! covered, through our own view of their section
//! (`crate::watch::title_memory_own`): the title's views stay free to be
//! guarded and remapped, since a heap locks only the view it is opened on.
//! The heaps stay open while the section lives; the runtime's unmapping of
//! our view lets them go first (`crate::pinned`).
//! shortcut: private memory with no copy is not covered: a heap over it
//! would make the title's and the guards' protections fail; open it once
//! the counters show such misses.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;

use crate::watch::PAGE;

/// Opens `[at, at + len)` of committed process memory (64 KiB aligned, one
/// protection throughout) as a host heap and places one buffer over all of
/// it, which shaders may read and write, in the common state. While the
/// heap lives the memory cannot be protected, decommitted, freed or
/// unmapped (`crate::pinned`).
pub(crate) fn open_at(device: &ID3D12Device, at: usize, len: usize) -> windows::core::Result<(ID3D12Heap, ID3D12Resource)> {
    let device13: ID3D12Device13 = device.cast()?;
    // SAFETY: the caller's committed memory, which outlives the heap.
    let heap: ID3D12Heap = unsafe { device13.OpenExistingHeapFromAddress1(at as *const c_void, len) }?;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: len as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation in a live heap.
    unsafe { device.CreatePlacedResource(&heap, 0, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer) }?;
    let buffer = buffer.ok_or_else(|| windows::core::Error::from(windows::Win32::Foundation::E_FAIL))?;
    Ok((heap, buffer))
}

/// Whether title memory no copy holds goes to the page table in place
/// (`VICTORY_INPLACE_BACKSTOP=1`).
pub(crate) fn backstop() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("VICTORY_INPLACE_BACKSTOP").is_some_and(|v| v == "1"));
    *ON
}

/// The most bytes open in place at once (`VICTORY_INPLACE_MB`, 1024):
/// opening locks the pages into the PC's memory.
fn budget() -> usize {
    static MB: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("VICTORY_INPLACE_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1024)
    });
    MB.saturating_mul(1 << 20)
}

/// The longest run: one 256 MiB region of the page table.
const MOST: usize = 1 << 28;

/// Our own view's `[at, at + len)` open in place: a heap over it and one
/// buffer placed over all of it.
pub(crate) struct Run {
    /// Tells runs apart in the page tables' view pairs (an address may be
    /// opened again after a release).
    pub(crate) id: u64,
    pub(crate) len: usize,
    pub(crate) buffer: ID3D12Resource,
    _heap: ID3D12Heap,
}

// SAFETY: D3D12 objects are free-threaded; the rest is plain data.
unsafe impl Send for Run {}
// SAFETY: as above.
unsafe impl Sync for Run {}

/// Runs open now, by address.
static RUNS: Mutex<BTreeMap<usize, Arc<Run>>> = Mutex::new(BTreeMap::new());
/// Where the host refused to open a run: not asked again.
static REFUSED: Mutex<std::collections::BTreeSet<usize>> = Mutex::new(std::collections::BTreeSet::new());
static NEXT: AtomicU64 = AtomicU64::new(1);
static OPEN: AtomicUsize = AtomicUsize::new(0);
/// Bumped when what the backstop would name changes: views of our own
/// mapped or gone, a run let go. Page tables are rewritten when it moves.
static GENERATION: AtomicU64 = AtomicU64::new(0);
/// Since the start: runs refused for the budget, refused by the host; the
/// pages named by the last table written; pages shaders wrote in place
/// since the last frame line.
static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

pub(crate) fn generation() -> u64 {
    GENERATION.load(Ordering::Relaxed)
}

/// What views of our own show changed: page tables name it anew.
pub(crate) fn views_changed() {
    if backstop() {
        GENERATION.fetch_add(1, Ordering::Relaxed);
    }
}

/// The value of a page-table entry's second dword for the title page at
/// `page` shown `offset` bytes into a run: shaders take the address's low
/// dword less it as the byte offset in the run's view.
pub(crate) fn entry_base(page: u64, offset: usize) -> u32 {
    (page as u32).wrapping_sub(offset as u32)
}

/// `pages` (title page index, our view's address), in page order, cut into
/// runs of consecutive pages that our view shows consecutively, each at
/// most [`MOST`].
pub(crate) fn runs(pages: &[(usize, usize)]) -> Vec<&[(usize, usize)]> {
    let mut out = Vec::new();
    let mut start = 0;
    for i in 1..=pages.len() {
        let joined = i < pages.len()
            && pages[i].0 == pages[i - 1].0 + 1
            && pages[i].1 == pages[i - 1].1 + PAGE
            && (i - start) * PAGE < MOST;
        if !joined {
            out.push(&pages[start..i]);
            start = i;
        }
    }
    out.retain(|r| !r.is_empty());
    out
}

/// The run holding our view's `[at, at + len)`, and where it starts in
/// it: one open already, else opened now (the pages committed through our
/// view too, which commits nothing new: the title's view committed them).
/// None if the budget or the host refuses.
pub(crate) fn run_for(at: usize, len: usize) -> Option<(Arc<Run>, usize)> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, PAGE_READWRITE, VirtualAlloc};
    {
        let runs = RUNS.lock().unwrap_or_else(|e| e.into_inner());
        for (&start, run) in runs.range(..=at).rev() {
            if at - start >= MOST {
                break;
            }
            if at + len <= start + run.len {
                return Some((run.clone(), at - start));
            }
        }
    }
    if REFUSED.lock().unwrap_or_else(|e| e.into_inner()).contains(&at) {
        return None;
    }
    if OPEN.load(Ordering::Relaxed) + len > budget() {
        STATS[0].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let host = crate::host::host()?;
    // SAFETY: our own view of a section whose pages the title's view
    // committed.
    let committed = !unsafe { VirtualAlloc(at as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null();
    let opened = if committed {
        open_at(&host.device, at, len).map_err(|e| e.to_string())
    } else {
        Err(std::io::Error::last_os_error().to_string())
    };
    let (heap, buffer) = match opened {
        Ok(opened) => opened,
        Err(error) => {
            REFUSED.lock().unwrap_or_else(|e| e.into_inner()).insert(at);
            if STATS[1].fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::warn!(at = format_args!("{at:#x}"), len, %error, "title memory not opened in place: it reads as missing");
            }
            return None;
        }
    };
    crate::host::name(&buffer, || format!("title memory in place {at:#x}+{len:#x}"));
    let run = Arc::new(Run { id: NEXT.fetch_add(1, Ordering::Relaxed), len, buffer, _heap: heap });
    RUNS.lock().unwrap_or_else(|e| e.into_inner()).insert(at, run.clone());
    OPEN.fetch_add(len, Ordering::Relaxed);
    // Let go before our view is unmapped.
    // shortcut: a page table naming the run keeps its heap until rewritten,
    // so an unmap meanwhile fails after its retries and leaves our view
    // mapped; sections are freed rarely (the pool is made once).
    let id = run.id;
    let _ = crate::pinned::pin(at, len, Box::new(move || let_go(at, id)));
    tracing::debug!(at = format_args!("{at:#x}"), len, "title memory opened in place");
    Some((run, 0))
}

/// The run `id` at `at` is let go: page tables name it no more.
fn let_go(at: usize, id: u64) {
    let mut runs = RUNS.lock().unwrap_or_else(|e| e.into_inner());
    if runs.get(&at).is_some_and(|r| r.id == id)
        && let Some(run) = runs.remove(&at)
    {
        OPEN.fetch_sub(run.len, Ordering::Relaxed);
        GENERATION.fetch_add(1, Ordering::Relaxed);
        drop(runs);
        crate::host::release_later(Box::new(run));
    }
}

/// The last page table written named `pages` in place.
pub(crate) fn named(pages: u64) {
    STATS[2].store(pages, Ordering::Relaxed);
}

/// Shaders wrote `pages` pages in place.
pub(crate) fn written(pages: u64) {
    STATS[3].fetch_add(pages, Ordering::Relaxed);
}

/// The frame line, with the backstop on.
pub(crate) fn log_frame() {
    if !backstop() {
        return;
    }
    let written = STATS[3].swap(0, Ordering::Relaxed);
    let named = STATS[2].load(Ordering::Relaxed);
    if named + written == 0 {
        return;
    }
    tracing::info!(
        "title memory in place (VICTORY_INPLACE_BACKSTOP): {} runs, {:.1} MB open, {named} pages named by the last page table, {written} pages written this frame; runs refused: {} past VICTORY_INPLACE_MB, {} by the host",
        RUNS.lock().unwrap_or_else(|e| e.into_inner()).len(),
        OPEN.load(Ordering::Relaxed) as f64 / 1e6,
        STATS[0].load(Ordering::Relaxed),
        STATS[1].load(Ordering::Relaxed)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_shown_one_after_another_make_one_run() {
        let o = 0x7000_0000_0000usize;
        let pages = [
            (0x100, o),
            (0x101, o + PAGE),
            (0x102, o + 2 * PAGE),
            // Our view shows the next page elsewhere: a run of its own.
            (0x103, o + 9 * PAGE),
            // A page between them has none: another run.
            (0x105, o + 11 * PAGE),
        ];
        let found = runs(&pages);
        let lens: Vec<usize> = found.iter().map(|r| r.len()).collect();
        assert_eq!(lens, [3, 1, 1]);
        assert_eq!(found[1][0], (0x103, o + 9 * PAGE));
        assert!(runs(&[]).is_empty());
    }

    #[test]
    fn a_run_is_at_most_a_region() {
        let o = 0x7000_0000_0000usize;
        let n = MOST / PAGE + 2;
        let pages: Vec<(usize, usize)> = (0..n).map(|k| (k, o + k * PAGE)).collect();
        let lens: Vec<usize> = runs(&pages).iter().map(|r| r.len()).collect();
        assert_eq!(lens, [MOST / PAGE, 2]);
    }

    /// What shaders compute from an entry (the address's low dword less the
    /// entry's) is the byte's offset in the run, across a 4 GiB boundary too.
    #[test]
    fn an_entry_gives_the_bytes_offset_in_its_run() {
        for (page, offset) in [(0x12_3456_0000u64, 0x30_0000usize), (0x1_0001_0000, 0x2_0000), (0x5_0000, 0)] {
            let base = entry_base(page, offset);
            for inside in [0u64, 4, 0xFFFC] {
                let lo = (page + inside) as u32;
                assert_eq!(u64::from(lo.wrapping_sub(base)), offset as u64 + inside);
            }
        }
    }
}

#[cfg(test)]
mod probe {
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory as mem;

    use super::*;

    const K64: usize = 64 << 10;
    /// The host's write watching (not in windows-sys' memory module).
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    fn fill(at: usize, len: usize, seed: u32) {
        // SAFETY: the caller's committed, writable memory.
        let words = unsafe { std::slice::from_raw_parts_mut(at as *mut u32, len / 4) };
        for (i, w) in words.iter_mut().enumerate() {
            *w = (i as u32).wrapping_mul(0x9E37_79B9) ^ seed;
        }
    }

    fn wrong(at: usize, len: usize, seed: u32) -> usize {
        // SAFETY: the caller's committed memory.
        let words = unsafe { std::slice::from_raw_parts(at as *const u32, len / 4) };
        words.iter().enumerate().filter(|&(i, &w)| w != (i as u32).wrapping_mul(0x9E37_79B9) ^ seed).count()
    }

    fn error() -> u32 {
        // SAFETY: plain read of this thread's last error.
        unsafe { GetLastError() }
    }

    fn protect(at: usize, len: usize) -> Result<(), u32> {
        let mut old = 0;
        // SAFETY: our memory; set back to read-write right after.
        if unsafe { mem::VirtualProtect(at as *const c_void, len, mem::PAGE_READONLY, &mut old) } == 0 {
            return Err(error());
        }
        // SAFETY: as above.
        unsafe { mem::VirtualProtect(at as *const c_void, len, mem::PAGE_READWRITE, &mut old) };
        Ok(())
    }

    /// What the GPU makes of the in-place buffer `buffer` over `at`: it
    /// reads the CPU's bytes (copied out into a buffer of ours) and its
    /// writes land where the CPU reads (copied in at `len / 2`). Wrong
    /// dwords each way.
    fn gpu_round_trip(buffer: &ID3D12Resource, at: usize, len: usize) -> (usize, usize) {
        let host = crate::host::host().expect("a host");
        let half = len / 2;
        let (out, out_data) = host.shared_buffer(half as u64).expect("a buffer");
        fill(at, half, 0x1234_5678);
        host.copy_now(&[(&out, 0, buffer, 0, half as u64)]);
        let read = wrong(out_data as usize, half, 0x1234_5678);
        fill(out_data as usize, half, 0x0BAD_F00D);
        host.copy_now(&[(buffer, half as u64, &out, 0, half as u64)]);
        let written = wrong(at + half, half, 0x0BAD_F00D);
        (read, written)
    }

    /// GB/s of a GPU copy of `len` bytes out of the in-place buffer into
    /// video memory (the whole call, submission included: a lower bound).
    fn read_speed(buffer: &ID3D12Resource, len: usize) -> f64 {
        let host = crate::host::host().expect("a host");
        let vram = host.uav_buffer(len as u64).expect("a buffer");
        host.copy_now(&[(&vram, 0, buffer, 0, len as u64)]);
        let started = std::time::Instant::now();
        host.copy_now(&[(&vram, 0, buffer, 0, len as u64)]);
        len as f64 / started.elapsed().as_secs_f64() / 1e9
    }

    /// The probe. Asserts only what the in-place stages build on when an
    /// opening works (the GPU's bytes are the memory's); the rest is
    /// printed. `VICTORY_INPLACE_PROBE_MAX=1` also tries 4095 and 4096 MiB
    /// (commits and locks 4 GiB for a moment).
    #[test]
    #[ignore = "a probe of the host's driver, run by hand"]
    fn probe_title_memory_in_place() {
        let Some(host) = crate::host::host() else {
            println!("no host GPU");
            return;
        };
        if host.device.cast::<ID3D12Device13>().is_err() {
            println!("no ID3D12Device13: OpenExistingHeapFromAddress1 unavailable");
            return;
        }
        let device = &host.device;
        const BYTES: usize = 8 << 20;

        // Private memory, as `xmem` reserves the title's heaps.
        // SAFETY: plain allocation, released below.
        let at = unsafe {
            mem::VirtualAlloc(std::ptr::null(), BYTES, mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH, mem::PAGE_READWRITE)
        } as usize;
        assert_ne!(at, 0);
        for (what, off, len) in [
            ("4 KiB in", 0x1000, K64),
            ("64 KiB in, 60 KiB long", K64, K64 - 0x1000),
        ] {
            let r = open_at(device, at + off, len);
            println!("private, {what}: {}", r.as_ref().map(|_| "opened".to_owned()).unwrap_or_else(|e| e.to_string()));
        }
        {
            // The middle half opened.
            let (part, len) = (at + BYTES / 4, BYTES / 2);
            match open_at(device, part, len) {
                Ok((_heap, buffer)) => {
                    let (read, written) = gpu_round_trip(&buffer, part, len);
                    println!("private part: GPU read {read} wrong dwords, wrote {written} wrong");
                    assert_eq!((read, written), (0, 0), "the GPU's bytes are the memory's");
                    println!("private part: protect inside {:?}, outside {:?}", protect(part, K64), protect(at, K64));
                    println!("private part: GPU copy out at {:.2} GB/s", read_speed(&buffer, len));
                }
                Err(e) => println!("private part: refused, {e}"),
            }
        }
        // SAFETY: ours, no heap on it any more.
        assert_ne!(unsafe { mem::VirtualFree(at as *mut c_void, 0, mem::MEM_RELEASE) }, 0);

        // A section reserved and committed through one view (the title's),
        // opened through another (ours), as `xmem` makes physical pages.
        // SAFETY: plain section creation, closed below.
        let section = unsafe {
            mem::CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), mem::PAGE_READWRITE | mem::SEC_RESERVE, 0, BYTES as u32, std::ptr::null())
        };
        assert!(!section.is_null());
        // SAFETY: views of our section, unmapped below.
        let (title, own) = unsafe {
            (
                mem::MapViewOfFile(section, mem::FILE_MAP_WRITE, 0, 0, BYTES).Value as usize,
                mem::MapViewOfFile(section, mem::FILE_MAP_WRITE, 0, 0, BYTES).Value as usize,
            )
        };
        assert!(title != 0 && own != 0);
        let len = BYTES / 2;
        // SAFETY: the title view's first half, committed.
        assert!(!unsafe { mem::VirtualAlloc(title as *const c_void, len, mem::MEM_COMMIT, mem::PAGE_READWRITE) }.is_null());
        let first = open_at(device, own, len);
        println!(
            "section view committed through another: {}",
            first.as_ref().map(|_| "opened".to_owned()).unwrap_or_else(|e| e.to_string())
        );
        drop(first);
        // Committed through our own view too (what the backstop does).
        // SAFETY: our view's first half, the same section pages.
        assert!(!unsafe { mem::VirtualAlloc(own as *const c_void, len, mem::MEM_COMMIT, mem::PAGE_READWRITE) }.is_null());
        match open_at(device, own, len) {
            Ok((heap, buffer)) => {
                // The GPU writes through our view what the title's shows.
                let (read, written) = gpu_round_trip(&buffer, own, len);
                println!("section: GPU read {read} wrong dwords, wrote {written} wrong");
                assert_eq!((read, written), (0, 0), "the GPU's bytes are the memory's");
                assert_eq!(wrong(title + len / 2, len / 2, 0x0BAD_F00D), 0, "the title's view shows the GPU's writes");
                println!("section: protect the title's view {:?}, our view {:?}", protect(title, K64), protect(own, K64));
                // SAFETY: the title's view, unmapped while our view's heap lives.
                let unmapped = unsafe { mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: title as *mut c_void }) } != 0;
                println!("section: the title's view unmapped while ours is open: {unmapped} ({})", error());
                // SAFETY: our view, while its heap lives.
                let refused = unsafe { mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: own as *mut c_void }) } == 0;
                println!("section: our view's unmap refused while open: {refused} ({})", error());
                drop((heap, buffer));
                if !unmapped {
                    // SAFETY: as above, no heap now.
                    unsafe { mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: title as *mut c_void }) };
                }
                if refused {
                    // SAFETY: our view, no heap now.
                    unsafe { mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: own as *mut c_void }) };
                }
            }
            Err(e) => {
                println!("section through our committed view: refused, {e}");
                // SAFETY: our views.
                unsafe {
                    mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: title as *mut c_void });
                    mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: own as *mut c_void });
                }
            }
        }
        // SAFETY: our handle.
        unsafe { CloseHandle(section) };

        if std::env::var_os("VICTORY_INPLACE_PROBE_MAX").is_some_and(|v| v == "1") {
            for mib in [4095usize, 4096] {
                let len = mib << 20;
                // SAFETY: plain allocation, released below.
                let at = unsafe { mem::VirtualAlloc(std::ptr::null(), len, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE) } as usize;
                if at == 0 {
                    println!("{mib} MiB: no memory ({})", error());
                    continue;
                }
                let started = std::time::Instant::now();
                let r = open_at(device, at, len).map(|_| "opened".to_owned()).unwrap_or_else(|e| e.to_string());
                println!("{mib} MiB: {r} in {:.0} ms", started.elapsed().as_secs_f64() * 1e3);
                // SAFETY: ours, the heap dropped above.
                unsafe { mem::VirtualFree(at as *mut c_void, 0, mem::MEM_RELEASE) };
            }
        }
    }
}
