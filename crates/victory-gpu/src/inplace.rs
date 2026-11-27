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
//! Runs are opened on a thread of their own, joined with the open runs they
//! overlap. The heaps stay open while the section lives; when the runtime
//! lets go of our view, its runs are forgotten and the view stays mapped
//! until no page table names them (a frame's end, [`settle`]).
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

/// The longest run asked for: one 256 MiB region of the page table.
const MOST: usize = 1 << 28;
/// The longest run opened: asked-for runs joined with the open ones they
/// overlap, up to a raw view's reach (2^27 dwords).
const LONGEST: usize = 2 * MOST;

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

/// Our view of a section the runtime let go, held mapped until no page
/// table names a run over it.
struct Gone {
    runs: Vec<Arc<Run>>,
    /// Unmaps it; nonzero when that worked.
    unmap: Box<dyn FnMut() -> i32 + Send>,
    /// Frame ends it has waited.
    frames: u32,
}

#[derive(Default)]
struct Runs {
    /// Runs open now, by address.
    open: BTreeMap<usize, Arc<Run>>,
    /// Asked for and not open yet (the worker opens them), and refused by
    /// the host (not asked again).
    pending: std::collections::BTreeSet<(usize, usize)>,
    refused: std::collections::BTreeSet<(usize, usize)>,
    /// Ids of runs over views gone that page tables may still name.
    dead: Vec<u64>,
    gone: Vec<Gone>,
}

static RUNS: Mutex<Option<Runs>> = Mutex::new(None);

fn with_runs<R>(f: impl FnOnce(&mut Runs) -> R) -> R {
    f(RUNS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default())
}

/// Held from a run's open to its insertion, and by a view's going: a view
/// never goes between its run's open and the run's insertion.
static OPENING: Mutex<()> = Mutex::new(());
static NEXT: AtomicU64 = AtomicU64::new(1);
static OPEN: AtomicUsize = AtomicUsize::new(0);
/// Bumped when what the backstop would name changes: views of our own
/// mapped or gone, a run opened or let go. Page tables are rewritten when
/// it moves.
static GENERATION: AtomicU64 = AtomicU64::new(0);
/// Runs refused for the budget and by the host since the start; pages the
/// last table named; since the last frame line: pages shaders wrote in
/// place, those of them in lists or frames whose writes are withheld,
/// runs opened, and the microseconds opening took (on the worker).
static STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

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

/// `run` cut where an allocation ends (`end_of` gives the end of the one
/// holding an address of our views): sections' views may lie side by
/// side, and a heap is opened on one allocation.
pub(crate) fn per_view(run: &[(usize, usize)], end_of: impl Fn(usize) -> usize) -> Vec<&[(usize, usize)]> {
    let mut out = Vec::new();
    let mut rest = run;
    while let Some(&(_, own)) = rest.first() {
        let end = end_of(own);
        let n = rest.iter().take_while(|&&(_, o)| o.saturating_add(PAGE) <= end).count().max(1);
        let (head, tail) = rest.split_at(n);
        out.push(head);
        rest = tail;
    }
    out
}

/// Where the allocation (a view) holding `at` ends; no end if Windows
/// does not say.
pub(crate) fn allocation_end(at: usize) -> usize {
    use windows_sys::Win32::System::Memory::{MEM_FREE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let query = |p: usize| {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) };
        (n != 0).then_some(info)
    };
    let Some(first) = query(at) else {
        return usize::MAX;
    };
    let mut end = first.BaseAddress as usize + first.RegionSize;
    while let Some(info) = query(end)
        && info.AllocationBase == first.AllocationBase
        && info.State != MEM_FREE
    {
        end = info.BaseAddress as usize + info.RegionSize;
    }
    end
}

/// The open run holding `[at, at + len)` in `open`, and where the range
/// starts in it.
fn holding(open: &BTreeMap<usize, Arc<Run>>, at: usize, len: usize) -> Option<(Arc<Run>, usize)> {
    for (&start, run) in open.range(..=at).rev() {
        if at - start >= LONGEST {
            break;
        }
        if at + len <= start + run.len {
            return Some((run.clone(), at - start));
        }
    }
    None
}

/// `[at, at + len)` joined with the open runs it overlaps, if that stays
/// within [`LONGEST`]: the range to open and the runs it replaces.
fn joined(open: &BTreeMap<usize, Arc<Run>>, at: usize, len: usize) -> ((usize, usize), Vec<usize>) {
    let (mut start, mut end) = (at, at + len);
    let mut over = Vec::new();
    for (&s, run) in open.range(at.saturating_sub(LONGEST)..at + len) {
        if s + run.len > at {
            over.push(s);
            start = start.min(s);
            end = end.max(s + run.len);
        }
    }
    if end - start > LONGEST {
        return ((at, len), Vec::new());
    }
    ((start, end - start), over)
}

/// The run holding our view's `[at, at + len)`, and where the range starts
/// in it; None while it is not open yet: it is asked of the worker, which
/// opens it off the submitting thread (a later table names it), unless the
/// host refused it before.
pub(crate) fn run_for(at: usize, len: usize) -> Option<(Arc<Run>, usize)> {
    let ask = with_runs(|r| {
        if let Some(found) = holding(&r.open, at, len) {
            return Err(found);
        }
        Ok(!r.refused.contains(&(at, len)) && r.pending.insert((at, len)))
    });
    match ask {
        Err(found) => Some(found),
        Ok(true) => {
            worker(at, len);
            None
        }
        Ok(false) => None,
    }
}

/// Sends `[at, at + len)` to the opening thread.
fn worker(at: usize, len: usize) {
    static WORK: std::sync::OnceLock<Mutex<std::sync::mpsc::Sender<(usize, usize)>>> = std::sync::OnceLock::new();
    let work = WORK.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<(usize, usize)>();
        let _ = std::thread::Builder::new()
            .name("victory-in-place".into())
            .spawn(move || rx.into_iter().for_each(|(at, len)| open_run(at, len)));
        Mutex::new(tx)
    });
    if work.lock().unwrap_or_else(|e| e.into_inner()).send((at, len)).is_err() {
        with_runs(|r| r.pending.remove(&(at, len)));
    }
}

/// Opens our view's `[at, at + len)`, joined with the open runs it
/// overlaps, which it replaces (the page tables naming them keep them
/// until rewritten); on the opening thread.
fn open_run(at: usize, len: usize) {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, PAGE_READWRITE, VirtualAlloc};
    let _opening = OPENING.lock().unwrap_or_else(|e| e.into_inner());
    let started = std::time::Instant::now();
    let ((start, bytes), over, replaced) = with_runs(|r| {
        let (range, over) = joined(&r.open, at, len);
        let replaced: usize = over.iter().filter_map(|s| r.open.get(s)).map(|run| run.len).sum();
        (range, over, replaced)
    });
    if OPEN.load(Ordering::Relaxed) + bytes > budget() + replaced {
        STATS[0].fetch_add(1, Ordering::Relaxed);
        with_runs(|r| r.pending.remove(&(at, len)));
        return;
    }
    let opened = crate::host::host().ok_or_else(|| "no host".to_owned()).and_then(|host| {
        // SAFETY: our own view of sections whose pages the title's views
        // committed: commits nothing new.
        if unsafe { VirtualAlloc(start as *const c_void, bytes, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
            return Err(std::io::Error::last_os_error().to_string());
        }
        open_at(&host.device, start, bytes).map_err(|e| e.to_string())
    });
    let run = match opened {
        Ok((heap, buffer)) => {
            crate::host::name(&buffer, || format!("title memory in place {start:#x}+{bytes:#x}"));
            Arc::new(Run { id: NEXT.fetch_add(1, Ordering::Relaxed), len: bytes, buffer, _heap: heap })
        }
        Err(error) => {
            with_runs(|r| {
                r.pending.remove(&(at, len));
                r.refused.insert((at, len));
            });
            if STATS[1].fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::warn!(at = format_args!("{start:#x}"), len = bytes, %error, "title memory not opened in place: it reads as missing");
            }
            return;
        }
    };
    with_runs(|r| {
        r.pending.remove(&(at, len));
        for s in over {
            if let Some(old) = r.open.remove(&s) {
                OPEN.fetch_sub(old.len, Ordering::Relaxed);
            }
        }
        r.open.insert(start, run);
    });
    OPEN.fetch_add(bytes, Ordering::Relaxed);
    GENERATION.fetch_add(1, Ordering::Relaxed);
    STATS[5].fetch_add(1, Ordering::Relaxed);
    STATS[6].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    tracing::debug!(at = format_args!("{start:#x}"), len = bytes, "title memory opened in place");
}

/// The runtime lets go of our own view `[at, at + len)` of a section:
/// `unmap` unmaps it (nonzero when that worked). With no run over it, now;
/// else the runs are forgotten, and the view stays mapped until no page
/// table names them ([`settle`], at a frame's end): their heaps lock it.
pub fn title_memory_own_gone(at: usize, len: usize, mut unmap: Box<dyn FnMut() -> i32 + Send>) {
    let opening = OPENING.lock().unwrap_or_else(|e| e.into_inner());
    let runs = with_runs(|r| {
        let end = at.saturating_add(len);
        let over: Vec<usize> = r.open.range(..end).filter(|(s, run)| **s + run.len > at).map(|(s, _)| *s).collect();
        let runs: Vec<Arc<Run>> = over.into_iter().filter_map(|s| r.open.remove(&s)).collect();
        r.refused.retain(|&(s, n)| s + n <= at || end <= s);
        runs
    });
    if runs.is_empty() {
        drop(opening);
        unmap();
        return;
    }
    for run in &runs {
        OPEN.fetch_sub(run.len, Ordering::Relaxed);
    }
    GENERATION.fetch_add(1, Ordering::Relaxed);
    with_runs(|r| {
        r.dead.extend(runs.iter().map(|run| run.id));
        r.gone.push(Gone { runs, unmap, frames: 0 });
    });
}

/// Ids of runs over views gone that page tables may still name: the
/// tables drop them at a frame's end, with no submission running.
pub(crate) fn dead() -> Vec<u64> {
    with_runs(|r| r.dead.clone())
}

/// At a frame's end, after the page tables dropped the [`dead`] runs:
/// views gone whose runs nothing else holds are unmapped (tried again for
/// a moment: the system unlocks a heap's pages late); after ~10 s of
/// frames a view still held is left mapped.
pub(crate) fn settle() {
    use windows_sys::Win32::Foundation::{ERROR_INVALID_PARAMETER, GetLastError};
    let gone = with_runs(|r| std::mem::take(&mut r.gone));
    if gone.is_empty() {
        return;
    }
    let mut kept = Vec::new();
    for mut g in gone {
        if g.runs.iter().any(|run| Arc::strong_count(run) > 1) {
            g.frames += 1;
            if g.frames < 600 {
                kept.push(g);
            } else {
                tracing::warn!("a view of title memory held in place stays mapped: a page table still names it");
            }
            continue;
        }
        drop(std::mem::take(&mut g.runs));
        let started = std::time::Instant::now();
        while (g.unmap)() == 0
            // SAFETY: plain reads of this thread's last error.
            && unsafe { GetLastError() } == ERROR_INVALID_PARAMETER
            && started.elapsed() < std::time::Duration::from_millis(100)
        {
            std::thread::yield_now();
        }
    }
    with_runs(|r| {
        // Views gone meanwhile are in `r.gone` already.
        let live: Vec<u64> = kept.iter().chain(&r.gone).flat_map(|g| g.runs.iter().map(|run| run.id)).collect();
        r.dead.retain(|id| live.contains(id));
        r.gone.extend(kept);
    });
}

/// The last page table written named `pages` in place.
pub(crate) fn named(pages: u64) {
    STATS[2].store(pages, Ordering::Relaxed);
}

/// Shaders wrote `pages` pages in place, `withheld` of them where their
/// writes are withheld (in lists that missed, or frames that withhold):
/// in place those land anyway.
pub(crate) fn written(pages: u64, withheld: u64) {
    STATS[3].fetch_add(pages, Ordering::Relaxed);
    STATS[4].fetch_add(withheld, Ordering::Relaxed);
}

/// The frame line, with the backstop on.
pub(crate) fn log_frame() {
    if !backstop() {
        return;
    }
    let written = STATS[3].swap(0, Ordering::Relaxed);
    let withheld = STATS[4].swap(0, Ordering::Relaxed);
    let opened = STATS[5].swap(0, Ordering::Relaxed);
    let took = STATS[6].swap(0, Ordering::Relaxed);
    let named = STATS[2].load(Ordering::Relaxed);
    if named + written + opened == 0 {
        return;
    }
    let (runs, waiting, gone) = with_runs(|r| (r.open.len(), r.pending.len(), r.gone.len()));
    tracing::info!(
        "title memory in place (VICTORY_INPLACE_BACKSTOP): {runs} runs, {:.1} MB open, {named} pages named by the last page table, {written} pages written this frame ({withheld} of them where writes are withheld), {opened} runs opened in {:.1} ms off the submitting thread, {waiting} waiting, {gone} views gone kept mapped; runs refused: {} past VICTORY_INPLACE_MB, {} by the host",
        OPEN.load(Ordering::Relaxed) as f64 / 1e6,
        took as f64 / 1e3,
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

    /// Two sections' views side by side: a run over both is cut where the
    /// first ends.
    #[test]
    fn a_run_over_two_views_is_cut_between_them() {
        let o = 0x7000_0000_0000usize;
        let pages: Vec<(usize, usize)> = (0..5).map(|k| (0x200 + k, o + k * PAGE)).collect();
        let end_of = |at: usize| if at < o + 2 * PAGE { o + 2 * PAGE } else { o + 64 * PAGE };
        let cut = per_view(&pages, end_of);
        let lens: Vec<usize> = cut.iter().map(|r| r.len()).collect();
        assert_eq!(lens, [2, 3]);
        assert_eq!(cut[1][0], (0x202, o + 2 * PAGE));
        // One view: whole. No end known: whole.
        assert_eq!(per_view(&pages, |_| o + 64 * PAGE).len(), 1);
        assert_eq!(per_view(&pages, |_| usize::MAX).len(), 1);
        assert!(per_view(&[], |_| 0).is_empty());
    }

    /// The real allocation end: two views of one section side by side are
    /// two allocations.
    #[test]
    fn allocation_end_is_the_views_end() {
        use windows_sys::Win32::System::Memory as mem;
        const BYTES: usize = 4 * PAGE;
        // SAFETY: plain section creation, closed below.
        let section = unsafe {
            mem::CreateFileMappingW(
                windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE,
                std::ptr::null(),
                mem::PAGE_READWRITE,
                0,
                BYTES as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: a view of our section, unmapped below.
        let at = unsafe { mem::MapViewOfFile(section, mem::FILE_MAP_WRITE, 0, 0, BYTES) }.Value as usize;
        assert_ne!(at, 0);
        assert_eq!(allocation_end(at + PAGE), at + BYTES);
        // SAFETY: our view and handle.
        unsafe {
            mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: at as *mut c_void });
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
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
