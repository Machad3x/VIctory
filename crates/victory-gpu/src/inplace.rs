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
//! overlap, once checked to be views of ours still (a table written over a
//! view since gone may ask for an address holding other memory) and mapped
//! views at all ([`mapped_view`]). No lock is held over an open: a view
//! going meanwhile is kept mapped until the open ends. The heaps stay open
//! while the section lives; when the runtime lets go of our view, its runs
//! are forgotten and the view stays mapped until no page table names them
//! (a frame's end, [`settle`], tried again at the frames after while the
//! system has not unlocked the pages yet).
//!
//! Private memory (`MEM_PRIVATE`: `xmem`'s heaps, the title's
//! `VirtualAlloc`s) is out of the backstop's scope: pages no copy holds
//! stay misses (zeros, writes dropped, writes withheld). A heap over it
//! locks the title's own addresses, so the title's `VirtualProtect` and
//! `VirtualFree` and the write guards fail there (error 87). [`open_run`]
//! refuses any range that is not one mapped view of ours, with one
//! exception, below: a private run, asked as such, wholly in private
//! memory the runtime reported ([`private_open`]).
//!
//! `VICTORY_RT_MEMORY=1` (off by default; it turns the backstop on) covers
//! it too, for the title's ray-tracing structures: its acceleration-
//! structure builder reaches its inputs, scratch and output by addresses
//! it computes, in committed private memory no view ever named (gpu287,
//! gpu288: ~12,000 pages, whole 256 MiB regions), so every builder list
//! read zeros there and all it wrote was dropped. The runtime reports the
//! title's private allocations ([`title_memory_private`]); each page of
//! them that is committed, read-write, held by no copy and no tracker is
//! named in place, in runs of at most [`CHUNK`] pages. The worker opens
//! them when the runtime reports the memory, long before a list reads it;
//! a page table that still finds some asked for and not open waits for the
//! worker ([`wait_private`], bounded per frame): they are there before the
//! submission runs. Nothing is copied: what shaders write
//! there is in title memory at once, for later submissions and for the
//! title's CPU. With it, lists that still miss memory withhold nothing
//! (`crate::memory::withholds`): part of a build in place and part
//! dropped would be worse than either.
//!
//! What the lock on such pages asks:
//! * The title protects, decommits or frees them: each run is registered
//!   with `crate::pinned`, whose callers (the runtime's `VirtualProtect`,
//!   `VirtualFree` and unmapping) let it go first ([`release_private`]:
//!   forgotten, dropped from the page tables once no submission runs, its
//!   heap released), and try again while the system unlocks the pages.
//! * A tracker is made over them (a new buffer's copy, a texture's
//!   staging) and wants to guard them: the runs there are let go first
//!   ([`make_way`], which never waits), and the pages around are named
//!   again. Until the heap is gone the pages stay locked ([`locked`]), and
//!   `crate::watch` sets no guard on them: they count as changed and are
//!   copied at each use, as memory that cannot be guarded.
//!
//! shortcut: such heaps are system memory, as loose copies are; move the
//! hot ones to video memory if the builder's lists run long.

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
    *ON || rt_memory()
}

/// Whether the title's private memory no copy holds is named in place too,
/// and nothing is withheld (`VICTORY_RT_MEMORY=1`).
pub(crate) fn rt_memory() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("VICTORY_RT_MEMORY").is_some_and(|v| v == "1"));
    *ON
}

/// The most bytes open in place at once (`VICTORY_INPLACE_MB`, 1024; 8192
/// with `VICTORY_RT_MEMORY`, the title's private heaps): opening locks the
/// pages into the PC's memory.
fn budget() -> usize {
    static MB: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        let most = if rt_memory() { 8192 } else { 1024 };
        std::env::var("VICTORY_INPLACE_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(most)
    });
    MB.saturating_mul(1 << 20)
}

/// The longest run asked for: one 256 MiB region of the page table.
const MOST: usize = 1 << 28;
/// The longest run opened: asked-for runs joined with the open ones they
/// overlap, up to a raw view's reach (2^27 dwords).
const LONGEST: usize = 2 * MOST;

/// Pages of a run over private memory, which starts and ends where title
/// addresses are multiples of it (8 MiB): a tracker made in one, or a
/// protection of the title's, lets go of that much, and opening it again
/// takes ~2 ms.
pub(crate) const CHUNK: usize = 128;
/// The longest page tables wait in one frame for the worker to open
/// private runs ([`wait_private`]), in microseconds: opening 256 MiB took
/// 70 ms where measured, and runs are opened when their memory is
/// reported, so a table mostly waits for nothing.
const PRIVATE_WAIT_US: u64 = 100_000;
/// Microseconds page tables waited for the worker since the last frame
/// line.
static WAITED_US: AtomicU64 = AtomicU64::new(0);

/// Our own view's `[at, at + len)` open in place: a heap over it and one
/// buffer placed over all of it.
pub(crate) struct Run {
    /// Tells runs apart in the page tables' view pairs (an address may be
    /// opened again after a release).
    pub(crate) id: u64,
    /// Where it starts (our view's address; private memory's own).
    at: usize,
    pub(crate) len: usize,
    pub(crate) buffer: ID3D12Resource,
    /// Over the title's private memory, at its own address: its pages are
    /// [`locked`] while it lives.
    private: bool,
    _heap: ID3D12Heap,
}

impl Drop for Run {
    fn drop(&mut self) {
        if self.private {
            // The heap goes right after: the system unlocks the pages.
            locked_ranges().remove(&(self.at, self.id));
            LOCKED.fetch_sub(1, Ordering::Relaxed);
            // Registered until now, let go or not: a change of the title's
            // meanwhile waits for the tables to drop it.
            unpin(self.id);
        }
    }
}

// SAFETY: D3D12 objects are free-threaded; the rest is plain data.
unsafe impl Send for Run {}
// SAFETY: as above.
unsafe impl Sync for Run {}

/// Our view of a section the runtime let go, held mapped until no page
/// table names a run over it.
struct Gone {
    at: usize,
    len: usize,
    runs: Vec<Arc<Run>>,
    /// Unmaps it; nonzero when that worked.
    unmap: Box<dyn FnMut() -> i32 + Send>,
    /// Frame ends it has waited.
    frames: u32,
    /// A run over it was being opened when it went: kept until that ends.
    opening: bool,
}

/// Frame ends (~10 s) a view gone waits for page tables to let its runs go,
/// and for the system to unlock its pages, before it is left mapped.
const FRAMES_KEPT: u32 = 600;

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
    /// The range the worker is opening now: a view going over it is kept
    /// mapped until the open ends ([`title_memory_own_gone`]).
    opening: Option<(usize, usize)>,
}

static RUNS: Mutex<Option<Runs>> = Mutex::new(None);

fn with_runs<R>(f: impl FnOnce(&mut Runs) -> R) -> R {
    f(RUNS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default())
}

/// Held over a private run's open, to its insertion: a change of the
/// title's there ([`release_private`]) waits for it. Opens over views of
/// ours never take it, nor does a view's going.
static PRIVATE_OPENING: Mutex<()> = Mutex::new(());
/// Runs over private memory open now, asked of the worker and not opened
/// yet, and alive (open, or let go with a page table still naming them).
static PRIVATE_OPEN: AtomicUsize = AtomicUsize::new(0);
static PRIVATE_PENDING: AtomicUsize = AtomicUsize::new(0);
static LOCKED: AtomicUsize = AtomicUsize::new(0);
/// The runs over private memory alive, (start, id) to end: their pages
/// cannot be protected, decommitted or freed.
static LOCKED_RANGES: Mutex<BTreeMap<(usize, u64), usize>> = Mutex::new(BTreeMap::new());
/// Each open private run's registration with `crate::pinned`, by id.
static PINS: Mutex<BTreeMap<u64, crate::pinned::Pin>> = Mutex::new(BTreeMap::new());
/// The title's private allocations the runtime reported, start to end,
/// merged ([`title_memory_private`]).
static PRIVATE: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());
/// Since the start: runs over private memory opened, let go for a tracker,
/// let go for a change of the title's, and page tables that stopped
/// waiting for the worker; pages named in private memory by the last
/// table; since the start again, page tables that waited for the worker
/// and the microseconds they waited.
static RT: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
/// Since the last frame line, of lists with the builder's dispatches:
/// pages written in place, pages written in copies that went back, pages
/// written in copies that were withheld, pages missed.
static BUILDER: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn locked_ranges() -> order::Ranges {
    order::Ranges::new(LOCKED_RANGES.lock().unwrap_or_else(|e| e.into_inner()))
}

/// The lock order around title memory in place, asserted in debug builds:
/// * `crate::watch`'s guard lock is taken before [`LOCKED_RANGES`]
///   (`protect_noting` asks [`locked`] under it), never after: nothing
///   takes the guard lock while holding the ranges.
/// * The page tables' lock is held while waiting for the opening thread
///   ([`wait_private`]), so that thread never takes it.
pub(crate) mod order {
    use std::collections::BTreeMap;
    use std::sync::MutexGuard;

    thread_local! {
        /// This thread holds [`super::LOCKED_RANGES`].
        static RANGES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        /// This thread is the opening thread.
        static WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    type Map = BTreeMap<(usize, u64), usize>;

    /// [`super::LOCKED_RANGES`] held, known to [`guards`].
    pub(crate) struct Ranges(MutexGuard<'static, Map>);

    impl Ranges {
        pub(super) fn new(guard: MutexGuard<'static, Map>) -> Ranges {
            RANGES.set(true);
            Ranges(guard)
        }
    }

    impl Drop for Ranges {
        fn drop(&mut self) {
            RANGES.set(false);
        }
    }

    impl std::ops::Deref for Ranges {
        type Target = Map;
        fn deref(&self) -> &Map {
            &self.0
        }
    }

    impl std::ops::DerefMut for Ranges {
        fn deref_mut(&mut self) -> &mut Map {
            &mut self.0
        }
    }

    /// Whether this thread holds the ranges.
    pub(super) fn ranges_held() -> bool {
        RANGES.get()
    }

    /// This thread is the opening thread from now on.
    pub(super) fn worker() {
        WORKER.set(true);
    }

    /// `crate::watch`'s guard lock is about to be taken.
    #[inline]
    pub(crate) fn guards() {
        debug_assert!(!RANGES.get(), "the guard lock taken while holding the ranges locked in place");
    }

    /// The page tables' lock is about to be taken.
    #[inline]
    pub(crate) fn tables() {
        debug_assert!(!WORKER.get(), "the page tables' lock taken on the opening thread, which its holder waits for");
    }

    #[cfg(test)]
    mod tests {
        /// The ranges count as held exactly while their guard lives, and
        /// only on the thread holding them.
        #[test]
        fn the_ranges_are_held_while_their_guard_lives() {
            assert!(!super::ranges_held());
            let held = super::super::locked_ranges();
            assert!(super::ranges_held());
            assert!(!std::thread::spawn(super::ranges_held).join().unwrap());
            drop(held);
            assert!(!super::ranges_held());
            super::guards(); // in order: the ranges are not held
        }

        /// Taking the guard lock under the ranges is out of order.
        #[test]
        #[cfg(debug_assertions)]
        #[should_panic(expected = "while holding the ranges")]
        fn the_guard_lock_under_the_ranges_is_refused() {
            let _held = super::super::locked_ranges();
            super::guards();
        }

        /// The opening thread never takes the page tables' lock.
        #[test]
        #[cfg(debug_assertions)]
        fn the_opening_thread_may_not_take_the_tables() {
            super::tables();
            let refused = std::thread::spawn(|| {
                super::worker();
                std::panic::catch_unwind(super::tables).is_err()
            });
            assert!(refused.join().unwrap());
        }
    }
}

/// Whether title page `page` lies in a run over private memory that is
/// alive: it cannot be guarded (`crate::watch` sets none on it). One atomic
/// load while there is none.
pub(crate) fn locked(page: usize) -> bool {
    LOCKED.load(Ordering::Relaxed) != 0 && in_ranges(&locked_ranges(), page * PAGE)
}

/// Whether one of `ranges` ((start, id) to end, none longer than
/// [`LONGEST`]) holds the page at `at`.
fn in_ranges(ranges: &BTreeMap<(usize, u64), usize>, at: usize) -> bool {
    ranges.range((at.saturating_sub(LONGEST), 0)..=(at, u64::MAX)).any(|(_, &end)| at + PAGE <= end)
}
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
/// most [`MOST`], and none across a title page that is a multiple of
/// `chunk` (0: no such cut).
pub(crate) fn runs(pages: &[(usize, usize)], chunk: usize) -> Vec<&[(usize, usize)]> {
    let mut out = Vec::new();
    let mut start = 0;
    for i in 1..=pages.len() {
        let joined = i < pages.len()
            && pages[i].0 == pages[i - 1].0 + 1
            && pages[i].1 == pages[i - 1].1 + PAGE
            && (chunk == 0 || !pages[i].0.is_multiple_of(chunk))
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

/// Whether `[at, at + len)` lies in one mapped view (`MEM_MAPPED`, reserved
/// or committed): never private memory, which the backstop leaves alone.
pub(crate) fn mapped_view(at: usize, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let end = at.saturating_add(len);
    let (mut p, mut base) = (at, None);
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(p as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) } == 0
            || info.Type != MEM_MAPPED
            || *base.get_or_insert(info.AllocationBase) != info.AllocationBase
        {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    len != 0
}

/// The one exception to opening only views of ours (`VICTORY_RT_MEMORY`):
/// whether `[at, at + len)` lies wholly in private memory the runtime
/// reported ([`title_memory_private`]) and is committed private read-write
/// memory now, throughout. Such a run is registered with `crate::pinned`
/// before its heap opens, so that the title's changes let it go first.
pub(crate) fn private_open(at: usize, len: usize) -> bool {
    let end = at.saturating_add(len);
    // Reported only with `VICTORY_RT_MEMORY`.
    if len == 0 || !PRIVATE.lock().unwrap_or_else(|e| e.into_inner()).range(..=at).next_back().is_some_and(|(_, &e)| e >= end) {
        return false;
    }
    // One allocation throughout: a heap is opened on one (the host refuses
    // a range across two with E_INVALIDARG, gpu308).
    let mut p = at;
    while p < end {
        let (next, opens) = region(p);
        if !opens {
            return false;
        }
        p = next;
    }
    allocation_end(at) >= end
}

/// Pages of private memory to name in place (in page order, at their own
/// addresses) cut into runs: at most [`CHUNK`] pages, ending where a chunk
/// does, and each within one allocation.
pub(crate) fn private_runs(pages: &[(usize, usize)]) -> Vec<&[(usize, usize)]> {
    runs(pages, CHUNK).into_iter().flat_map(|run| per_view(run, allocation_end)).collect()
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
/// opens it off the submitting thread (a later table names it, or this one
/// after [`wait_private`]), unless the host refused it before. `private`:
/// the title's own memory at its own address.
pub(crate) fn run_for(at: usize, len: usize, private: bool) -> Option<(Arc<Run>, usize)> {
    let ask = with_runs(|r| {
        if let Some(found) = holding(&r.open, at, len) {
            return Err(found);
        }
        Ok(!r.refused.contains(&(at, len)) && r.pending.insert((at, len)))
    });
    match ask {
        Err(found) => Some(found),
        Ok(true) => {
            if private {
                PRIVATE_PENDING.fetch_add(1, Ordering::Relaxed);
            }
            worker(at, len, private);
            None
        }
        Ok(false) => None,
    }
}

/// Waits for the worker to open the private runs asked of it: a page
/// table written for a submission names them before it runs (the
/// builder's first list reads no zeros). All the tables of a frame wait
/// [`PRIVATE_WAIT_US`] at most between them; one that stops waiting is
/// counted, and the pages not open miss for its submission. True if none
/// is left waiting. Called with the page tables' lock held, which the
/// worker never takes ([`order`]).
pub(crate) fn wait_private() -> bool {
    debug_assert!(!order::ranges_held(), "waiting for the opening thread while holding the ranges");
    if PRIVATE_PENDING.load(Ordering::Relaxed) == 0 {
        return true;
    }
    let started = std::time::Instant::now();
    let left = PRIVATE_WAIT_US.saturating_sub(WAITED_US.load(Ordering::Relaxed));
    let done = wait_pending(std::time::Duration::from_micros(left));
    let waited = started.elapsed().as_micros() as u64;
    WAITED_US.fetch_add(waited, Ordering::Relaxed);
    RT[5].fetch_add(1, Ordering::Relaxed);
    RT[6].fetch_add(waited, Ordering::Relaxed);
    if !done {
        RT[3].fetch_add(1, Ordering::Relaxed);
    }
    done
}

/// Signalled (under [`SETTLED_LOCK`]) when no private run is left asked of
/// the worker.
static SETTLED: std::sync::Condvar = std::sync::Condvar::new();
static SETTLED_LOCK: Mutex<()> = Mutex::new(());

/// A private run asked of the worker is no longer waited for.
fn asked_done() {
    if PRIVATE_PENDING.fetch_sub(1, Ordering::Relaxed) == 1 {
        // Taken so that a waiter between its test and its wait is not missed.
        let _settled = SETTLED_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        SETTLED.notify_all();
    }
}

/// Waits until no private run is left asked of the worker, for `limit` at
/// most; woken by [`asked_done`], no polling.
/// True if none is left.
fn wait_pending(limit: std::time::Duration) -> bool {
    let pending = |_: &mut ()| PRIVATE_PENDING.load(Ordering::Relaxed) != 0;
    let settled = SETTLED_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (mut settled, _) = SETTLED.wait_timeout_while(settled, limit, pending).unwrap_or_else(|e| e.into_inner());
    !pending(&mut settled)
}
/// Sends `[at, at + len)` to the opening thread.
fn worker(at: usize, len: usize, private: bool) {
    type Work = (usize, usize, bool);
    static WORK: std::sync::OnceLock<Mutex<std::sync::mpsc::Sender<Work>>> = std::sync::OnceLock::new();
    let work = WORK.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Work>();
        let _ = std::thread::Builder::new()
            .name("victory-in-place".into())
            .spawn(move || {
                order::worker();
                rx.into_iter().for_each(|(at, len, private)| open_run(at, len, private))
            });
        Mutex::new(tx)
    });
    if work.lock().unwrap_or_else(|e| e.into_inner()).send((at, len, private)).is_err() {
        with_runs(|r| r.pending.remove(&(at, len)));
        if private {
            asked_done();
        }
    }
}

/// A private run asked of the worker is no longer waited for, however
/// [`open_run`] ends.
struct Asked(bool);

impl Drop for Asked {
    fn drop(&mut self) {
        if self.0 {
            asked_done();
        }
    }
}

/// Opens our view's `[at, at + len)`, joined with the open runs it
/// overlaps, which it replaces (the page tables naming them keep them
/// until rewritten); on the opening thread. `private`: the title's own
/// memory, committed by the title (nothing is committed here), registered
/// with `crate::pinned` so that the title's changes let it go first.
fn open_run(at: usize, len: usize, private: bool) {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, PAGE_READWRITE, VirtualAlloc};
    let _asked = Asked(private);
    let _opening = private.then(|| PRIVATE_OPENING.lock().unwrap_or_else(|e| e.into_inner()));
    let started = std::time::Instant::now();
    // Set before the checks: a view going from here on stays mapped until
    // this open ends; one gone before has its views forgotten already
    // (the runtime forgets them before letting go of the section).
    with_runs(|r| r.opening = Some((at, len)));
    // Asked from a table written over views since gone: the address may
    // hold other memory now, which a heap would lock. Private memory only
    // as the one exception, asked as such ([`private_open`]).
    let allowed = if private { private_open(at, len) } else { crate::watch::own_views_show(at, len) && mapped_view(at, len) };
    if !allowed {
        with_runs(|r| finish(r, at, len, None));
        return;
    }
    let ((start, bytes), over, replaced) = with_runs(|r| {
        let (range, over) = joined(&r.open, at, len);
        let replaced: usize = over.iter().filter_map(|s| r.open.get(s)).map(|run| run.len).sum();
        (range, over, replaced)
    });
    if OPEN.load(Ordering::Relaxed) + bytes > budget() + replaced {
        STATS[0].fetch_add(1, Ordering::Relaxed);
        with_runs(|r| finish(r, at, len, None));
        return;
    }
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    if private {
        // Locked and registered before the heap is opened: a guard is not
        // set there meanwhile, and a change of the title's there waits for
        // the open to end ([`release_private`] takes [`PRIVATE_OPENING`]) and lets
        // the run go, instead of meeting the pages locked with nothing to
        // release.
        locked_ranges().insert((start, id), start + bytes);
        LOCKED.fetch_add(1, Ordering::Relaxed);
        let pin = crate::pinned::pin(start, bytes, Box::new(move || release_private(start, id)));
        PINS.lock().unwrap_or_else(|e| e.into_inner()).insert(id, pin);
        #[cfg(test)]
        stall::opening(start, bytes);
        // Checked again under the registration: the title may have changed
        // the memory between the check above and it (nothing to let go
        // then). From here on its changes through the runtime wait for this
        // open and let the run go.
        if !private_open(start, bytes) {
            locked_ranges().remove(&(start, id));
            LOCKED.fetch_sub(1, Ordering::Relaxed);
            unpin(id);
            with_runs(|r| finish(r, at, len, None));
            return;
        }
    }
    let opened = crate::host::host().ok_or_else(|| "no host".to_owned()).and_then(|host| {
        // SAFETY: our own view of sections whose pages the title's views
        // committed: commits nothing new.
        if !private && unsafe { VirtualAlloc(start as *const c_void, bytes, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
            return Err(std::io::Error::last_os_error().to_string());
        }
        let opened = open_at(&host.device, start, bytes).map_err(|e| e.to_string())?;
        // And once open, when nothing can change it any more: a commit with
        // another protection (`XMemVirtualAlloc` tries the host first) may
        // have come before the heap. Dropped here, before the run lets go.
        if private && !private_open(start, bytes) {
            return Err("the title changed it while it was opened".to_owned());
        }
        Ok(opened)
    });
    let run = match opened {
        Ok((heap, buffer)) => {
            crate::host::name(&buffer, || format!("title memory in place {start:#x}+{bytes:#x}"));
            Arc::new(Run { id, at: start, len: bytes, buffer, private, _heap: heap })
        }
        Err(error) => {
            if private {
                locked_ranges().remove(&(start, id));
                LOCKED.fetch_sub(1, Ordering::Relaxed);
                unpin(id);
            }
            with_runs(|r| {
                finish(r, at, len, None);
                r.refused.insert((at, len));
            });
            if STATS[1].fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::warn!(at = format_args!("{start:#x}"), len = bytes, %error, "title memory not opened in place: it reads as missing");
            }
            return;
        }
    };
    STATS[5].fetch_add(1, Ordering::Relaxed);
    STATS[6].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    let Some(old) = with_runs(|r| finish(r, at, len, Some((start, run, over)))) else {
        return;
    };
    if private {
        // The runs it replaces stay registered until their heaps go (their
        // drop): a change of the title's there drops them from the tables.
        let replaced = old.iter().filter(|run| run.private).count();
        PRIVATE_OPEN.fetch_sub(replaced, Ordering::Relaxed);
        PRIVATE_OPEN.fetch_add(1, Ordering::Relaxed);
        RT[0].fetch_add(1, Ordering::Relaxed);
    }
    drop(old);
    GENERATION.fetch_add(1, Ordering::Relaxed);
    tracing::debug!(at = format_args!("{start:#x}"), len = bytes, private, "title memory opened in place");
}

/// The open of `[at, at + len)` ended, with the run opened (at `start`, and
/// the open runs it replaces) or not: it is open now, or, its view gone
/// meanwhile, held by that view's going until no one else holds it. The
/// runs it replaced, if it is open.
fn finish(r: &mut Runs, at: usize, len: usize, opened: Option<(usize, Arc<Run>, Vec<usize>)>) -> Option<Vec<Arc<Run>>> {
    r.opening = None;
    r.pending.remove(&(at, len));
    let mut went = false;
    for g in r.gone.iter_mut().filter(|g| g.opening) {
        g.opening = false;
        if !went && let Some((_, run, _)) = &opened {
            r.dead.push(run.id);
            g.runs.push(run.clone());
        }
        went = true;
    }
    let (start, run, over) = opened.filter(|_| !went)?;
    let mut old: Vec<Arc<Run>> = over.iter().filter_map(|s| r.open.remove(s)).collect();
    OPEN.fetch_add(run.len, Ordering::Relaxed);
    // A run at the same address that the join left out (past LONGEST).
    old.extend(r.open.insert(start, run));
    for run in &old {
        OPEN.fetch_sub(run.len, Ordering::Relaxed);
    }
    // Private runs replaced: the tables naming them drop them when the
    // title changes the memory ([`release_private`]).
    r.dead.extend(old.iter().filter(|run| run.private).map(|run| run.id));
    Some(old)
}

/// Tests: an open over the armed address stops once its run is registered
/// (locked, pinned, [`PRIVATE_OPENING`] held, heap not open) until a release of it
/// has begun ([`release_private`]), which then waits for the open: the
/// title's free lands in the middle of the open, every time.
#[cfg(test)]
mod stall {
    use std::sync::{Condvar, Mutex};

    /// The armed address (0: none), whether an open over it is stopped,
    /// whether its release has begun, and the stopped run's start (a join
    /// may start it below the address).
    static STATE: Mutex<(usize, bool, bool, usize)> = Mutex::new((0, false, false, 0));
    static MOVED: Condvar = Condvar::new();
    /// Far past any step's time: a test that never releases fails, and the
    /// worker goes on, instead of both hanging.
    const MOST: std::time::Duration = std::time::Duration::from_secs(10);

    fn state() -> std::sync::MutexGuard<'static, (usize, bool, bool, usize)> {
        STATE.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(super) fn arm(at: usize) {
        *state() = (at, false, false, 0);
    }

    /// On the opening thread, the run `[start, start + bytes)` registered
    /// and not opened.
    pub(super) fn opening(start: usize, bytes: usize) {
        let mut s = state();
        if s.0 == 0 || !(start..start + bytes).contains(&s.0) {
            return;
        }
        (s.1, s.3) = (true, start);
        MOVED.notify_all();
        drop(MOVED.wait_timeout_while(s, MOST, |s| !s.2).unwrap_or_else(|e| e.into_inner()));
    }

    /// A release of the run at `start` begins.
    pub(super) fn releasing(start: usize) {
        let mut s = state();
        if s.1 && s.3 == start {
            s.2 = true;
            MOVED.notify_all();
        }
    }

    /// Waits for the armed open to stop; true if it did.
    pub(super) fn stopped() -> bool {
        MOVED.wait_timeout_while(state(), MOST, |s| !s.1).unwrap_or_else(|e| e.into_inner()).0.1
    }
}
/// The private run `id` is no longer registered with `crate::pinned`.
fn unpin(id: u64) {
    let pin = PINS.lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
    if let Some(pin) = pin {
        pin.forget();
    }
}

/// The open private run `id` at `start` is forgotten: no page table names
/// it anew, those naming it drop it ([`dead`]), and its pages stay
/// [`locked`] until its heap goes. The run, if it was open.
fn forget_private(start: usize, id: u64) -> Option<Arc<Run>> {
    let run = with_runs(|r| {
        if r.open.get(&start).is_none_or(|run| run.id != id) {
            return None;
        }
        r.dead.push(id);
        r.open.remove(&start)
    })?;
    OPEN.fetch_sub(run.len, Ordering::Relaxed);
    PRIVATE_OPEN.fetch_sub(1, Ordering::Relaxed);
    GENERATION.fetch_add(1, Ordering::Relaxed);
    Some(run)
}

/// `crate::pinned`'s release of the private run `id` at `start`, before
/// the title protects, decommits or frees memory in it: forgotten, and
/// dropped from the page tables once no submission runs (waited for: a
/// submission reading memory the title frees is the title's race), so
/// that its heap goes and the system unlocks the pages. The caller tries
/// its change again while that settles.
fn release_private(start: usize, id: u64) {
    // Released by the registry: nothing to forget there.
    PINS.lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
    // An open under way ends first (it registered before opening), and its
    // run is forgotten here. Not held while waiting for the tables below:
    // their holder may wait for the opening thread.
    #[cfg(test)]
    stall::releasing(start);
    let run = {
        let _opening = PRIVATE_OPENING.lock().unwrap_or_else(|e| e.into_inner());
        forget_private(start, id)
    };
    if let Some(run) = run {
        RT[2].fetch_add(1, Ordering::Relaxed);
        drop(run);
    }
    // It, or a run let go for a tracker whose heap a table still holds, and
    // the runs it replaced.
    crate::memory::drop_runs_wait(&dead());
}

/// A tracker is about to guard `[at, at + len)`: the open runs over
/// private memory holding part of it are let go (their pages cannot be
/// protected). Never waits: the page tables drop them at their next
/// rewrite, and until the heaps are gone the pages stay [`locked`] and
/// unguarded; their registration with `crate::pinned` stays until then, so
/// that a change of the title's there still waits for the heaps to go. One
/// atomic load while none is open.
pub(crate) fn make_way(at: usize, len: usize) {
    if PRIVATE_OPEN.load(Ordering::Relaxed) == 0 || len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let hit: Vec<(usize, u64)> = with_runs(|r| {
        r.open
            .range(at.saturating_sub(LONGEST)..end)
            .filter(|(s, run)| run.private && **s + run.len > at)
            .map(|(&s, run)| (s, run.id))
            .collect()
    });
    for (start, id) in hit {
        if forget_private(start, id).is_some() {
            RT[1].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// The runtime made or committed private title memory at `[at, at + len)`
/// (`XMemVirtualAlloc`), or shaders missed committed private memory there:
/// with `VICTORY_RT_MEMORY`, its pages no copy holds are named in place
/// from the next submission on. Only below 2^40, where page tables reach.
pub fn title_memory_private(at: usize, len: usize) {
    if !rt_memory() || len == 0 || at.saturating_add(len) > 1 << 40 {
        return;
    }
    let (start, end) = (at & !(PAGE - 1), (at + len).next_multiple_of(PAGE));
    merge(&mut PRIVATE.lock().unwrap_or_else(|e| e.into_inner()), start, end);
    // What the host refused there may open now (committed, unprotected).
    with_runs(|r| unrefuse(r, start, end));
    let regions: Vec<usize> = (start >> 28..=(end - 1) >> 28).collect();
    crate::memory::want(&regions);
    // Opened now, on the worker, so that the page table of the first
    // submission reading it waits for nothing: the pages no tracker holds
    // (fresh memory has no copy; a copy made later lets its run go).
    let tracked = crate::watch::tracked_pages(start / PAGE, end / PAGE);
    let candidates: Vec<usize> = (start / PAGE..end / PAGE).filter(|page| tracked.binary_search(page).is_err()).collect();
    for run in runs(&private_pages(&candidates), CHUNK) {
        let _ = run_for(run[0].1, run.len() * PAGE, true);
    }
    // Committed since the last table, maybe: asked again.
    GENERATION.fetch_add(1, Ordering::Relaxed);
}

/// Shaders missed `[at, at + len)`, `private` telling whether it is
/// private memory (asked only for memory not reported yet). Memory not
/// reported is reported ([`title_memory_private`]). Memory reported already
/// was not named (committed or made read-write since the last table, not
/// open yet, or refused by the host): the page tables look again, once a
/// frame. What the host refused stays refused until the title commits there
/// again ([`title_memory_private`], [`title_memory_committed`]): asking it
/// again at each miss made every table wait for the same refusal. Each miss
/// came to O(pages) work and a table rewrite per submission before (~12,000
/// missed pages, 256 MiB regions); this is one lookup a miss.
pub(crate) fn missed_private(at: usize, len: usize, private: impl FnOnce() -> bool) {
    if !rt_memory() || len == 0 || at.saturating_add(len) > 1 << 40 {
        return;
    }
    let (start, end) = (at & !(PAGE - 1), (at + len).next_multiple_of(PAGE));
    if !holds(&PRIVATE.lock().unwrap_or_else(|e| e.into_inner()), start, end) {
        if private() {
            title_memory_private(at, len);
        }
        return;
    }
    if !MISSED_AGAIN.swap(true, Ordering::Relaxed) {
        GENERATION.fetch_add(1, Ordering::Relaxed);
    }
}

/// The title committed `[at, at + len)` with its own `VirtualAlloc`: over
/// private memory reported already, what the host refused may open now, and
/// the page tables look again. Nothing new is reported (memory the title
/// allocates itself is reported when shaders miss it). Cheap: two lookups.
pub fn title_memory_committed(at: usize, len: usize) {
    if !rt_memory() || len == 0 || at.saturating_add(len) > 1 << 40 {
        return;
    }
    let (start, end) = (at & !(PAGE - 1), (at + len).next_multiple_of(PAGE));
    let reported = PRIVATE.lock().unwrap_or_else(|e| e.into_inner());
    let touches = reported.range(..end).next_back().is_some_and(|(_, &e)| e > start);
    drop(reported);
    if touches {
        with_runs(|r| unrefuse(r, start, end));
        GENERATION.fetch_add(1, Ordering::Relaxed);
    }
}

/// A miss of reported memory rewrote the page tables this frame
/// ([`missed_private`]).
static MISSED_AGAIN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether `ranges` (start to end, disjoint, merged) hold `[start, end)`
/// whole.
fn holds(ranges: &BTreeMap<usize, usize>, start: usize, end: usize) -> bool {
    ranges.range(..=start).next_back().is_some_and(|(_, &e)| e >= end)
}

/// What the host refused over `[start, end)` may be asked again.
fn unrefuse(r: &mut Runs, start: usize, end: usize) {
    // No refused run is longer than [`LONGEST`].
    let hit: Vec<(usize, usize)> = r
        .refused
        .range((start.saturating_sub(LONGEST), 0)..(end, 0))
        .filter(|&&(s, n)| s + n > start)
        .copied()
        .collect();
    for key in hit {
        r.refused.remove(&key);
    }
}

/// `[start, end)` joins `ranges` (start to end, disjoint), merged with
/// those it touches.
fn merge(ranges: &mut BTreeMap<usize, usize>, mut start: usize, mut end: usize) {
    let touching: Vec<usize> = ranges
        .range(..=end)
        .rev()
        .take_while(|&(_, &e)| e >= start)
        .map(|(&s, _)| s)
        .collect();
    for s in touching {
        if let Some(e) = ranges.remove(&s) {
            (start, end) = (start.min(s), end.max(e));
        }
    }
    ranges.insert(start, end);
}

/// The pages `first..last` (page numbers) of the private allocations
/// reported, in order.
pub(crate) fn reported(first: usize, last: usize) -> Vec<usize> {
    let ranges = PRIVATE.lock().unwrap_or_else(|e| e.into_inner());
    let (from, to) = (first * PAGE, last * PAGE);
    let before = ranges.range(..from).next_back().filter(|&(_, &e)| e > from);
    before
        .into_iter()
        .chain(ranges.range(from..to))
        .flat_map(|(&s, &e)| s.max(from) / PAGE..e.min(to) / PAGE)
        .collect()
}

/// Those of `candidates` (page numbers in order, held by no copy and no
/// tracker) that can be named in place, as (page, address): the ones in a
/// run over private memory open already, and the ones wholly in committed
/// private memory that is read-write now and not [`locked`] by a run let
/// go (a heap over those would outlive the change that let it go).
pub(crate) fn private_pages(candidates: &[usize]) -> Vec<(usize, usize)> {
    let (Some(&first), Some(&last)) = (candidates.first(), candidates.last()) else {
        return Vec::new();
    };
    let (from, to) = (first * PAGE, (last + 1) * PAGE);
    // Only the runs reaching the candidates, merged: one binary search a
    // page (65,536 pages for a 256 MiB region).
    let open = spans(with_runs(|r| {
        r.open
            .range(from.saturating_sub(LONGEST)..to)
            .filter(|(_, run)| run.private)
            .map(|(&s, run)| (s, s + run.len))
            .collect()
    }));
    let alive = spans(locked_ranges().range((from.saturating_sub(LONGEST), 0)..(to, 0)).map(|(&(s, _), &e)| (s, e)).collect());
    let covered = |page: usize| within(&open, page * PAGE);
    let picked = pick(candidates, covered, region);
    picked.into_iter().filter(|&(page, at)| covered(page) || !within(&alive, at)).collect()
}

/// `ranges` (start, end) sorted and merged where they overlap or touch.
fn spans(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (s, e) in ranges {
        match out.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => out.push((s, e)),
        }
    }
    out
}

/// Whether the page at `at` lies wholly in one of `spans` ([`spans`]).
fn within(spans: &[(usize, usize)], at: usize) -> bool {
    let i = spans.partition_point(|&(s, _)| s <= at);
    i > 0 && at + PAGE <= spans[i - 1].1
}

/// [`private_pages`], with `covered` (a page lies in an open run) and
/// `region` (the end of the memory of one kind holding an address, and
/// whether that kind can be opened) given. A page reaching past its
/// region is left out: a heap needs one protection throughout.
fn pick(candidates: &[usize], covered: impl Fn(usize) -> bool, mut region: impl FnMut(usize) -> (usize, bool)) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < candidates.len() {
        let page = candidates[i];
        if covered(page) {
            out.push((page, page * PAGE));
            i += 1;
            continue;
        }
        let (end, opens) = region(page * PAGE);
        // Every candidate wholly below the region's end shares the answer.
        let whole = candidates[i..].partition_point(|&p| (p + 1) * PAGE <= end);
        out.extend(candidates[i..i + whole].iter().filter(|&&p| opens || covered(p)).map(|&p| (p, p * PAGE)));
        // None whole: the page straddles the end, and is left out.
        i += whole.max(1);
    }
    out
}

/// The end of the memory of one state, type and protection holding `at`,
/// and whether it is committed private read-write memory.
fn region(at: usize) -> (usize, bool) {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    if unsafe { VirtualQuery(at as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
        return (at + PAGE, false);
    }
    let end = (info.BaseAddress as usize + info.RegionSize).max(at + 1);
    (end, info.State == MEM_COMMIT && info.Type == MEM_PRIVATE && info.Protect == PAGE_READWRITE)
}

/// A list with dispatches of the title's acceleration-structure builder
/// ran: the pages its shaders wrote in place, wrote in copies that went
/// back into title memory, wrote in copies that were withheld, and missed.
pub(crate) fn builder_list(in_place: u64, back: u64, withheld: u64, missed: u64) {
    for (n, x) in BUILDER.iter().zip([in_place, back, withheld, missed]) {
        n.fetch_add(x, Ordering::Relaxed);
    }
}

/// The runtime lets go of our own view `[at, at + len)` of a section:
/// `unmap` unmaps it (nonzero when that worked). With no run over it, now;
/// else the runs are forgotten, and the view stays mapped until no page
/// table names them ([`settle`], at a frame's end): their heaps lock it.
/// Never waits: a run being opened over it meanwhile keeps it mapped until
/// that open ends.
pub fn title_memory_own_gone(at: usize, len: usize, unmap: Box<dyn FnMut() -> i32 + Send>) {
    let end = at.saturating_add(len);
    let now = with_runs(|r| {
        let over: Vec<usize> = r.open.range(..end).filter(|(s, run)| **s + run.len > at).map(|(s, _)| *s).collect();
        let runs: Vec<Arc<Run>> = over.into_iter().filter_map(|s| r.open.remove(&s)).collect();
        r.refused.retain(|&(s, n)| s + n <= at || end <= s);
        let opening = r.opening.is_some_and(|(s, n)| s < end && at < s + n);
        if runs.is_empty() && !opening {
            return Some(unmap);
        }
        if !runs.is_empty() {
            for run in &runs {
                OPEN.fetch_sub(run.len, Ordering::Relaxed);
            }
            GENERATION.fetch_add(1, Ordering::Relaxed);
        }
        r.dead.extend(runs.iter().map(|run| run.id));
        r.gone.push(Gone { at, len, runs, unmap, frames: 0, opening });
        None
    });
    if let Some(mut unmap) = now {
        unmap();
    }
}

/// Ids of runs over views gone that page tables may still name: the
/// tables drop them at a frame's end, with no submission running.
pub(crate) fn dead() -> Vec<u64> {
    with_runs(|r| r.dead.clone())
}

/// At a frame's end, after the page tables dropped the [`dead`] runs: the
/// heaps of views gone whose runs nothing else holds are let go (off this
/// thread), and such views with no run left are unmapped. Nothing here
/// waits: a view the system has not unlocked yet is tried again at the
/// next frame's end; after [`FRAMES_KEPT`] a view is left mapped.
pub(crate) fn settle() {
    use windows_sys::Win32::Foundation::{ERROR_INVALID_PARAMETER, GetLastError};
    let ready = with_runs(|r| {
        // Private runs let go or replaced stay [`dead`] while their heaps
        // live (until the tables dropped them). Read under the runs' lock
        // (RUNS before LOCKED_RANGES, as a run's drop under it takes them):
        // a run forgotten after a read outside it would be lost from `dead`.
        let alive: Vec<u64> = locked_ranges().keys().map(|&(_, id)| id).collect();
        let mut ready = Vec::new();
        let mut i = 0;
        while i < r.gone.len() {
            let g = &mut r.gone[i];
            if g.opening {
                i += 1;
            } else if g.runs.iter().any(|run| Arc::strong_count(run) > 1) {
                g.frames += 1;
                if g.frames < FRAMES_KEPT {
                    i += 1;
                } else {
                    tracing::warn!(at = format_args!("{:#x}", g.at), g.len, "a view of title memory held in place stays mapped: a page table still names it");
                    r.gone.swap_remove(i);
                }
            } else if !g.runs.is_empty() {
                // Released on a thread of its own (kernel calls); unmapped
                // from the next frame's end on.
                crate::host::release_later(Box::new(std::mem::take(&mut g.runs)));
                i += 1;
            } else {
                ready.push(r.gone.swap_remove(i));
            }
        }
        let live: Vec<u64> = r.gone.iter().flat_map(|g| g.runs.iter().map(|run| run.id)).collect();
        r.dead.retain(|id| live.contains(id) || alive.contains(id));
        ready
    });
    // Unmapped outside the lock (the unmap waits for writes through our views).
    for mut g in ready {
        if (g.unmap)() != 0 {
            continue;
        }
        // SAFETY: plain read of this thread's last error.
        let error = unsafe { GetLastError() };
        g.frames += 1;
        if error == ERROR_INVALID_PARAMETER && g.frames < FRAMES_KEPT {
            // The system unlocks a released heap's pages late.
            with_runs(|r| r.gone.push(g));
        } else {
            tracing::warn!(at = format_args!("{:#x}", g.at), g.len, error, "a view of title memory held in place could not be unmapped: left mapped");
        }
    }
}

/// The last page table written named `pages` in place, `private` of them
/// in private memory.
pub(crate) fn named(pages: u64, private: u64) {
    STATS[2].store(pages, Ordering::Relaxed);
    RT[4].store(private, Ordering::Relaxed);
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
    MISSED_AGAIN.store(false, Ordering::Relaxed);
    let written = STATS[3].swap(0, Ordering::Relaxed);
    let withheld = STATS[4].swap(0, Ordering::Relaxed);
    let opened = STATS[5].swap(0, Ordering::Relaxed);
    let took = STATS[6].swap(0, Ordering::Relaxed);
    let named = STATS[2].load(Ordering::Relaxed);
    let builder = BUILDER.each_ref().map(|n| n.swap(0, Ordering::Relaxed));
    // A frame's allowance of waiting starts anew.
    let waited = WAITED_US.swap(0, Ordering::Relaxed);
    if rt_memory() && (named > 0 || builder.iter().any(|&n| n > 0)) {
        // Persisted: in place, or copied back. Lost: withheld or missed.
        tracing::info!(
            "ray tracing memory (VICTORY_RT_MEMORY): private memory in place: {} pages named, {} runs open, so far {} opened, {} let go for trackers, {} for the title's changes; page tables waited for the opening thread {:.1} ms this frame, so far {} times and {:.1} ms, {} of them stopped waiting; builder lists this frame: {} pages written in place, {} copied back ({:.1} MB reached the title), {} withheld, {} missed",
            RT[4].load(Ordering::Relaxed),
            PRIVATE_OPEN.load(Ordering::Relaxed),
            RT[0].load(Ordering::Relaxed),
            RT[1].load(Ordering::Relaxed),
            RT[2].load(Ordering::Relaxed),
            waited as f64 / 1e3,
            RT[5].load(Ordering::Relaxed),
            RT[6].load(Ordering::Relaxed) as f64 / 1e3,
            RT[3].load(Ordering::Relaxed),
            builder[0],
            builder[1],
            ((builder[0] + builder[1]) << 16) as f64 / 1e6,
            builder[2],
            builder[3]
        );
    }
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
        let found = runs(&pages, 0);
        let lens: Vec<usize> = found.iter().map(|r| r.len()).collect();
        assert_eq!(lens, [3, 1, 1]);
        assert_eq!(found[1][0], (0x103, o + 9 * PAGE));
        assert!(runs(&[], 0).is_empty());
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
        assert!(mapped_view(at, BYTES) && mapped_view(at + PAGE, PAGE));
        assert!(!mapped_view(at, BYTES + PAGE), "past the view");
        // SAFETY: plain allocation, released below.
        let private = unsafe { mem::VirtualAlloc(std::ptr::null(), PAGE, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE) };
        assert!(!private.is_null());
        assert!(!mapped_view(private as usize, PAGE), "private memory is never opened in place");
        // SAFETY: our view, handle and allocation.
        unsafe {
            mem::VirtualFree(private, 0, mem::MEM_RELEASE);
            mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: at as *mut c_void });
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// A view going while a run over it is opened stays mapped until the
    /// open ends, with no wait; a view the system has not unlocked yet is
    /// unmapped at a later frame's end, not waited for.
    #[test]
    fn a_view_going_mid_open_is_unmapped_after_it_frame_by_frame() {
        use windows_sys::Win32::Foundation::{ERROR_INVALID_PARAMETER, SetLastError};
        let at = 0x7e00_0000_0000usize;
        let calls = Arc::new(AtomicUsize::new(0));
        let counting = |refusals: usize| -> Box<dyn FnMut() -> i32 + Send> {
            let c = calls.clone();
            Box::new(move || {
                let k = c.fetch_add(1, Ordering::SeqCst);
                // SAFETY: plain write of this thread's last error.
                unsafe { SetLastError(ERROR_INVALID_PARAMETER) };
                i32::from(k >= refusals)
            })
        };
        with_runs(|r| r.opening = Some((at + PAGE, PAGE)));
        title_memory_own_gone(at, 4 * PAGE, counting(1));
        assert_eq!(calls.load(Ordering::SeqCst), 0, "kept mapped while the open runs");
        settle();
        assert_eq!(calls.load(Ordering::SeqCst), 0, "still opening");
        assert!(with_runs(|r| finish(r, at + PAGE, PAGE, None)).is_none());
        settle();
        assert_eq!(calls.load(Ordering::SeqCst), 1, "refused as locked: kept");
        settle();
        assert_eq!(calls.load(Ordering::SeqCst), 2, "unmapped");
        settle();
        assert_eq!(calls.load(Ordering::SeqCst), 2, "gone");
        // Nothing over it and nothing opening: unmapped at once.
        title_memory_own_gone(at, 4 * PAGE, counting(0));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert!(with_runs(|r| r.gone.is_empty()));
    }

    /// Runs over private memory end where a chunk does, so that a tracker
    /// or a change of the title's lets go of one chunk only.
    #[test]
    fn private_runs_end_at_chunk_boundaries() {
        let page = |k: usize| (k, k * PAGE);
        let first = 5 * CHUNK - 2;
        let pages: Vec<(usize, usize)> = (first..first + CHUNK + 5).map(page).collect();
        let cut = runs(&pages, CHUNK);
        let lens: Vec<usize> = cut.iter().map(|r| r.len()).collect();
        assert_eq!(lens, [2, CHUNK, 3]);
        assert_eq!(cut[1][0], page(5 * CHUNK));
    }

    /// Reported allocations merge when they touch or overlap.
    #[test]
    fn reported_private_ranges_merge() {
        let all = |r: &BTreeMap<usize, usize>| r.iter().map(|(&s, &e)| (s, e)).collect::<Vec<_>>();
        let mut r = BTreeMap::new();
        merge(&mut r, 0x10_0000, 0x20_0000);
        merge(&mut r, 0x40_0000, 0x50_0000);
        assert_eq!(r.len(), 2);
        // A commit inside one changes nothing.
        merge(&mut r, 0x11_0000, 0x12_0000);
        assert_eq!(all(&r), [(0x10_0000, 0x20_0000), (0x40_0000, 0x50_0000)]);
        // One touching both joins them.
        merge(&mut r, 0x20_0000, 0x40_0000);
        assert_eq!(all(&r), [(0x10_0000, 0x50_0000)]);
    }

    /// Candidate pages are named in place when a run holds them already,
    /// or when their whole 64 KiB is memory that opens; one query answers
    /// for every candidate in a region, and a page reaching past its
    /// region is left out.
    #[test]
    fn private_pages_are_those_that_open() {
        // Memory: pages 0-3 and 4 KiB of page 4 open, the rest of page 4
        // guarded, pages 5-7 reserved only, pages 8-9 open.
        let regions = [(4 * PAGE + 0x1000, true), (5 * PAGE, false), (8 * PAGE, false), (10 * PAGE, true)];
        let mut asked = 0;
        let region = |at: usize| {
            asked += 1;
            *regions.iter().find(|&&(end, _)| at < end).expect("inside")
        };
        // Page 6 lies in an open run (it was committed when opened).
        let got = pick(&[0, 1, 3, 4, 5, 6, 7, 8, 9], |p| p == 6, region);
        let pages: Vec<usize> = got.iter().map(|&(p, _)| p).collect();
        assert_eq!(pages, [0, 1, 3, 6, 8, 9]);
        assert!(got.iter().all(|&(p, at)| at == p * PAGE), "private memory is at its own address");
        // Pages 0-3, page 4, pages 5-7, pages 8-9.
        assert_eq!(asked, 4);
        assert!(pick(&[], |_| false, |at| (at + PAGE, true)).is_empty());
    }

    /// The pages of runs over private memory that are alive are locked
    /// (no guard is set on them), whole pages only, whatever their order.
    #[test]
    fn pages_of_live_private_runs_are_locked() {
        let base = 0x6000_0000_0000usize;
        let mut ranges = BTreeMap::new();
        ranges.insert((base, 7), base + 4 * PAGE);
        // A run replaced by a longer one at the same start, both alive.
        ranges.insert((base, 9), base + 8 * PAGE);
        ranges.insert((base + 32 * PAGE, 8), base + 33 * PAGE);
        assert!(in_ranges(&ranges, base));
        assert!(in_ranges(&ranges, base + 7 * PAGE));
        assert!(!in_ranges(&ranges, base + 8 * PAGE), "past the run's end");
        assert!(!in_ranges(&ranges, base - PAGE));
        assert!(in_ranges(&ranges, base + 32 * PAGE));
        assert!(!in_ranges(&BTreeMap::new(), base));
    }

    /// Two private allocations side by side: runs over both are cut where
    /// the first ends, and a range across both is not opened (gpu308: the
    /// host refused one with E_INVALIDARG); each alone is.
    #[test]
    fn private_runs_stay_within_one_allocation() {
        let _one = worker_tests();
        use windows_sys::Win32::System::Memory as mem;
        let len = 4 * PAGE;
        // SAFETY: a reservation to find room, released at once; then two
        // allocations in it, released below.
        let (at, ok) = unsafe {
            let at = mem::VirtualAlloc(std::ptr::null(), len, mem::MEM_RESERVE, mem::PAGE_NOACCESS) as usize;
            assert_ne!(at, 0);
            mem::VirtualFree(at as *mut c_void, 0, mem::MEM_RELEASE);
            let a = mem::VirtualAlloc(at as *const c_void, 2 * PAGE, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE);
            let b = mem::VirtualAlloc((at + 2 * PAGE) as *const c_void, 2 * PAGE, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE);
            (at, a as usize == at && b as usize == at + 2 * PAGE)
        };
        if ok {
            let pages: Vec<(usize, usize)> = (0..4).map(|k| (at / PAGE + k, at + k * PAGE)).collect();
            let cut = private_runs(&pages);
            assert_eq!(cut.iter().map(|r| r.len()).sum::<usize>(), 4);
            assert!(cut.iter().all(|r| r[0].1 >= at + 2 * PAGE || r[r.len() - 1].1 < at + 2 * PAGE), "none across: {cut:?}");
            merge(&mut PRIVATE.lock().unwrap(), at, at + len);
            let _unreport = Unreport(at, len);
            assert!(!private_open(at, len), "across two allocations");
            assert!(private_open(at, 2 * PAGE) && private_open(at + 2 * PAGE, 2 * PAGE));
        }
        // SAFETY: ours (a release of what is not there fails harmlessly).
        unsafe {
            mem::VirtualFree(at as *mut c_void, 0, mem::MEM_RELEASE);
            mem::VirtualFree((at + 2 * PAGE) as *mut c_void, 0, mem::MEM_RELEASE);
        }
        assert!(ok, "the two allocations went where asked");
    }
    /// The tests that use the opening thread and its pending count, one at
    /// a time: a stalled open holds the one worker.
    fn worker_tests() -> std::sync::MutexGuard<'static, ()> {
        static ONE: Mutex<()> = Mutex::new(());
        ONE.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The reported private memory over `[at, at + len)` is forgotten (on
    /// drop, however the test ends), so
    /// that a later test given the same addresses finds it unreported (the
    /// tests reporting memory hold [`worker_tests`]).
    struct Unreport(usize, usize);

    impl Drop for Unreport {
        fn drop(&mut self) {
            let (at, end) = (self.0, self.0 + self.1);
            PRIVATE.lock().unwrap_or_else(|e| e.into_inner()).retain(|&s, &mut e| e <= at || s >= end);
        }
    }

    /// How long a test waits for the worker: far past an open's time.
    const WORKER_WAIT: std::time::Duration = std::time::Duration::from_secs(10);
    /// The title protects, then frees, private memory open in place: the
    /// plain calls fail while the run's heap locks the pages, and through
    /// the runtime's path (`crate::pinned`, as the title's `VirtualProtect`
    /// and `VirtualFree` go) the run is let go first and both work.
    #[test]
    fn a_title_protect_over_an_open_private_run_lets_it_go_first() {
        let _one = worker_tests();
        use windows_sys::Win32::System::Memory as mem;
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let Some(host) = crate::host::host() else {
            eprintln!("SKIPPED a_title_protect_over_an_open_private_run_lets_it_go_first: no host GPU here");
            return;
        };
        let len = 4 * PAGE;
        // Private memory as `xmem` makes the title's heaps.
        // SAFETY: plain allocation, freed below.
        let at = unsafe {
            mem::VirtualAlloc(std::ptr::null(), len, mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH, mem::PAGE_READWRITE)
        } as usize;
        assert_ne!(at, 0);
        let free = || {
            // SAFETY: our allocation.
            crate::pinned::title_memory_change(at, len, || unsafe { mem::VirtualFree(at as *mut c_void, 0, mem::MEM_RELEASE) })
        };
        let protect = |to: u32| {
            let mut old = 0;
            // SAFETY: our allocation's first page.
            unsafe { mem::VirtualProtect(at as *const c_void, PAGE, to, &mut old) }
        };
        // The one exception to views of ours: private memory the runtime
        // reported (as `VICTORY_RT_MEMORY` does), and only that.
        assert!(!private_open(at, len), "not reported: refused");
        merge(&mut PRIVATE.lock().unwrap(), at, at + len);
        let _unreport = Unreport(at, len);
        assert!(private_open(at, len) && !private_open(at, len + PAGE));
        // Opened as the worker opens a run a page table asked for.
        assert!(run_for(at, len, true).is_none(), "asked of the worker");
        assert!(wait_pending(WORKER_WAIT), "the worker opened it or was refused");
        let Some((run, 0)) = with_runs(|r| holding(&r.open, at, len)) else {
            // Not open: only if the host itself refuses such a heap (a
            // driver or Windows without `ID3D12Device13`), said out loud.
            let direct = open_at(&host.device, at, len).map(drop);
            assert!(direct.is_err(), "the host opens private memory in place, the worker did not");
            assert_ne!(free(), 0);
            eprintln!("SKIPPED a_title_protect_over_an_open_private_run_lets_it_go_first: the host does not open private memory in place ({direct:?})");
            return;
        };
        let id = run.id;
        // Lists ran with a layout naming it, and their table was rewritten
        // since: that layout alone would keep the heap.
        let key = usize::MAX - 0x51;
        crate::memory::ran_with(key, run);
        assert!(crate::memory::ran_names(id));
        assert!(locked(at / PAGE) && locked(at / PAGE + 3));
        assert!(!locked(at / PAGE + 4));
        assert_eq!(protect(mem::PAGE_READONLY), 0, "locked while the heap lives");
        // The title's call, through the runtime.
        let done = crate::pinned::title_memory_change(at, PAGE, || protect(mem::PAGE_READONLY));
        assert_ne!(done, 0, "let go first, then protected");
        assert!(with_runs(|r| holding(&r.open, at, len)).is_none(), "forgotten");
        assert!(with_runs(|r| r.open.values().all(|run| run.id != id)));
        assert!(!crate::memory::ran_names(id), "the layout its lists ran with let it go too");
        assert!(!locked(at / PAGE), "its heap is gone");
        assert_ne!(protect(mem::PAGE_READWRITE), 0);
        // Open again, and the title commits a page of it anew with another
        // protection (`XMemVirtualAlloc` at an address): let go if the host
        // refuses that as locked, and done either way.
        assert!(run_for(at, len, true).is_none());
        assert!(wait_pending(WORKER_WAIT));
        let reopened = with_runs(|r| holding(&r.open, at, len)).is_some();
        let commit = || {
            // SAFETY: our allocation's first page, committed already.
            let p = unsafe { mem::VirtualAlloc(at as *const c_void, PAGE, mem::MEM_COMMIT, mem::PAGE_READONLY) };
            i32::from(!p.is_null())
        };
        assert_ne!(crate::pinned::title_memory_retry(at, PAGE, commit), 0, "committed read-only (run open: {reopened})");
        // SAFETY: plain data filled in by the query.
        let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { mem::VirtualQuery(at as *const c_void, &mut info, size_of::<mem::MEMORY_BASIC_INFORMATION>()) };
        assert_eq!(info.Protect, mem::PAGE_READONLY);
        assert!(!locked(at / PAGE) || with_runs(|r| holding(&r.open, at, len)).is_some(), "locked only by an open run");
        assert_ne!(crate::pinned::title_memory_change(at, PAGE, || protect(mem::PAGE_READWRITE)), 0);
        // Open again and let go for a tracker while a layout lists ran with
        // still holds its heap: the pages stay locked, and the title's free
        // still lets the heap go first.
        assert!(run_for(at, len, true).is_none());
        assert!(wait_pending(WORKER_WAIT));
        if let Some((run, 0)) = with_runs(|r| holding(&r.open, at, len)) {
            let id = run.id;
            crate::memory::ran_with(key, run);
            make_way(at, len);
            assert!(with_runs(|r| holding(&r.open, at, len)).is_none(), "let go for the tracker");
            assert!(locked(at / PAGE), "its heap lives in the layout");
            assert_ne!(free(), 0, "the heap let go first, then freed");
            assert!(!crate::memory::ran_names(id));
        } else {
            assert_ne!(free(), 0, "and freed");
        }
    }

    /// The title frees private memory while the worker opens a run over it:
    /// the open stops once the run is registered ([`stall`]) and the free
    /// begins then, so it always lands mid-open. The free waits for the open,
    /// lets the run go and works; nothing is left open, locked or pinned.
    #[test]
    fn a_title_free_during_an_open_waits_for_it_and_works() {
        let _one = worker_tests();
        use windows_sys::Win32::System::Memory as mem;
        let len = 4 * PAGE;
        // SAFETY: plain allocation, freed below.
        let at = unsafe { mem::VirtualAlloc(std::ptr::null(), len, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE) } as usize;
        assert_ne!(at, 0);
        // Reported, as the runtime reports the title's private heaps.
        merge(&mut PRIVATE.lock().unwrap(), at, at + len);
        let _unreport = Unreport(at, len);
        stall::arm(at);
        assert!(run_for(at, len, true).is_none(), "asked of the worker");
        assert!(stall::stopped(), "the open stopped, registered");
        assert!(locked(at / PAGE), "locked before the heap opens");
        assert!(crate::pinned::title_memory_held());
        let freer = std::thread::spawn(move || {
            // SAFETY: our allocation.
            crate::pinned::title_memory_change(at, len, || unsafe { mem::VirtualFree(at as *mut c_void, 0, mem::MEM_RELEASE) })
        });
        let freed = freer.join().unwrap();
        stall::arm(0);
        assert_ne!(freed, 0, "freed after the open let go");
        assert!(wait_pending(WORKER_WAIT));
        assert!(with_runs(|r| holding(&r.open, at, len)).is_none(), "nothing open there");
        assert!(!locked(at / PAGE), "nothing locks it");
        assert!(locked_ranges().keys().all(|&(s, _)| s != at), "unregistered");
        // The free's own answer says it was freed: a query here could find
        // the addresses taken again by another test's allocation.
    }

    /// The wait for the worker ends when the last asked run is done, woken
    /// rather than polled, and a bounded wait with one left says so.
    #[test]
    fn the_wait_for_the_worker_ends_with_the_last_asked_run() {
        let _one = worker_tests();
        PRIVATE_PENDING.fetch_add(1, Ordering::Relaxed);
        let asked = Asked(true);
        assert!(!wait_pending(std::time::Duration::ZERO), "one left");
        let done = std::thread::spawn(move || drop(asked));
        assert!(wait_pending(WORKER_WAIT));
        done.join().unwrap();
    }

    /// Spans merge touching ranges, and a page counts as within one only
    /// whole; reported ranges hold a range only whole.
    #[test]
    fn spans_answer_for_whole_pages() {
        let s = spans(vec![(4 * PAGE, 6 * PAGE), (0, 2 * PAGE), (2 * PAGE, 3 * PAGE), (5 * PAGE, 8 * PAGE)]);
        assert_eq!(s, [(0, 3 * PAGE), (4 * PAGE, 8 * PAGE)]);
        assert!(within(&s, 0) && within(&s, 2 * PAGE) && within(&s, 7 * PAGE));
        assert!(!within(&s, 3 * PAGE) && !within(&s, 8 * PAGE));
        assert!(!within(&[], 0));
        let mut r = BTreeMap::new();
        merge(&mut r, 0x10_0000, 0x20_0000);
        assert!(holds(&r, 0x10_0000, 0x20_0000) && holds(&r, 0x11_0000, 0x12_0000));
        assert!(!holds(&r, 0x1F_0000, 0x21_0000) && !holds(&r, 0, 0x1_0000));
    }

    /// A commit lets the host be asked again where it refused, there only.
    #[test]
    fn unrefuse_clears_overlapping_refusals_only() {
        let mut r = Runs::default();
        let base = 0x5000_0000_0000usize;
        r.refused.extend([(base, 4 * PAGE), (base + 8 * PAGE, PAGE), (base - MOST, MOST)]);
        unrefuse(&mut r, base + 3 * PAGE, base + 4 * PAGE);
        assert_eq!(r.refused.iter().copied().collect::<Vec<_>>(), [(base - MOST, MOST), (base + 8 * PAGE, PAGE)]);
        unrefuse(&mut r, base - PAGE, base);
        assert_eq!(r.refused.len(), 1);
    }

    #[test]
    fn a_run_is_at_most_a_region() {
        let o = 0x7000_0000_0000usize;
        let n = MOST / PAGE + 2;
        let pages: Vec<(usize, usize)> = (0..n).map(|k| (k, o + k * PAGE)).collect();
        let lens: Vec<usize> = runs(&pages, 0).iter().map(|r| r.len()).collect();
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
