//! Diagnostics: where the submission copies of title memory go (which
//! ranges, how often, of what memory) and what limits their rate, logged
//! with the frame-rate line. `VICTORY_COPY_MODE=plain` copies without the
//! mapped-page guard, `none` skips the copies (both for timing only).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Counters since the last line.
pub(crate) enum C {
    Submissions,
    Copies,
    CopyBytes,
    Refreshes,
    RefreshBytes,
    Skipped,
    SkippedBytes,
    Runs,
    RunUs,
    Queries,
    QueryUs,
    MoveUs,
    Unmapped,
    /// Write-watch askings and their time.
    Watches,
    WatchUs,
    /// Bytes of watched ranges checked, and of those found unchanged.
    Checked,
    Unchanged,
    /// Copies that met an unmapped page.
    Fallbacks,
    /// Writes taken by the guard on view memory.
    GuardFaults,
    /// Texture bytes detiled after a watched write; hashed; detiled after a
    /// hash change.
    TexWatched,
    TexHashed,
    TexChanged,
    /// Time in the kernel's write-watch answers, in the view check of the
    /// guard, in protecting; bytes and time of copies of memory neither
    /// watched nor guarded (copied whole).
    KernelUs,
    ViewQueryUs,
    ProtectUs,
    WholeBytes,
    WholeUs,
    /// Snapshots of ranges shaders write (bytes, time); page faults while
    /// preparing submissions and while copying back.
    SnapshotBytes,
    SnapshotUs,
    FaultsPrep,
    FaultsBack,
    /// Pages marked changed because a write landed on another mapping of
    /// them.
    AliasMarks,
    /// Time in the guard handler; protection changes making pages read-only
    /// (calls, pages) and giving write access back (pages).
    HandlerUs,
    ProtectCalls,
    ProtectPages,
    OpenPages,
    /// Tracked pages written (distinct, per frame), and of those the ones
    /// written the frame before too.
    WrittenPages,
    WrittenAgain,
    /// In change checks: time asking the runtime for other mappings, and
    /// waiting for the guard lock.
    AliasUs,
    GuardWaitUs,
    /// Preparing submissions: refresh batches done (copies and change
    /// checks); time in each kind of step (a step that flushes the batch
    /// counts that too); texture uploads for heaps; state preludes.
    FlushUs,
    StepRefreshUs,
    StepCopyUs,
    StepWorkUs,
    StepGpuWriteUs,
    StepTextureUs,
    StepHeapUs,
    StepMemoryUs,
    UploadsUs,
    PreludeUs,
    /// Copying back: bytes of copies compared with their snapshots, and
    /// the time.
    CompareBytes,
    CompareUs,
    /// Copying back pages shaders marked as written through views: pages
    /// compared, bytes written back, the time; pages that kept the title's
    /// bytes (it wrote them, their bytes from before unknown); pages whose
    /// bytes the title's first write saved.
    MarkedPages,
    MarkedBytes,
    MarkedUs,
    MarkedKept,
    SavedPages,
    /// Preparing submissions: descriptors the title copied into its heaps
    /// as bytes, made on the host first.
    SyncHeapsUs,
    /// Of those: heap slots looked at (every slot of each page written).
    SyncSlots,
    /// Time in copies redone piece by piece after meeting an unmapped page.
    FallbackUs,
    /// Title threads writing a view into a shader-visible heap: time
    /// waiting for the heaps' lock (held by heap syncs and other writers).
    MirrorWaitUs,
    /// Marked copy-backs: the pull from video memory and the pages' wall
    /// time; thread time bringing the mirror up to date, reading title
    /// memory, comparing, taking its other bytes (these four in
    /// nanoseconds), writing it and taking pages again; spans written; per
    /// submission the smaller of pull and wall time (the upper bound a
    /// future overlap of the two could save, not a measured overlap).
    MarkedPullUs,
    MarkedBackWallUs,
    MarkedFreshenNs,
    MarkedReadNs,
    MarkedDiffNs,
    MarkedTakeNs,
    MarkedWriteUs,
    MarkedRetakeUs,
    MarkedSpans,
    MarkedOverlapUs,
    /// Null UAV barriers recorded: for title buffer barriers (by cause, see
    /// `crate::list`), the title's own, and for enhanced barrier groups.
    BarriersReadRead,
    BarriersBeginOnly,
    BarriersOther,
    BarriersTitle,
    BarriersGroups,
    /// Indirect argument clamps, and those with no shader write recorded in
    /// the list since the one before.
    Clamps,
    ClampsNoWrite,
    /// Windows of indirect draws clamped together (`VICTORY_CLAMP_WINDOW`),
    /// the draws they took, and indirect draws clamped alone because their
    /// shaders may store.
    ClampWindows,
    ClampWindowDraws,
    ClampStoring,
    /// Open windows ended by a storing pipeline: compute, graphics storing
    /// only through table unordered views (maybe never written), other
    /// graphics.
    WindowEndCompute,
    WindowEndTableUav,
    WindowEndGraphics,
    /// Open windows ended by an indirect draw reading another buffer, by
    /// one that did not fit, and by a storing draw closing its window.
    WindowEndSource,
    WindowEndFull,
    WindowEndClosing,
    /// Of those closing draws, the stages' reasons to store (several may
    /// count for one draw): view writes, title memory bitmap bound (reads
    /// too), heap writes, written root views, root UAV view, table UAVs.
    ClosingViewMarks,
    ClosingMemory,
    ClosingHeap,
    ClosingRootUavs,
    ClosingRootView,
    ClosingTable,
    /// The guard handler: nanoseconds waiting for the guard lock, and
    /// holding it.
    HandlerWaitNs,
    HandlerHoldNs,
    /// In the guard handler: nanoseconds in `Changed::save`, nanoseconds
    /// in `VirtualProtect` opening pages, pages saved, and nanoseconds
    /// waiting for trackers' own locks while marking.
    HandlerSaveNs,
    HandlerOpenNs,
    HandlerSaves,
    HandlerTrackerWaitNs,
    /// Heap sync slots found stale: never made, or nulled earlier (unmade;
    /// also a made slot whose entry is newer than the sync's generations),
    /// from the same view entry of an older generation (recycled), from
    /// another view entry (rewritten), and made while the record is not
    /// current (released, non-view record, or out-of-range index).
    SyncStaleFirst,
    SyncStaleRecycled,
    SyncStaleRewritten,
    SyncStaleReleased,
    /// Descriptors written through the API (view creation, copies into
    /// shader-visible kinds only, to compare with the stale counts), and
    /// `descriptor::forget` calls (generation bumps on release).
    ApiViewWrites,
    ForgetCalls,
    /// Flight study (title submissions only): nanoseconds from entering the
    /// submission to sending its lists (of them, syncing heaps), waiting on
    /// the card's fence, and settling after it (copy-back, deferred copies
    /// and work, bottom-of-pipe writes).
    FlightPrepNs,
    FlightSyncHeapsNs,
    FlightGpuWaitNs,
    FlightSettleNs,
    /// Title submissions that, with the one before still in flight, would
    /// have to settle it first, by reason (one may have several): touching
    /// a range it declared written, a title fence or memory value wait,
    /// our own synchronous submissions, a present; and those with none.
    DrainRange,
    DrainFence,
    DrainSyncExec,
    DrainPresent,
    DrainFree,
    /// Title fence polls (`GetCompletedValue`), waiting or not.
    FlightFencePolls,
    /// Guard faults by the memory they hit (`VICTORY_DIAG_FAULT_KINDS=1`):
    /// write-watched pool (and of those, pages some tracker had armed),
    /// mapped view, fixed private reservation, other.
    GuardFaultsPool,
    GuardFaultsPoolArmed,
    GuardFaultsView,
    GuardFaultsFixed,
    GuardFaultsOther,
    /// Indirect mesh dispatches dropped (not recorded).
    MeshIndirectDropped,
    /// Submissions whose pull ran right after their lists, under their
    /// fence wait (`VICTORY_PULL_SAME_EXECUTE`).
    PullsRiding,
    /// Heap sync slots handed to `mirror` (record or view generation
    /// changed, or unmade holding a view record), of the `SyncSlots`
    /// looked at.
    SyncChanged,
    /// Nanoseconds threads waited for a copy's stale page lock another
    /// held (a riding pull holds them over its submission's wait).
    StaleLockWaitNs,
    /// Bytes refreshed into video memory only (mirrored copies).
    VideoBytes,
    /// Bytes of taken pages refreshed, by who wrote the page (a copy-back
    /// of ours this frame or the one before, else the title) and where the
    /// refresh went (video memory of a mirrored copy, else host memory).
    RefreshOursVideo,
    RefreshOursHost,
    RefreshTitleVideo,
    RefreshTitleHost,
    /// Refresh copies made at once on the preparing thread (bytes, time),
    /// and bytes queued for the copy pool.
    RefreshSerialBytes,
    RefreshSerialUs,
    RefreshPoolBytes,
    /// Pipelining study (title submissions only): title bytes a submission
    /// takes that the one before declared written, by step: a refresh of a
    /// clean copy (nothing copied), of a changed copy the one before wrote
    /// (same copy) or did not (alias), a texture staging, a title copy or
    /// fill, a range its shaders write; submissions with a real conflict
    /// (any but clean, or a heap page below).
    RangeClean,
    RangeSame,
    RangeAlias,
    RangeStaging,
    RangeCopy,
    RangeGpuWrite,
    DrainReal,
    /// Heap pages synced that the submission before declared written.
    HeapInFlight,
    /// Queue memory value waits at their first test, by what last wrote the
    /// word: a bottom-of-pipe value (predictable), a query result, a
    /// timestamp, none of ours (the CPU, or unknown); queue fence Waits;
    /// title event waits for a fence value not reached.
    WaitsValue,
    WaitsHostValue,
    WaitsTimestamp,
    WaitsCpu,
    QueueFenceWaits,
    FenceEventsUnmet,
    /// Null views made after a submission for shaders that read them: slots,
    /// and submissions making any.
    MadeReadSlots,
    MadeReadSubs,
    /// Title submissions using a host copy the one before wrote.
    SharedCopySubs,
    /// Preparing a title submission while the one before would still be on
    /// the GPU (the hazards that keep `VICTORY_PIPELINE` from overlapping
    /// them): submissions meeting any; changed copies refreshed that the
    /// one before used; heap pages synced (slots it may read); submissions
    /// with marked writes after one with marked writes (one marks bitmap);
    /// snapshot ranges and marked ranges of copies the one before wrote.
    PrepHazardSubs,
    PrepDirtyInUse,
    PrepHeapPages,
    PrepMarkedAfterMarked,
    PrepSnapshotWritten,
    PrepMarkedWritten,
    /// Nanoseconds the title's threads spent in ExecuteCommandLists.
    TitleExecNs,
}
const COUNTERS: usize = C::TitleExecNs as usize + 1;
static COUNT: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];

/// Whether the refresh's own statistics are kept (who wrote the pages
/// refreshed, time in the copies made at once): `VICTORY_REFRESH_STATS=1`.
/// Off, their hot paths do nothing: [`ours_wrote`] runs for every span a
/// copy-back writes, on its threads, and [`refresh_cause`] for every run
/// of pages a refresh takes. Asked once.
pub(crate) fn refresh_stats() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::switch::on("VICTORY_REFRESH_STATS");
        tracing::info!(on, "refresh statistics by cause and path (VICTORY_REFRESH_STATS)");
        on
    })
}

/// The title pages our copy-backs wrote last, and when: a table with no
/// lock and no allocation, a slot per page by its hash, each holding the
/// page (64 KiB index) above [`FRAME_BITS`] bits of the frame. Two pages
/// that share a slot leave the later one: the statistics then take a page
/// we wrote for one the title wrote, never the reverse.
static OURS: [AtomicU64; OURS_SLOTS] = [const { AtomicU64::new(u64::MAX) }; OURS_SLOTS];
const OURS_SLOTS: usize = 1 << 16;
const FRAME_BITS: u32 = 20;
const FRAME_MASK: u64 = (1 << FRAME_BITS) - 1;

/// A page's slot in [`OURS`] (FxHash's multiply, the high bits).
fn ours_slot(page: usize) -> usize {
    ((page as u64).wrapping_mul(0x517c_c1b7_2722_0a95) >> 48) as usize % OURS_SLOTS
}

/// What [`OURS`] holds for `page` written in `frame`.
fn ours_entry(page: usize, frame: u64) -> u64 {
    ((page as u64) << FRAME_BITS) | (frame & FRAME_MASK)
}

/// Whether `entry` says that `page` was written in the frame `now` or the
/// one before.
fn ours_recent(entry: u64, page: usize, now: u64) -> bool {
    (entry >> FRAME_BITS) == page as u64 && (now.wrapping_sub(entry) & FRAME_MASK) <= 1
}

/// A copy-back wrote title memory `[at, at + len)` (our copies into title
/// memory too: [`crate::host::copy_mapped`] to a title destination, and a
/// texture's copy back in `list.rs`).
pub(crate) fn ours_wrote(at: usize, len: usize) {
    use crate::watch::PAGE;
    if !refresh_stats() {
        return;
    }
    let frame = crate::queue::presented();
    for p in at / PAGE..(at + len).div_ceil(PAGE) {
        OURS[ours_slot(p)].store(ours_entry(p, frame), Relaxed);
    }
}

/// A refresh of taken title memory `[a, b)`, into video memory or not.
pub(crate) fn refresh_cause(a: usize, b: usize, video: bool) {
    use crate::watch::PAGE;
    if !refresh_stats() {
        return;
    }
    let now = crate::queue::presented();
    let mut ours = 0;
    for p in a / PAGE..b.div_ceil(PAGE) {
        if ours_recent(OURS[ours_slot(p)].load(Relaxed), p, now) {
            ours += (b.min((p + 1) * PAGE) - a.max(p * PAGE)) as u64;
        }
    }
    let (o, t) = if video {
        (C::RefreshOursVideo, C::RefreshTitleVideo)
    } else {
        (C::RefreshOursHost, C::RefreshTitleHost)
    };
    add(o, ours);
    add(t, (b - a) as u64 - ours);
}

/// The guard handler's longest single wait for the guard lock since the
/// last line, nanoseconds.
pub(crate) static HANDLER_WAIT_MAX: AtomicU64 = AtomicU64::new(0);
/// The guard handler's longest single save, and longest single open
/// (`VirtualProtect`), since the last line, nanoseconds.
pub(crate) static HANDLER_SAVE_MAX: AtomicU64 = AtomicU64::new(0);
pub(crate) static HANDLER_OPEN_MAX: AtomicU64 = AtomicU64::new(0);

pub(crate) fn add(c: C, n: u64) {
    COUNT[c as usize].fetch_add(n, Relaxed);
}

/// [`crate::watch::Timed`] in nanoseconds: per-page times under a
/// microsecond would add as zero.
pub(crate) struct TimedNs(pub(crate) std::time::Instant, pub(crate) C);

impl Drop for TimedNs {
    fn drop(&mut self) {
        let c = std::mem::replace(&mut self.1, C::Submissions);
        add(c, self.0.elapsed().as_nanos() as u64);
    }
}

/// Set around the title's submissions (`crate::queue`): other executes are
/// ours, synchronous.
thread_local! {
    pub(crate) static TITLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
/// A title fence poll or wait, or a queue's memory value or fence wait,
/// since the last title submission.
pub(crate) static FLIGHT_FENCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// One of our synchronous executes since the last title submission.
static FLIGHT_SYNC: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Frames presented at the last title submission.
static FLIGHT_PRESENTED: AtomicU64 = AtomicU64::new(0);
/// The last title submission's written title ranges (start, end), sorted
/// and merged, and the host copies it wrote (sorted); only set under the
/// host's submission lock.
static FLIGHT_WRITES: Mutex<Last> = Mutex::new(Last { writes: Vec::new(), copies: Vec::new(), used: Vec::new(), marked: false });

/// What the last title submission wrote and used: title ranges (merged),
/// host copies written and used (sorted), and whether it marked pages.
#[derive(Clone, Default)]
pub(crate) struct Last {
    pub(crate) writes: Vec<(usize, usize)>,
    pub(crate) copies: Vec<usize>,
    pub(crate) used: Vec<usize>,
    pub(crate) marked: bool,
}

/// A title submission's own findings, for [`flight_end`]: a conflict that
/// needs the one before settled; it used a copy the one before wrote; the
/// copies it used; it marked pages; it met an overlap hazard.
pub(crate) struct Mine {
    pub(crate) real: bool,
    pub(crate) shared: bool,
    pub(crate) used: Vec<usize>,
    pub(crate) marked: bool,
    pub(crate) hazard: bool,
}
/// Whether a title submission is being prepared: set at its start, unset
/// once its lists are made ([`flight_prepared`]), so heap syncs elsewhere
/// (tests, our own submissions) are not counted.
static FLIGHT_TITLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// A heap page synced for this submission was written by the one before.
static FLIGHT_HEAP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// A heap page was synced for this title submission at all.
static FLIGHT_HEAP_ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// What last wrote each bottom-of-pipe word (0 a value, 1 a query result,
/// 2 a timestamp) and the frame it did; words not written for two frames
/// are dropped at each line.
static LABELS: Mutex<Option<HashMap<u64, (u8, u64)>>> = Mutex::new(None);

/// Why a title submission would drain the one before (fence, sync
/// executes, present); `None` for our own executes.
pub(crate) struct Flight(bool, bool, bool);

/// At a submission's start: `None` (and noted) for our own.
pub(crate) fn flight_begin() -> Option<Flight> {
    let title = TITLE.replace(false);
    FLIGHT_TITLE.store(title, Relaxed);
    if !title {
        FLIGHT_SYNC.store(true, Relaxed);
        return None;
    }
    let frames = crate::queue::presented();
    Some(Flight(
        FLIGHT_FENCE.swap(false, Relaxed),
        FLIGHT_SYNC.swap(false, Relaxed),
        FLIGHT_PRESENTED.swap(frames, Relaxed) != frames,
    ))
}

/// As a title submission's lists go to the card: waits and executes during
/// its preparation came after the one before, so they count for it.
pub(crate) fn flight_sent(f: &mut Option<Flight>) {
    if let Some(f) = f {
        f.0 |= FLIGHT_FENCE.swap(false, Relaxed);
        f.1 |= FLIGHT_SYNC.swap(false, Relaxed);
    }
}

/// What the last title submission wrote and used (empty for our own).
pub(crate) fn flight_writes(flight: &Option<Flight>) -> Last {
    if flight.is_none() {
        return Default::default();
    }
    FLIGHT_WRITES.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// A submission's lists are made: heap syncs after are not its own.
pub(crate) fn flight_prepared() {
    FLIGHT_TITLE.store(false, Relaxed);
}

/// A heap page a title submission's sync takes: counted, and also if the
/// submission before declared it written.
pub(crate) fn heap_page(at: usize, len: usize) {
    if !FLIGHT_TITLE.load(Relaxed) {
        return;
    }
    add(C::PrepHeapPages, 1);
    FLIGHT_HEAP_ANY.store(true, Relaxed);
    if touches(&FLIGHT_WRITES.lock().unwrap_or_else(|e| e.into_inner()).writes, at, len) {
        add(C::HeapInFlight, 1);
        FLIGHT_HEAP.store(true, Relaxed);
    }
}

/// A bottom-of-pipe write at `at`, for [`wait_source`].
pub(crate) fn label_written(at: usize, what: &crate::queue::Written) {
    let kind = match what {
        crate::queue::Written::Value(_) => 0,
        crate::queue::Written::HostValue(_) => 1,
        crate::queue::Written::GpuTimestamp | crate::queue::Written::SystemTimestamp => 2,
    };
    let frame = crate::queue::presented();
    LABELS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default().insert(at as u64, (kind, frame));
}

/// A queue's memory value wait at its first test, counted by what last
/// wrote its word.
pub(crate) fn wait_source(address: u64) {
    let kind = LABELS.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|l| l.get(&address).map(|w| w.0));
    add(
        match kind {
            Some(0) => C::WaitsValue,
            Some(1) => C::WaitsHostValue,
            Some(_) => C::WaitsTimestamp,
            None => C::WaitsCpu,
        },
        1,
    );
}

/// At a title submission's end: its counts, and what it wrote and used,
/// for the next.
pub(crate) fn flight_end(f: Flight, range: bool, times: [u64; 4], mut writes: Vec<(usize, usize)>, mut copies: Vec<usize>, mine: Mine) {
    let Mine { real, shared, mut used, marked, hazard } = mine;
    let after_marked = marked && FLIGHT_WRITES.lock().unwrap_or_else(|e| e.into_inner()).marked;
    if after_marked {
        add(C::PrepMarkedAfterMarked, 1);
    }
    if hazard | after_marked | FLIGHT_HEAP_ANY.swap(false, Relaxed) {
        add(C::PrepHazardSubs, 1);
    }
    if real | FLIGHT_HEAP.swap(false, Relaxed) {
        add(C::DrainReal, 1);
    }
    if shared {
        add(C::SharedCopySubs, 1);
    }
    for (c, t) in [C::FlightPrepNs, C::FlightSyncHeapsNs, C::FlightGpuWaitNs, C::FlightSettleNs].into_iter().zip(times) {
        add(c, t);
    }
    for (c, on) in [(C::DrainRange, range), (C::DrainFence, f.0), (C::DrainSyncExec, f.1), (C::DrainPresent, f.2)] {
        if on {
            add(c, 1);
        }
    }
    if !(range || f.0 || f.1 || f.2) {
        add(C::DrainFree, 1);
    }
    merge(&mut writes);
    copies.sort_unstable();
    copies.dedup();
    used.sort_unstable();
    used.dedup();
    *FLIGHT_WRITES.lock().unwrap_or_else(|e| e.into_inner()) = Last { writes, copies, used, marked };
}

/// Sorts and merges ranges (start, end) into disjoint ones.
fn merge(r: &mut Vec<(usize, usize)>) {
    r.retain(|&(a, b)| a < b);
    r.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(r.len());
    for &(a, b) in r.iter() {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    *r = out;
}

/// Whether `len` bytes at `at` touch merged ranges.
pub(crate) fn touches(merged: &[(usize, usize)], at: usize, len: usize) -> bool {
    let i = merged.partition_point(|&(_, end)| end <= at);
    merged.get(i).is_some_and(|&(start, _)| start < at + len)
}

#[cfg(test)]
#[test]
fn flight_ranges_merge_and_touch() {
    let mut r = vec![(30, 40), (0, 10), (5, 20), (50, 50)];
    merge(&mut r);
    assert_eq!(r, [(0, 20), (30, 40)]);
    assert!(touches(&r, 19, 1));
    assert!(!touches(&r, 20, 10));
    assert!(touches(&r, 25, 6));
    assert!(!touches(&r, 40, 100));
}

/// How copies run: 0 guarded (as always), 1 plain, 2 not at all.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_COPY_MODE").as_deref() {
        Ok("plain") => 1,
        Ok("none") => 2,
        _ => 0,
    })
}

/// Per source range (start, length): copies, bytes, microseconds, memory type.
type Ranges = HashMap<(usize, usize), (u64, u64, u64, u32)>;
static RANGES: Mutex<Option<Ranges>> = Mutex::new(None);

/// A refresh of `len` bytes from `src` (memory type `kind`) copied `copied`
/// of them and took `us`.
pub(crate) fn range(src: usize, len: usize, copied: usize, us: u64, kind: u32) {
    let mut map = RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let e = map
        .get_or_insert_with(HashMap::new)
        .entry((src, len))
        .or_insert((0, 0, 0, kind));
    e.0 += 1;
    e.1 += copied as u64;
    e.2 += us;
}

/// Snapshot bytes per host copy (its title start and length).
static SNAPSHOTS: Mutex<Option<HashMap<(u64, u64), u64>>> = Mutex::new(None);

/// A snapshot of `bytes` of the copy of `[start, start + len)`.
pub(crate) fn snapshot(start: u64, len: u64, bytes: u64) {
    *SNAPSHOTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .entry((start, len))
        .or_default() += bytes;
}

/// Our process's working set, bytes.
fn working_set() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    c.WorkingSetSize as u64
}

/// Our process's hard page faults so far (read from disk or the page
/// file), from the system's process list: `HardFaultCount` at +0x10 of
/// each entry, the process id at +0x50.
fn hard_faults() -> u64 {
    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn NtQuerySystemInformation(class: u32, info: *mut u8, len: u32, ret: *mut u32) -> i32;
    }
    const PROCESSES: u32 = 5;
    let id = std::process::id() as usize;
    let mut buf = vec![0u8; 4 << 20];
    let mut ret = 0;
    // SAFETY: a buffer of that length.
    if unsafe { NtQuerySystemInformation(PROCESSES, buf.as_mut_ptr(), buf.len() as u32, &mut ret) } < 0 {
        return 0;
    }
    let mut at = 0usize;
    loop {
        let entry = &buf[at..];
        let next = u32::from_le_bytes(entry[0..4].try_into().unwrap_or_default()) as usize;
        let pid = usize::from_le_bytes(entry[0x50..0x58].try_into().unwrap_or_default());
        if pid == id {
            return u64::from(u32::from_le_bytes(entry[0x10..0x14].try_into().unwrap_or_default()));
        }
        if next == 0 || at + next + 0x58 > buf.len() {
            return 0;
        }
        at += next;
    }
}

pub(crate) fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// Logs the counters per frame over `frames` frames, and the ranges that
/// took most bytes.
pub(crate) fn log(frames: f64) {
    let now = crate::queue::presented();
    if let Some(labels) = LABELS.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        labels.retain(|_, w| w.1 + 2 >= now);
    }
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let faults = page_faults();
    let faults = faults - FAULTS.swap(faults, Relaxed).min(faults);
    let c: Vec<f64> = COUNT
        .iter()
        .map(|a| a.swap(0, Relaxed) as f64 / frames)
        .collect();
    let mb = |v: f64| v / 1e6;
    let ms = |v: f64| v / 1e3;
    tracing::info!(
        "copies per frame: {:.1} submissions; {:.0} title copies {:.1} MB; {:.0} refreshes {:.1} MB, \
         {:.0} skipped {:.1} MB; {:.0} runs {:.1} ms; {:.0} queries {:.2} ms, moves {:.2} ms, \
         {:.2} MB unmapped; {:.0} page faults; {:.0} watches {:.2} ms, {:.1} MB checked, {:.1} MB \
         unchanged; {:.1} fallbacks ({:.1} ms); {:.0} guard faults; textures {:.1} MB watched and \
         changed, {:.1} MB hashed, {:.1} MB of them changed; watch time: kernel {:.1} ms, view queries {:.1} ms, protecting {:.1} ms; copied whole (not tracked) {:.1} MB in {:.1} ms; snapshots {:.1} MB in {:.1} ms; page faults {:.0} preparing, {:.0} copying back; {:.0} tracked pages marked through another mapping; \
         guards: {:.2} ms in the handler, {:.0} protect calls for {:.0} pages, {:.0} pages opened; \
         {:.0} tracked pages written ({:.0} of them the frame before too); change checks: {:.1} ms \
         asking for other mappings, {:.1} ms waiting for the guard lock; preparing: {:.1} ms in \
         refresh batches, steps (each with the batch it flushes): refresh {:.1} ms, copy {:.1} ms, \
         work {:.1} ms, GPU writes {:.1} ms, textures {:.1} ms, heaps {:.1} ms, title memory {:.1} ms; \
         heap texture uploads {:.1} ms, state preludes {:.1} ms, heap slots the title copied {:.1} ms ({:.0} slots), title threads waiting to write views {:.1} ms; copying back compared {:.1} MB          in {:.1} ms; marked pages: {:.0} compared, {:.1} MB written back in {:.1} ms, {:.0} kept the title's, {:.0} saved; working set {:.0} MB, \
         {:.0} hard faults",
        c[C::Submissions as usize],
        c[C::Copies as usize],
        mb(c[C::CopyBytes as usize]),
        c[C::Refreshes as usize],
        mb(c[C::RefreshBytes as usize]),
        c[C::Skipped as usize],
        mb(c[C::SkippedBytes as usize]),
        c[C::Runs as usize],
        ms(c[C::RunUs as usize]),
        c[C::Queries as usize],
        ms(c[C::QueryUs as usize]),
        ms(c[C::MoveUs as usize]),
        mb(c[C::Unmapped as usize]),
        faults as f64 / frames,
        c[C::Watches as usize],
        ms(c[C::WatchUs as usize]),
        mb(c[C::Checked as usize]),
        mb(c[C::Unchanged as usize]),
        c[C::Fallbacks as usize],
        ms(c[C::FallbackUs as usize]),
        c[C::GuardFaults as usize],
        mb(c[C::TexWatched as usize]),
        mb(c[C::TexHashed as usize]),
        mb(c[C::TexChanged as usize]),
        ms(c[C::KernelUs as usize]),
        ms(c[C::ViewQueryUs as usize]),
        ms(c[C::ProtectUs as usize]),
        mb(c[C::WholeBytes as usize]),
        ms(c[C::WholeUs as usize]),
        mb(c[C::SnapshotBytes as usize]),
        ms(c[C::SnapshotUs as usize]),
        c[C::FaultsPrep as usize],
        c[C::FaultsBack as usize],
        c[C::AliasMarks as usize],
        ms(c[C::HandlerUs as usize]),
        c[C::ProtectCalls as usize],
        c[C::ProtectPages as usize],
        c[C::OpenPages as usize],
        c[C::WrittenPages as usize],
        c[C::WrittenAgain as usize],
        ms(c[C::AliasUs as usize]),
        ms(c[C::GuardWaitUs as usize]),
        ms(c[C::FlushUs as usize]),
        ms(c[C::StepRefreshUs as usize]),
        ms(c[C::StepCopyUs as usize]),
        ms(c[C::StepWorkUs as usize]),
        ms(c[C::StepGpuWriteUs as usize]),
        ms(c[C::StepTextureUs as usize]),
        ms(c[C::StepHeapUs as usize]),
        ms(c[C::StepMemoryUs as usize]),
        ms(c[C::UploadsUs as usize]),
        ms(c[C::PreludeUs as usize]),
        ms(c[C::SyncHeapsUs as usize]),
        c[C::SyncSlots as usize],
        ms(c[C::MirrorWaitUs as usize]),
        mb(c[C::CompareBytes as usize]),
        ms(c[C::CompareUs as usize]),
        c[C::MarkedPages as usize],
        mb(c[C::MarkedBytes as usize]),
        ms(c[C::MarkedUs as usize]),
        c[C::MarkedKept as usize],
        c[C::SavedPages as usize],
        working_set() as f64 / 1e6,
        {
            static HARD: AtomicU64 = AtomicU64::new(0);
            let hard = hard_faults();
            (hard - HARD.swap(hard, Relaxed).min(hard)) as f64 / frames
        },
    );
    tracing::info!(
        "  heap sync stale by reason per frame: unmade {:.0}, recycled {:.0}, rewritten {:.0}, released {:.0}; api view writes {:.0}, forgets {:.0}",
        c[C::SyncStaleFirst as usize],
        c[C::SyncStaleRecycled as usize],
        c[C::SyncStaleRewritten as usize],
        c[C::SyncStaleReleased as usize],
        c[C::ApiViewWrites as usize],
        c[C::ForgetCalls as usize],
    );
    if c[C::SyncChanged as usize] != 0.0 {
        tracing::info!(
            "  heap sync per frame: {:.0} slots mirrored of {:.0} looked at",
            c[C::SyncChanged as usize],
            c[C::SyncSlots as usize],
        );
    }
    tracing::info!(
        "  marked copy-back per frame: pull {:.1} ms, pages {:.1} ms wall (upper bound a future overlap of the two could save, not measured: {:.1} ms); thread time: freshening {:.1} ms, reading {:.1} ms, comparing {:.1} ms, taking others {:.1} ms, writing {:.1} ms, taking and reading again {:.1} ms; {:.0} spans; {:.0} pulls run under their submission's wait; {:.2} ms waited on stale page locks",
        ms(c[C::MarkedPullUs as usize]),
        ms(c[C::MarkedBackWallUs as usize]),
        ms(c[C::MarkedOverlapUs as usize]),
        c[C::MarkedFreshenNs as usize] / 1e6,
        c[C::MarkedReadNs as usize] / 1e6,
        c[C::MarkedDiffNs as usize] / 1e6,
        c[C::MarkedTakeNs as usize] / 1e6,
        ms(c[C::MarkedWriteUs as usize]),
        ms(c[C::MarkedRetakeUs as usize]),
        c[C::MarkedSpans as usize],
        c[C::PullsRiding as usize],
        c[C::StaleLockWaitNs as usize] / 1e6,
    );
    tracing::info!(
        "  null UAV barriers per frame: title buffer barriers {:.0} read to read, {:.0} begin only, {:.0} other; {:.0} the title's own; {:.0} for barrier groups; {:.0} indirect clamps ({:.0} with no shader write since the one before); {:.1} MB refreshed into video memory only; taken pages refreshed: written by our copy-backs {:.1} MB to video memory, {:.1} MB to host, by the title {:.1} MB to video memory, {:.1} MB to host; refresh copies made at once {:.1} MB in {:.2} ms, {:.1} MB queued for the pool",
        c[C::BarriersReadRead as usize],
        c[C::BarriersBeginOnly as usize],
        c[C::BarriersOther as usize],
        c[C::BarriersTitle as usize],
        c[C::BarriersGroups as usize],
        c[C::Clamps as usize],
        c[C::ClampsNoWrite as usize],
        mb(c[C::VideoBytes as usize]),
        mb(c[C::RefreshOursVideo as usize]),
        mb(c[C::RefreshOursHost as usize]),
        mb(c[C::RefreshTitleVideo as usize]),
        mb(c[C::RefreshTitleHost as usize]),
        mb(c[C::RefreshSerialBytes as usize]),
        ms(c[C::RefreshSerialUs as usize]),
        mb(c[C::RefreshPoolBytes as usize]),
    );
    if crate::indirect::windows_on() {
        tracing::info!(
            "  clamp windows per frame: {:.0} holding {:.0} indirect draws; {:.0} indirect draws alone as they may store",
            c[C::ClampWindows as usize],
            c[C::ClampWindowDraws as usize],
            c[C::ClampStoring as usize],
        );
        tracing::info!(
            "  clamp windows ended per frame by a storing pipeline: {:.0} compute, {:.0} graphics through table UAVs only, {:.0} other graphics",
            c[C::WindowEndCompute as usize],
            c[C::WindowEndTableUav as usize],
            c[C::WindowEndGraphics as usize],
        );
        tracing::info!(
            "  clamp windows ended per frame by a draw: {:.0} reading another buffer, {:.0} not fitting, {:.0} storing (closing)",
            c[C::WindowEndSource as usize],
            c[C::WindowEndFull as usize],
            c[C::WindowEndClosing as usize],
        );
        tracing::info!(
            "  closing draws' reasons per frame: {:.0} view writes, {:.0} title memory, {:.0} heap writes, {:.0} written root UAVs, {:.0} root UAV view, {:.0} table UAVs",
            c[C::ClosingViewMarks as usize],
            c[C::ClosingMemory as usize],
            c[C::ClosingHeap as usize],
            c[C::ClosingRootUavs as usize],
            c[C::ClosingRootView as usize],
            c[C::ClosingTable as usize],
        );
    }
    tracing::info!(
        "flight study per frame: prep {:.2} ms (sync heaps {:.2}), gpu wait {:.2} ms, settle {:.2} ms; next would drain: range {:.1}, fence {:.1}, sync exec {:.1}, present {:.1}, free {:.1}; fence polls {:.1}; pipelining: real drains {:.1}; taken while the one before wrote them: clean refreshes {:.1}, changed same copy {:.1}, changed alias {:.1}, stagings {:.1}, copies {:.1}, shader writes {:.1}, heap pages {:.1}; value waits by writer: value {:.1}, query {:.1}, timestamp {:.1}, cpu or unknown {:.1}; queue fence waits {:.1}; unmet fence events {:.1}; null views made {:.1} in {:.1} submissions; submissions using a copy the one before wrote {:.1}; title in ExecuteCommandLists {:.2} ms; overlap hazards: submissions {:.1}, changed copies in use {:.1}, heap pages {:.1}, marked after marked {:.1}, snapshots of written copies {:.1}, marks on written copies {:.1}",
        c[C::FlightPrepNs as usize] / 1e6,
        c[C::FlightSyncHeapsNs as usize] / 1e6,
        c[C::FlightGpuWaitNs as usize] / 1e6,
        c[C::FlightSettleNs as usize] / 1e6,
        c[C::DrainRange as usize],
        c[C::DrainFence as usize],
        c[C::DrainSyncExec as usize],
        c[C::DrainPresent as usize],
        c[C::DrainFree as usize],
        c[C::FlightFencePolls as usize],
        c[C::DrainReal as usize],
        c[C::RangeClean as usize],
        c[C::RangeSame as usize],
        c[C::RangeAlias as usize],
        c[C::RangeStaging as usize],
        c[C::RangeCopy as usize],
        c[C::RangeGpuWrite as usize],
        c[C::HeapInFlight as usize],
        c[C::WaitsValue as usize],
        c[C::WaitsHostValue as usize],
        c[C::WaitsTimestamp as usize],
        c[C::WaitsCpu as usize],
        c[C::QueueFenceWaits as usize],
        c[C::FenceEventsUnmet as usize],
        c[C::MadeReadSlots as usize],
        c[C::MadeReadSubs as usize],
        c[C::SharedCopySubs as usize],
        c[C::TitleExecNs as usize] / 1e6,
        c[C::PrepHazardSubs as usize],
        c[C::PrepDirtyInUse as usize],
        c[C::PrepHeapPages as usize],
        c[C::PrepMarkedAfterMarked as usize],
        c[C::PrepSnapshotWritten as usize],
        c[C::PrepMarkedWritten as usize],
    );
    let kinds = [
        C::GuardFaultsPool as usize,
        C::GuardFaultsPoolArmed as usize,
        C::GuardFaultsView as usize,
        C::GuardFaultsFixed as usize,
        C::GuardFaultsOther as usize,
        C::MeshIndirectDropped as usize,
    ];
    if kinds.iter().any(|&k| c[k] != 0.0) {
        tracing::info!(
            "  guard faults by kind per frame: pool {:.0} (armed {:.0}), view {:.0}, fixed {:.0}, other {:.0}; {:.0} indirect mesh dispatches dropped",
            c[C::GuardFaultsPool as usize],
            c[C::GuardFaultsPoolArmed as usize],
            c[C::GuardFaultsView as usize],
            c[C::GuardFaultsFixed as usize],
            c[C::GuardFaultsOther as usize],
            c[C::MeshIndirectDropped as usize],
        );
    }
    let times = crate::watch::held_times();
    let handler_hold = c[C::HandlerHoldNs as usize] / 1e6;
    let all_held: f64 = times.iter().map(|(_, total, _)| total / frames).sum();
    let held: Vec<String> = times
        .iter()
        .map(|(line, total, longest)| format!("watch.rs:{line} {:.2} ms (longest {longest:.3})", total / frames))
        .collect();
    tracing::info!(
        "  guard lock per frame: handler waiting {:.2} ms (longest wait {:.3} ms), holding {:.2} ms (saving {:.0} pages {:.2} ms, longest {:.3}; waiting for trackers {:.2} ms; opening {:.2} ms, longest {:.3}); others holding {:.2} ms; held by line: {}",
        c[C::HandlerWaitNs as usize] / 1e6,
        HANDLER_WAIT_MAX.swap(0, Relaxed) as f64 / 1e6,
        handler_hold,
        c[C::HandlerSaves as usize],
        c[C::HandlerSaveNs as usize] / 1e6,
        HANDLER_SAVE_MAX.swap(0, Relaxed) as f64 / 1e6,
        c[C::HandlerTrackerWaitNs as usize] / 1e6,
        c[C::HandlerOpenNs as usize] / 1e6,
        HANDLER_OPEN_MAX.swap(0, Relaxed) as f64 / 1e6,
        (all_held - handler_hold).max(0.0),
        held.join(", ")
    );
    let mut snapshots: Vec<((u64, u64), u64)> =
        SNAPSHOTS.lock().unwrap_or_else(|e| e.into_inner()).take().into_iter().flatten().collect();
    snapshots.sort_by_key(|&(_, bytes)| std::cmp::Reverse(bytes));
    let snapshots: Vec<String> = snapshots
        .iter()
        .take(6)
        .map(|((start, len), bytes)| format!("{start:#x}+{len:#x} {:.1} MB", *bytes as f64 / 1e6 / frames))
        .collect();
    tracing::info!("  snapshots per frame by copy: {}", snapshots.join(", "));
    let ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut ranges: Vec<_> = ranges.into_iter().flatten().collect();
    ranges.sort_by_key(|&(_, (_, bytes, _, _))| std::cmp::Reverse(bytes));
    for ((src, len), (n, bytes, us, _)) in ranges.into_iter().take(10) {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery(src as *const _, &mut info, size_of_val(&info)) };
        let kind = info.Type;
        tracing::info!(
            "  copied from {src:#x}+{len:#x} (memory {kind:#x}): {:.1}x/frame, {:.2} MB/frame, {:.2} ms/frame",
            n as f64 / frames,
            bytes as f64 / 1e6 / frames,
            us as f64 / 1e3 / frames
        );
    }
}

/// A global allocator that counts the bytes Rust code holds (the
/// statistics' "our heap"; the binary installs it in diagnostics builds).
pub struct CountingAlloc;

/// Bytes [`CountingAlloc`] has handed out and not taken back.
pub(crate) static HEAP_BYTES: AtomicU64 = AtomicU64::new(0);

/// Allocations of at least this many bytes name their caller, once per
/// power-of-two size class (who holds the big blocks of our heap).
const LARGE: usize = 32 << 20;

/// Logs the stack of a large allocation of `size` bytes, the first of its
/// size class.
fn note_large(size: usize) {
    static SEEN: std::sync::Mutex<u64> = std::sync::Mutex::new(0);
    std::thread_local!(static INSIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) });
    let class = 1u64 << size.ilog2().min(63);
    if INSIDE.with(|i| i.replace(true)) {
        return;
    }
    let first = {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let first = *seen & class == 0;
        *seen |= class;
        first
    };
    if first {
        let trace = std::backtrace::Backtrace::force_capture().to_string();
        let frames: Vec<&str> = trace.lines().filter(|l| l.contains("victory")).take(24).collect();
        tracing::info!(mb = size >> 20, "a large heap allocation (first of its size class): {}", frames.join(" | "));
    }
    INSIDE.with(|i| i.set(false));
}

// SAFETY: every call goes to the system allocator unchanged; only counted.
unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        // SAFETY: the caller's layout, passed on.
        let p = unsafe { std::alloc::System.alloc(layout) };
        if !p.is_null() {
            HEAP_BYTES.fetch_add(layout.size() as u64, Relaxed);
            if layout.size() >= LARGE {
                note_large(layout.size());
            }
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        // SAFETY: as above.
        let p = unsafe { std::alloc::System.alloc_zeroed(layout) };
        if !p.is_null() {
            HEAP_BYTES.fetch_add(layout.size() as u64, Relaxed);
            if layout.size() >= LARGE {
                note_large(layout.size());
            }
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: std::alloc::Layout) {
        HEAP_BYTES.fetch_sub(layout.size() as u64, Relaxed);
        // SAFETY: a block this allocator handed out with this layout.
        unsafe { std::alloc::System.dealloc(p, layout) }
    }

    unsafe fn realloc(&self, p: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        // SAFETY: as above.
        let q = unsafe { std::alloc::System.realloc(p, layout, size) };
        if !q.is_null() {
            HEAP_BYTES.fetch_sub(layout.size() as u64, Relaxed);
            HEAP_BYTES.fetch_add(size as u64, Relaxed);
            if size >= LARGE {
                note_large(size);
            }
        }
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A page our copy-back wrote counts as ours in that frame and the
    /// next, not after, and not for a page that shares its slot.
    #[test]
    fn a_page_we_wrote_is_ours_for_two_frames() {
        let page = 0x1_2345;
        let entry = ours_entry(page, 100);
        assert!(ours_recent(entry, page, 100) && ours_recent(entry, page, 101));
        assert!(!ours_recent(entry, page, 102) && !ours_recent(entry, page, 99));
        assert!(!ours_recent(entry, page + 1, 100), "another page");
        assert!(!ours_recent(u64::MAX, page, 100), "a slot never written");
        // The frame count wraps in its bits.
        let last = FRAME_MASK;
        let entry = ours_entry(page, last);
        assert!(ours_recent(entry, page, last + 1) && !ours_recent(entry, page, last + 2));
        // Slots are inside the table, and pages next to each other apart.
        let slots: Vec<usize> = (0..1000).map(|p| ours_slot(0x10_0000 + p)).collect();
        assert!(slots.iter().all(|&s| s < OURS_SLOTS));
        let mut apart = slots.clone();
        apart.sort_unstable();
        apart.dedup();
        assert!(apart.len() > 990, "{} of 1000 slots", apart.len());
    }
}
