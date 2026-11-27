//! Where the bytes the GPU uses live, and how they move.
//!
//! The console has one memory; the host has system memory and video
//! memory with a bus between them. Each buffer here has one [`Class`],
//! and each class one way in and one way out: the CPU writes video memory
//! through the BAR and never reads it, the GPU's results come back through
//! read-back memory, and only what no copy covers stays where the title
//! put it.
//!
//! `VICTORY_MEMSYS=1` turns the new paths on ([`on`]); the counters are
//! kept either way, so that a run without it gives the numbers to compare
//! with ([`frame`]).
// The ring, queue, dirty set and gate are wired with the submissions'
// tickets; until then only their tests use them.
#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

#[cfg(windows)]
pub(crate) mod diff;
pub(crate) mod dirty;
pub(crate) mod gate;
#[cfg(windows)]
pub(crate) mod heaps;
pub(crate) mod pages;
pub(crate) mod queue;
pub(crate) mod ring;

/// Whether the new memory paths are on: `VICTORY_MEMSYS=1`. Asked once.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::switch::on("VICTORY_MEMSYS");
        tracing::info!(on, "the memory system's new paths (VICTORY_MEMSYS)");
        on
    })
}

/// Whether one behaviour of the new paths is on: [`on`], and its own
/// switch `name` not set to `0` (to change one thing per run).
fn part(name: &str, cell: &OnceLock<bool>) -> bool {
    *cell.get_or_init(|| {
        let part = on() && crate::switch::on_by_default(name);
        if on() {
            tracing::info!(on = part, "memory system: {name}");
        }
        part
    })
}

/// Small host buffers placed in heaps made ahead (`heaps`):
/// `VICTORY_MEMSYS_POOL=0` keeps them committed one by one.
pub(crate) fn pooling() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    part("VICTORY_MEMSYS_POOL", &ON)
}

/// The last host submission sent to the GPU, and the last that has run.
static SENT: AtomicU64 = AtomicU64::new(0);
static DONE: AtomicU64 = AtomicU64::new(0);

/// The host submission numbered `value` goes to the GPU now.
pub(crate) fn submission_sent(value: u64) {
    SENT.fetch_max(value, Ordering::Release);
}

/// The host submission numbered `value` has run (it was waited for).
pub(crate) fn submission_done(value: u64) {
    DONE.fetch_max(value, Ordering::Release);
}

/// The number of the last host submission sent to the GPU.
pub(crate) fn sent() -> u64 {
    SENT.load(Ordering::Acquire)
}

/// The number of the last host submission that has run: memory only
/// submissions up to it used is the GPU's no more.
pub(crate) fn done() -> u64 {
    DONE.load(Ordering::Acquire)
}

/// Where a buffer's bytes live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Class {
    /// Video memory the CPU does not map: the GPU writes and reads it.
    Vram,
    /// Video memory the CPU writes through the BAR (a GPU upload heap) and
    /// never reads.
    Bar,
    /// Write-combined system memory the GPU reads once (staging).
    Upload,
    /// System memory the GPU copies into and the CPU reads.
    Readback,
    /// System memory both read and write; slow for the GPU.
    Shared,
    /// The title's own pages, opened as a heap.
    InPlace,
}

impl Class {
    const COUNT: usize = 6;
}

/// The heap type the title gave a buffer: all that is left of what it
/// meant the memory for (the console's own memory types are not kept).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Heap {
    Default,
    Upload,
    Readback,
}

/// What is known of a range of title memory the GPU uses.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Traits {
    /// A texture (tiled in title memory, so never usable where it lies).
    pub(crate) texture: bool,
    /// A title buffer covers it. Not covered, shaders reach it only at
    /// addresses they compute, and it has no copy.
    pub(crate) covered: bool,
    /// Shaders use atomics on it, or write it scattered.
    pub(crate) atomics: bool,
    pub(crate) bytes: u64,
    /// Bytes shaders read of it per frame (an estimate).
    pub(crate) gpu_read: u64,
    pub(crate) gpu_writes: bool,
    /// The title's CPU reads what shaders wrote there, after a fence.
    pub(crate) cpu_reads: bool,
    pub(crate) heap: Option<Heap>,
}

/// From this size a buffer is hot whatever is read of it: one pass over
/// it from system memory costs more than keeping it in video memory.
const HOT_BYTES: u64 = 1 << 20;

/// Where a range with these traits belongs, on a host with (`bar`) or
/// without video memory the CPU can write.
///
/// Video memory is the rule: shaders read it some thirty times faster
/// than system memory, and scattered writes or atomics over the bus have
/// removed the device. The title's own pages serve where there is no copy
/// at all (they read as zeros otherwise), and for small results the
/// title's CPU reads, which then need no copy back.
pub(crate) fn place(t: &Traits, bar: bool) -> Class {
    let video = if bar { Class::Bar } else { Class::Shared };
    if t.texture {
        return Class::Vram;
    }
    if !t.covered {
        return Class::InPlace;
    }
    if t.atomics || t.bytes >= HOT_BYTES || t.gpu_read >= t.bytes.saturating_mul(4) {
        return video;
    }
    if t.gpu_writes && (t.cpu_reads || t.heap == Some(Heap::Readback)) {
        return Class::InPlace;
    }
    video
}

/// What is counted besides buffers made.
#[derive(Debug, Clone, Copy)]
pub(crate) enum N {
    /// Upload buffers made for one copy of a list, and their bytes: what
    /// an upload ring would hold.
    ListUploads,
    ListUploadBytes,
    /// Read-back buffers made for one copy of a list, and their bytes.
    ListReadbacks,
    ListReadbackBytes,
    /// Buffers placed in a pool's heap, and their bytes.
    Pooled,
    PooledBytes,
    /// Buffers a pool had no room for (made committed).
    PoolFull,
    /// Marked pages a submission handed to the compare with twins
    /// (`diff`), those that had a twin to compare with, and those equal
    /// to it: not pulled or copied back.
    DiffPages,
    DiffCompared,
    DiffSame,
    /// Blocks of 256 bytes that differed in the pages compared.
    DiffBlocks,
    /// Pages equal to their twins that wrote bytes to title memory all
    /// the same (counted while verifying only): skipped, they were lost.
    DiffLost,
    /// Microseconds in the compare's list, its wait included.
    DiffUs,
}

const NS: usize = 13;
/// Per class a count and bytes, then the [`N`]s.
const TOTAL: usize = 2 * Class::COUNT + NS;

static COUNTS: [AtomicU64; TOTAL] = [const { AtomicU64::new(0) }; TOTAL];

pub(crate) fn add(n: N, by: u64) {
    COUNTS[2 * Class::COUNT + n as usize].fetch_add(by, Ordering::Relaxed);
}

/// A committed buffer of `bytes` was made in `class`.
pub(crate) fn made(class: Class, bytes: u64) {
    COUNTS[2 * class as usize].fetch_add(1, Ordering::Relaxed);
    COUNTS[2 * class as usize + 1].fetch_add(bytes, Ordering::Relaxed);
}

/// Frames between two lines of [`frame`].
const EVERY: u64 = 512;

/// A frame was presented: every [`EVERY`] frames, what was counted since
/// the last line, per frame (nothing when nothing was).
pub(crate) fn frame(frame: u64) {
    if !frame.is_multiple_of(EVERY) {
        return;
    }
    static LAST: Mutex<[u64; TOTAL]> = Mutex::new([0; TOTAL]);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let mut per = [0f64; TOTAL];
    let mut any = false;
    for (i, count) in COUNTS.iter().enumerate() {
        let now = count.load(Ordering::Relaxed);
        per[i] = now.wrapping_sub(last[i]) as f64 / EVERY as f64;
        any |= now != last[i];
        last[i] = now;
    }
    drop(last);
    if !any {
        return;
    }
    const MIB: f64 = (1u64 << 20) as f64;
    let class = |c: Class| (per[2 * c as usize], per[2 * c as usize + 1] / MIB);
    let n = |n: N| per[2 * Class::COUNT + n as usize];
    tracing::info!(
        frame,
        on = on(),
        "memory system per frame over the last {EVERY} (count, MiB): committed buffers made in video memory \
         {:.2?}, BAR {:.2?}, upload {:.2?}, read-back {:.2?}, shared {:.2?}; list uploads {:.2} {:.2}, list \
         read-backs {:.2} {:.2}; pooled {:.2} {:.2}, no room in a pool {:.2}; marked pages {:.1}, compared with a twin {:.1}, \
         equal to it {:.1}, of them written to title memory all the same {:.2}, blocks of 256 bytes that differed \
         {:.1}, in the compare {:.2} ms",
        class(Class::Vram),
        class(Class::Bar),
        class(Class::Upload),
        class(Class::Readback),
        class(Class::Shared),
        n(N::ListUploads),
        n(N::ListUploadBytes) / MIB,
        n(N::ListReadbacks),
        n(N::ListReadbackBytes) / MIB,
        n(N::Pooled),
        n(N::PooledBytes) / MIB,
        n(N::PoolFull),
        n(N::DiffPages),
        n(N::DiffCompared),
        n(N::DiffSame),
        n(N::DiffLost),
        n(N::DiffBlocks),
        n(N::DiffUs) / 1e3,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_are_placed_by_what_uses_them() {
        let small = Traits {
            texture: false,
            covered: true,
            atomics: false,
            bytes: 64 << 10,
            gpu_read: 64 << 10,
            gpu_writes: false,
            cpu_reads: false,
            heap: None,
        };
        // Read by shaders, written by the CPU: video memory through the
        // BAR, or system memory where the host has none.
        assert_eq!(place(&small, true), Class::Bar);
        assert_eq!(place(&small, false), Class::Shared);
        assert_eq!(place(&Traits { texture: true, covered: false, ..small }, true), Class::Vram);
        // No copy at all: the title's pages, whatever else is known.
        let uncovered = Traits { covered: false, atomics: true, bytes: 64 << 20, ..small };
        assert_eq!(place(&uncovered, true), Class::InPlace);
        // A small result the CPU reads stays where the title reads it,
        // named so by a counter or by the title's heap type.
        let result = Traits { gpu_writes: true, cpu_reads: true, ..small };
        assert_eq!(place(&result, true), Class::InPlace);
        let by_heap = Traits { gpu_writes: true, heap: Some(Heap::Readback), ..small };
        assert_eq!(place(&by_heap, true), Class::InPlace);
        // Never under atomics, when hot, or when large.
        assert_eq!(place(&Traits { atomics: true, ..result }, true), Class::Bar);
        assert_eq!(place(&Traits { gpu_read: 256 << 10, ..result }, true), Class::Bar);
        assert_eq!(place(&Traits { bytes: 1 << 20, gpu_read: 0, ..result }, true), Class::Bar);
        // Shaders write it and nobody reads it on the CPU: video memory.
        assert_eq!(place(&Traits { gpu_writes: true, ..small }, true), Class::Bar);
        assert_eq!(place(&Traits { heap: Some(Heap::Upload), ..small }, true), Class::Bar);
    }

    /// Each counter has a slot of its own.
    #[test]
    fn counters_do_not_share_slots() {
        let before: Vec<u64> = COUNTS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        made(Class::InPlace, 7);
        add(N::PoolFull, 5);
        let after: Vec<u64> = COUNTS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        // Other tests count buffers meanwhile, never in place ones; a
        // pool may be full meanwhile only with the pools switched on.
        assert_eq!(after[2 * Class::InPlace as usize] - before[2 * Class::InPlace as usize], 1);
        assert_eq!(after[2 * Class::InPlace as usize + 1] - before[2 * Class::InPlace as usize + 1], 7);
        let full = 2 * Class::COUNT + N::PoolFull as usize;
        assert!(after[full] - before[full] >= 5);
        assert_eq!(N::DiffUs as usize, NS - 1, "NS counts every N");
    }
}
