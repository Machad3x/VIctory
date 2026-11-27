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

/// Refreshes of copies in video memory written by the copy workers with
/// a submission's other copies: `VICTORY_MEMSYS_VIDEO=0` keeps them on the
/// submitting thread.
pub(crate) fn video_later() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    part("VICTORY_MEMSYS_VIDEO", &ON)
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
    /// Refreshes of video memory left to the copy workers, and their
    /// bytes.
    VideoLater,
    VideoLaterBytes,
}

const NS: usize = 9;
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
         read-backs {:.2} {:.2}; pooled {:.2} {:.2}, no room in a pool {:.2}; video memory refreshes left to \
         the copy workers {:.1} {:.2}",
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
        n(N::VideoLater),
        n(N::VideoLaterBytes) / MIB,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each counter has a slot of its own.
    #[test]
    fn counters_do_not_share_slots() {
        let before: Vec<u64> = COUNTS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        made(Class::InPlace, 7);
        add(N::VideoLaterBytes, 5);
        let after: Vec<u64> = COUNTS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        // At least: other tests count buffers meanwhile, never these two.
        assert_eq!(after[2 * Class::InPlace as usize] - before[2 * Class::InPlace as usize], 1);
        assert_eq!(after[2 * Class::InPlace as usize + 1] - before[2 * Class::InPlace as usize + 1], 7);
        assert_eq!(after[TOTAL - 1] - before[TOTAL - 1], 5);
        assert_eq!(N::VideoLaterBytes as usize, NS - 1, "NS counts every N");
    }
}
