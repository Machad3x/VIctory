//! A submission's refreshes, copied on several threads.
//!
//! Bringing host copies of title memory up to date is a memory copy, and
//! one thread copies far below what the memory can move: on an RTX 5070 Ti
//! host (DDR5), the world's ~8,800 scattered 64 KiB pages per frame
//! (550 MB) took 60 ms on one thread, 43 on two and 37 on four, where the
//! memory is saturated. A submission's refreshes ([`queued`]) queue their
//! copies, and [`run`] copies them on a few workers with the submitting
//! thread; it returns once all are done, so what follows sees them all.

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// Called once a copy's bytes landed: `(f, context)`, as
/// `f(context, destination, bytes)`, on the thread that asked for the copy
/// (or called [`run`]), never on a worker: a hook may block (on a lock a
/// riding pull holds over its GPU wait) without holding up the pool.
pub(crate) type Then = (fn(usize, usize, usize), usize);

/// A copy of title memory into host memory: destination, source, bytes,
/// and what to call once they landed.
pub(crate) type Piece = (usize, usize, usize, Option<Then>);

thread_local! {
    static QUEUE: RefCell<Option<Vec<Piece>>> = const { RefCell::new(None) };
    /// The list of a [`run`] that ended, emptied, for the next [`queued`]:
    /// a submission queues and runs a few times, each with a list of
    /// thousands of pieces.
    static SPARE: RefCell<Vec<Piece>> = const { RefCell::new(Vec::new()) };
}

/// Keeps `pieces`, emptied, for this thread's next [`queued`].
fn keep(mut pieces: Vec<Piece>) {
    pieces.clear();
    SPARE.with(|s| {
        let mut s = s.borrow_mut();
        if s.capacity() < pieces.capacity() {
            *s = pieces;
        }
    });
}

/// Runs `f`; the copies [`copy`] is asked for on this thread meanwhile are
/// not made but returned, for [`run`].
pub(crate) fn queued<R>(f: impl FnOnce() -> R) -> (R, Vec<Piece>) {
    let spare = SPARE.with(|s| std::mem::take(&mut *s.borrow_mut()));
    let outer = QUEUE.with(|q| q.replace(Some(spare)));
    let result = f();
    let pieces = QUEUE.with(|q| q.replace(outer)).unwrap_or_default();
    (result, pieces)
}

/// Copies `len` bytes of title memory at `src` into host memory at `dst`
/// ([`crate::host::copy_mapped`]), or queues the copy inside [`queued`].
/// The caller must not read the destination before [`run`] then.
pub(crate) fn copy(dst: *mut u8, src: *const u8, len: usize) {
    copy_then(dst, src, len, None);
}

/// As [`copy`], calling `then` once the bytes landed (at once, or for a
/// queued copy in [`run`], after the pool's part and before it returns).
pub(crate) fn copy_then(dst: *mut u8, src: *const u8, len: usize, then: Option<Then>) {
    let piece = (dst as usize, src as usize, len, then);
    let queued = QUEUE.with(|q| q.borrow_mut().as_mut().map(|q| q.push(piece)).is_some());
    if !queued {
        land(piece);
    }
}

/// Makes one copy, then calls its `then`.
fn land((dst, src, len, then): Piece) {
    crate::host::copy_mapped(dst as *mut u8, src as *const u8, len, false);
    if let Some((f, context)) = then {
        f(context, dst, len);
    }
}

/// Copies are cut to this size, so that one big copy is shared too.
const PIECE: usize = 1 << 20;
/// Below this many bytes in all, one thread copies (waking the workers
/// costs tens of microseconds; 4 MiB take about 0.4 ms).
const PARALLEL_FROM: usize = 4 << 20;
/// Workers besides the submitting thread: four threads saturated the
/// memory above, eight gained 2%.
const WORKERS: usize = 3;

/// Tests: every copy on the calling thread (to compare with the workers).
/// Tests that set it, or need the workers, hold [`TEST_LOCK`].
#[cfg(test)]
pub(crate) static SERIAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Tests: held while a test sets [`SERIAL`] or another switch of the
/// refreshes' threads, so that two such tests do not interleave.
#[cfg(test)]
pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

/// Tests: [`TEST_LOCK`], also after a test that held it failed.
#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn serial() -> bool {
    #[cfg(test)]
    return SERIAL.load(Ordering::Relaxed);
    #[cfg(not(test))]
    false
}

/// Tests: the submitting thread leaves every piece to the workers, and the
/// pieces they copied.
#[cfg(test)]
static WORKERS_ALONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[cfg(test)]
static WORKER_PIECES: AtomicUsize = AtomicUsize::new(0);

fn caller_helps() -> bool {
    #[cfg(test)]
    return !WORKERS_ALONE.load(Ordering::Relaxed);
    #[cfg(not(test))]
    true
}

/// Makes the copies [`queued`] returned, on several threads where they are
/// many bytes; returns once all are made.
pub(crate) fn run(mut pieces: Vec<Piece>) {
    let total: usize = pieces.iter().map(|p| p.2).sum();
    let pool = (total >= PARALLEL_FROM && !serial()).then(pool).flatten();
    let Some(pool) = pool else {
        for piece in pieces.drain(..) {
            land(piece);
        }
        keep(pieces);
        return;
    };
    let mut cut = Vec::with_capacity(pieces.len() + total / PIECE);
    let mut thens = Vec::new();
    for (dst, src, len, then) in pieces.drain(..) {
        if let Some(then) = then {
            thens.push((then, dst, len));
        }
        let mut at = 0;
        while at < len {
            let n = (len - at).min(PIECE);
            cut.push((dst + at, src + at, n, None));
            at += n;
        }
    }
    keep(pieces);
    pool.run(cut);
    // After the pool's run released it: a hook waiting on a lock holds up
    // only this caller, not the next run.
    for ((f, context), dst, len) in thens {
        f(context, dst, len);
    }
}

/// One [`run`]'s pieces, taken by index by whichever thread is free.
struct Job {
    pieces: Vec<Piece>,
    next: AtomicUsize,
    done: AtomicUsize,
}

impl Job {
    /// Copies pieces until none are left to take (`worker`: on a worker).
    #[cfg_attr(not(test), allow(unused_variables))]
    fn work(&self, worker: bool) {
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            let Some(&piece) = self.pieces.get(i) else {
                return;
            };
            land(piece);
            #[cfg(test)]
            if worker {
                WORKER_PIECES.fetch_add(1, Ordering::Relaxed);
            }
            self.done.fetch_add(1, Ordering::Release);
        }
    }
}

struct Pool {
    /// The current job and its number (workers take each job once).
    job: Mutex<(u64, Option<Arc<Job>>)>,
    wake: Condvar,
    /// Signalled as workers finish their part of a job.
    finished: Condvar,
    /// One run at a time.
    running: Mutex<()>,
}

impl Pool {
    fn run(&self, pieces: Vec<Piece>) {
        let _one = self.running.lock().unwrap_or_else(|e| e.into_inner());
        let n = pieces.len();
        let job = Arc::new(Job {
            pieces,
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
        });
        {
            let mut slot = self.job.lock().unwrap_or_else(|e| e.into_inner());
            slot.0 += 1;
            slot.1 = Some(job.clone());
        }
        self.wake.notify_all();
        if caller_helps() {
            job.work(false);
        }
        let mut slot = self.job.lock().unwrap_or_else(|e| e.into_inner());
        while job.done.load(Ordering::Acquire) < n {
            slot = self.finished.wait(slot).unwrap_or_else(|e| e.into_inner());
        }
        slot.1 = None;
    }

    fn worker(&self) {
        let mut seen = 0;
        loop {
            let job = {
                let mut slot = self.job.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    if slot.0 != seen
                        && let Some(job) = &slot.1
                    {
                        seen = slot.0;
                        break job.clone();
                    }
                    slot = self.wake.wait(slot).unwrap_or_else(|e| e.into_inner());
                }
            };
            job.work(true);
            // Under the lock, so the waiting thread cannot miss it.
            let _slot = self.job.lock().unwrap_or_else(|e| e.into_inner());
            self.finished.notify_all();
        }
    }
}

/// The workers, started on first use (None where threads cannot start).
fn pool() -> Option<&'static Pool> {
    static POOL: OnceLock<Option<&'static Pool>> = OnceLock::new();
    *POOL.get_or_init(|| {
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            job: Mutex::new((0, None)),
            wake: Condvar::new(),
            finished: Condvar::new(),
            running: Mutex::new(()),
        }));
        let started = (0..WORKERS)
            .filter(|i| {
                std::thread::Builder::new()
                    .name(format!("victory refresh {i}"))
                    .spawn(move || pool.worker())
                    .is_ok()
            })
            .count();
        (started > 0).then_some(pool)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Past the threshold the workers copy: with the submitting thread
    /// leaving them every piece, they copy all of a run's pieces, and every
    /// byte lands where it belongs.
    #[test]
    fn the_workers_take_the_pieces_past_the_threshold() {
        let len = 8 << 20;
        let src: Vec<u8> = (0..len).map(|i| (i * 13 + i / 65536) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr(), dst.as_mut_ptr());
        // 128 pieces of 64 KiB, in another order than in memory.
        let ((), pieces) = queued(|| {
            for k in 0..128usize {
                let at = (k * 37 % 128) * (64 << 10);
                // SAFETY: (both) inside the two buffers.
                copy(unsafe { d.add(at) }, unsafe { s.add(at) }, 64 << 10);
            }
        });
        assert!(pool().is_some(), "workers started");
        // Not with a test that copies on the calling thread meanwhile.
        let _serial = test_lock();
        WORKERS_ALONE.store(true, Ordering::Relaxed);
        let before = WORKER_PIECES.load(Ordering::Relaxed);
        run(pieces);
        let copied = WORKER_PIECES.load(Ordering::Relaxed) - before;
        WORKERS_ALONE.store(false, Ordering::Relaxed);
        // At least: runs of other tests meanwhile count too.
        assert!(copied >= 128, "pieces the workers copied: {copied}");
        assert!(dst == src, "every byte where it belongs");
    }

    /// A copy's `then` is called once its bytes landed, once per copy (a
    /// big one cut into pieces too), on the thread that ran them.
    #[test]
    fn then_follows_each_piece_once_its_bytes_landed() {
        static LANDED: AtomicUsize = AtomicUsize::new(0);
        static WRONG: AtomicUsize = AtomicUsize::new(0);
        // `context`: source minus destination; the destination must hold
        // the source's bytes.
        fn landed(delta: usize, dst: usize, len: usize) {
            let src = dst.wrapping_add(delta);
            // SAFETY: (both) `len` bytes inside the test's buffers.
            let (s, d) = unsafe {
                (
                    std::slice::from_raw_parts(src as *const u8, len),
                    std::slice::from_raw_parts(dst as *const u8, len),
                )
            };
            let on_worker = std::thread::current().name().is_some_and(|n| n.starts_with("victory refresh"));
            if s != d || on_worker {
                WRONG.fetch_add(1, Ordering::Relaxed);
            }
            LANDED.fetch_add(len, Ordering::Relaxed);
        }
        let len = 12 << 20;
        let src: Vec<u8> = (0..len).map(|i| (i * 11 + i / 4096) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr(), dst.as_mut_ptr());
        let delta = (s as usize).wrapping_sub(d as usize);
        let ((), pieces) = queued(|| {
            // One big copy (cut into pieces) and two small ones.
            // SAFETY: inside the two buffers.
            let (d1, s1, d9, s9) = unsafe { (d.add(8 << 20), s.add(8 << 20), d.add(9 << 20), s.add(9 << 20)) };
            copy_then(d, s, 8 << 20, Some((landed, delta)));
            copy_then(d1, s1, 1 << 20, Some((landed, delta)));
            copy(d9, s9, 3 << 20);
        });
        assert_eq!(LANDED.load(Ordering::Relaxed), 0, "not before the run");
        // Not with a test switching the pool's threads meanwhile.
        let _serial = test_lock();
        run(pieces);
        assert_eq!(LANDED.load(Ordering::Relaxed), 9 << 20, "every byte once");
        assert_eq!(WRONG.load(Ordering::Relaxed), 0, "called after the bytes landed, not on a worker");
        assert!(dst == src, "every byte where it belongs");
        // Outside `queued`: at once.
        dst[..16].fill(0);
        copy_then(d, s, 16, Some((landed, delta)));
        assert_eq!(LANDED.load(Ordering::Relaxed), (9 << 20) + 16);
    }

    /// Queued copies are made by `run`, on the workers past the threshold,
    /// one big copy cut into pieces too, each byte where it belongs.
    #[test]
    fn queued_copies_are_made_on_the_workers() {
        let len = 24 << 20;
        let src: Vec<u8> = (0..len).map(|i| (i * 7 + i / 4096) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr(), dst.as_mut_ptr());
        // Scattered 64 KiB pieces, then one 8 MiB copy.
        let ((), pieces) = queued(|| {
            for k in 0..128usize {
                let at = (k * 37 % 256) * (64 << 10);
                // SAFETY: (both) inside the two buffers.
                copy(unsafe { d.add(at) }, unsafe { s.add(at) }, 64 << 10);
            }
            // SAFETY: inside the two buffers.
            copy(
                unsafe { d.add(16 << 20) },
                unsafe { s.add(16 << 20) },
                8 << 20,
            );
        });
        assert_eq!(pieces.len(), 129, "queued, not made");
        assert!(dst.iter().all(|&b| b == 0));
        run(pieces);
        let mut want = vec![0u8; len];
        for k in 0..128usize {
            let at = (k * 37 % 256) * (64 << 10);
            want[at..at + (64 << 10)].copy_from_slice(&src[at..at + (64 << 10)]);
        }
        want[16 << 20..].copy_from_slice(&src[16 << 20..]);
        assert!(dst == want, "every byte where it belongs");
        // Outside `queued`, a copy is made at once.
        copy(d, s, 16);
        assert_eq!(dst[..16], src[..16]);
        // Again, to reuse the workers.
        // SAFETY: (both) inside the two buffers.
        let (at, from) = unsafe { (d.add(1 << 20), s.add(1 << 20)) };
        let ((), pieces) = queued(|| copy(at, from, 6 << 20));
        dst[1 << 20..7 << 20].fill(0);
        run(pieces);
        assert!(dst[1 << 20..7 << 20] == src[1 << 20..7 << 20]);
    }
}
