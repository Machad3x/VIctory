//! The title's acceleration-structure builder waits for its pipelines.
//!
//! `XboxBvhBuilder.dll` builds and refits the ray-tracing structures in
//! title memory every frame through some fifty compute programs, one
//! dispatch after another, and the title's queues wait on what follows
//! them. A dispatch whose host pipeline is not made within the frame's
//! budget is dropped (`State::dispatch`), and with a cold pipeline cache
//! many of the builder's are still being made when it first runs: the
//! stages that run then work on what the dropped ones never wrote.
//!
//! So a builder dispatch first waits for every pipeline the builder has
//! created, queued ahead of all others, up to [`WAIT`] a frame (below the
//! title's 30 s watchdog): its frames run whole. Nothing is held back: a
//! frame that would skip the builder's dispatches leaves what comes after
//! them in its submission (labels, cross-queue waits) waiting on work that
//! never ran, and the title's watchdog ends the run (gpu307). A pipeline
//! refused for good, or one not made within [`WAIT`], still drops its own
//! dispatches only, said once.
//!
//! Its pipelines are known by where they come from: created by its code (a
//! return address in its image on the stack) or from bytecode in its image.

use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use super::make::{self, Pending};

/// The builder's image, `[start, end)`, once mapped.
static IMAGE: Mutex<Option<(usize, usize)>> = Mutex::new(None);

/// The builder's file name.
const FILE: &str = "XboxBvhBuilder.dll";

/// A title DLL was mapped at `base`, `size` bytes: noted if it is the
/// builder.
pub fn note_title_dll(file: &str, base: usize, size: usize) {
    if file.eq_ignore_ascii_case(FILE) {
        tracing::info!(base = format_args!("{base:#x}"), size, "BVH builder mapped");
        *IMAGE.lock().unwrap_or_else(|e| e.into_inner()) = Some((base, base + size));
    }
}

fn image() -> Option<(usize, usize)> {
    *IMAGE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the pipeline being created is the builder's: `code` (the title's
/// bytecode pointers) lies in its image, or its code is among our callers.
#[inline(never)]
pub(crate) fn created_by_builder(code: &[*const u8]) -> bool {
    let Some((start, end)) = image() else {
        return false;
    };
    let inside = |a: usize| (start..end).contains(&a);
    if code.iter().any(|&c| inside(c as usize)) {
        return true;
    }
    callers().into_iter().any(inside)
}

/// Return addresses on the stack, ours first.
fn callers() -> Vec<usize> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
        let mut frames = [std::ptr::null_mut(); 24];
        // SAFETY: room for that many frames.
        let n = unsafe {
            RtlCaptureStackBackTrace(
                1,
                frames.len() as u32,
                frames.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        frames[..usize::from(n)]
            .iter()
            .map(|&f| f as usize)
            .collect()
    }
    #[cfg(not(windows))]
    Vec::new()
}

/// The builder's pipelines (alive while the title holds them).
static PIPELINES: Mutex<Vec<Weak<Pending>>> = Mutex::new(Vec::new());

/// The longest the builder's dispatches of one frame wait for its
/// pipelines, below the title's 30 s watchdog.
pub(crate) const WAIT: Duration = Duration::from_secs(20);

/// The builder created pipeline `p`: its dispatches wait for it
/// ([`wait_for_pipelines`]), and it is made at once.
pub(crate) fn note(p: &Arc<Pending>) {
    PIPELINES.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(p));
    make::begin(p);
}

/// A dispatch of the builder's pipeline `p` was not run on the host (`why`):
/// said, the first times. Nothing else of the builder is held back.
pub(crate) fn dropped(_p: &Arc<Pending>, why: &str) {
    static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
        tracing::warn!(frame = crate::frame::current(), why, "a dispatch of the BVH builder was not run");
    }
}

/// Before a dispatch of the builder's: waits until every pipeline it has
/// created is made or refused, queued ahead of the others, for what is
/// left of [`WAIT`] in this frame. One lock and a scan once all are made.
pub(crate) fn wait_for_pipelines() {
    let unmade: Vec<Arc<Pending>> = {
        let mut pipelines = PIPELINES.lock().unwrap_or_else(|e| e.into_inner());
        pipelines.retain(|p| p.strong_count() > 0);
        pipelines.iter().filter_map(Weak::upgrade).filter(|p| make::state(p).is_none()).collect()
    };
    if unmade.is_empty() {
        return;
    }
    let left = left_in_frame(crate::frame::current(), Duration::ZERO);
    if left.is_zero() {
        return;
    }
    let started = Instant::now();
    let until = started + left;
    for p in &unmade {
        make::wait_made(p, until);
    }
    let waited = started.elapsed();
    left_in_frame(crate::frame::current(), waited);
    let still = unmade.iter().filter(|p| make::state(p).is_none()).count();
    let refused = unmade.iter().filter(|p| make::state(p) == Some(false)).count();
    tracing::info!(
        frame = crate::frame::current(),
        waited_for = unmade.len(),
        still_being_made = still,
        refused,
        ms = waited.as_millis() as u64,
        "BVH builder waited for its pipelines"
    );
}

/// What is left of [`WAIT`] in `frame`, after counting `waited` in it.
fn left_in_frame(frame: u64, waited: Duration) -> Duration {
    static WAITED: Mutex<(u64, Duration)> = Mutex::new((u64::MAX, Duration::ZERO));
    left_of(&mut WAITED.lock().unwrap_or_else(|e| e.into_inner()), frame, waited)
}

/// [`left_in_frame`] with `w` (the frame, and what was waited in it).
fn left_of(w: &mut (u64, Duration), frame: u64, waited: Duration) -> Duration {
    if w.0 != frame {
        *w = (frame, Duration::ZERO);
    }
    w.1 += waited;
    WAIT.saturating_sub(w.1)
}

/// Tests that set the builder's image, one at a time.
#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame's waits share [`WAIT`]; a new frame starts it anew.
    #[test]
    fn a_frame_s_waits_share_the_allowance() {
        let mut w = (u64::MAX, Duration::ZERO);
        assert_eq!(left_of(&mut w, 7, Duration::ZERO), WAIT);
        assert_eq!(left_of(&mut w, 7, Duration::from_secs(15)), Duration::from_secs(5));
        assert_eq!(left_of(&mut w, 7, Duration::from_secs(15)), Duration::ZERO);
        assert_eq!(left_of(&mut w, 8, Duration::ZERO), WAIT, "another frame");
    }

    /// Pipelines are the builder's by their bytecode in its image, or by
    /// its code among the callers; not otherwise.
    #[test]
    fn builder_pipelines_are_known_by_their_origin() {
        let _one = test_lock();
        // A range that holds this test's own code (a caller of the check).
        let here = builder_pipelines_are_known_by_their_origin as *const () as usize;
        let elsewhere = [0x7_0000_0000usize as *const u8];
        *IMAGE.lock().unwrap() = None;
        assert!(!created_by_builder(&elsewhere), "no builder mapped");
        // Bytecode inside the image.
        *IMAGE.lock().unwrap() = Some((0x7_0000_0000, 0x7_0001_0000));
        assert!(created_by_builder(&elsewhere));
        assert!(!created_by_builder(&[0x8_0000_0000usize as *const u8]));
        // A caller inside the image: this function's own code.
        *IMAGE.lock().unwrap() = Some((here - 0x10_0000, here + 0x10_0000));
        assert!(created_by_builder(&[]));
        *IMAGE.lock().unwrap() = None;
    }
}
