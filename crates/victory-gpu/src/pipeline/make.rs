//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and the
//! title's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//! Pipelines a draw waits for go ahead of those started early (at the
//! title's creation, because an earlier run drew them).
//!
//! Passes over the screen ([`GraphicsState::screen_pass`]) go ahead of
//! every other pipeline and have a longer budget of their own: the
//! title fills its lighting tables (`PS_IntegrateEnvBRDF*`,
//! `PS_AreaLightIlluminance`) with such draws once, at the world's first
//! frame, behind more than a thousand other pipelines. Dropped, the tables
//! stay empty for the whole run. Idea from Belobug's fork (18e57db).
//! Compute pipelines wait in the same class, for a one-off dispatch that
//! would be lost (a compute pass keeps state from frame to frame). Runs
//! gpu282 to gpu284 waited for none in the opening scene once warm, so it
//! is not what lights the interior there. With a cold cache
//! frames then took 15 to 26 s against about 1 s. `VICTORY_COMPUTE_WAIT=0`
//! turns it off. (Idea from Biskwit/VIctory-reloaded 0b6c0e46.)
//!
//! All these waits of one presented frame share one [`ALLOWANCE`], well
//! under the title's 30 s watchdog: screen passes 20 s, compute 20 s and
//! the builder's 20 s, each a frame (a frame as counted by the title's
//! frame waits, which go on while a submission waits), added up to more
//! than 30 s with no present. A pipeline being waited for goes to the front
//! of its queue (one a draw gave up on stays behind it), and one pipeline is
//! waited for at most [`ONE_MOST`] over all frames: one that never comes is
//! not waited for again every frame. Passes over the screen have two
//! workers of their own and a share of the allowance compute cannot take
//! (idea from tn1 6563de95, after Belobug 18e57db8).
//!
//! shortcut: the budgets are guesses; any other draw the title runs only
//! once and whose pipeline takes longer than half a second is lost.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};
use super::library::Save;

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queued: AtomicBool,
    /// Whether it went into the queue of pipelines a draw waits for, and
    /// whether a worker took it (it may be in both queues).
    now: AtomicBool,
    taken: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    #[cfg(feature = "diagnostics")]
    pub(super) skipped_shader: Option<String>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    pub(crate) key: String,
    used: AtomicBool,
    /// Its place in the early list, if started early.
    place: AtomicUsize,
    /// Microseconds title threads waited for it, over all frames.
    waited_us: AtomicU64,
    /// One of the acceleration-structure builder's ([`begin_builder`]).
    builder: AtomicBool,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(
        shaders: Vec<Arc<Shader>>,
        graphics: Option<Box<GraphicsState>>,
        #[cfg(feature = "diagnostics")] skip_hashes: &[String],
    ) -> Pending {
        let key = key(&shaders, graphics.as_deref());
        #[cfg(feature = "diagnostics")]
        log("created", &key, || {
            let state = graphics.as_ref().map(|g| g.describe()).unwrap_or_default();
            format!("{shaders:?}\t{state}")
        });
        #[cfg(feature = "diagnostics")]
        let skipped_shader = super::skipped_shader(&shaders, skip_hashes).map(str::to_owned);
        let made = OnceLock::new();
        // Publish the completed result before the pipeline can be queued.
        #[cfg(feature = "diagnostics")]
        if skipped_shader.is_some() {
            let _ = made.set(None);
        }
        Pending {
            key,
            shaders,
            graphics,
            queued: AtomicBool::new(false),
            now: AtomicBool::new(false),
            taken: AtomicBool::new(false),
            made,
            #[cfg(feature = "diagnostics")]
            skipped_shader,
            used: AtomicBool::new(false),
            place: AtomicUsize::new(usize::MAX),
            waited_us: AtomicU64::new(0),
            builder: AtomicBool::new(false),
        }
    }

    /// Which share of the allowance its waits take.
    fn class(&self) -> Class {
        if self.builder.load(Ordering::Relaxed) {
            return Class::Builder;
        }
        match (self.screen(), self.urgent()) {
            (true, _) => Class::Screen,
            (false, true) => Class::Urgent,
            (false, false) => Class::Other,
        }
    }

    fn screen(&self) -> bool {
        self.graphics.as_ref().is_some_and(|g| g.screen_pass())
    }

    /// Waited for like a screen pass ([`waits_compute`] for compute).
    fn urgent(&self) -> bool {
        static COMPUTE: OnceLock<bool> = OnceLock::new();
        let compute = || waits_compute(std::env::var_os("VICTORY_COMPUTE_WAIT").as_deref());
        self.screen()
            || *COMPUTE.get_or_init(compute) && self.shaders.iter().any(|s| s.stage == "compute")
    }

    fn make(&self) {
        self.made.get_or_init(|| {
            let (started, cpu) = (Instant::now(), super::thread_cpu());
            let made = HostPipeline::new(&self.shaders, self.graphics.as_deref());
            #[cfg(feature = "diagnostics")]
            log_built(&self.key);
            let (user, kernel) = super::thread_cpu_since(cpu);
            tracing::debug!(
                shaders = ?self.shaders,
                ms = started.elapsed().as_millis() as u64,
                user_ms = user,
                kernel_ms = kernel,
                "pipeline made"
            );
            made
        });
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// What identifies a title pipeline across runs: its shader containers
/// and graphics state.
pub(crate) fn key(shaders: &[Arc<Shader>], graphics: Option<&GraphicsState>) -> String {
    let mut key = blake3::Hasher::new();
    for s in shaders {
        key.update(s.stage.as_bytes());
        key.update(s.hash.as_bytes());
    }
    key.update(graphics.map(|g| g.describe()).unwrap_or_default().as_bytes());
    key.finalize().to_hex()[..32].to_owned()
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Logs the library name of the host pipeline this thread just made or
/// loaded for `key` (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn log_built(key: &str) {
    if let Some((name, loaded)) = super::library::LAST.take() {
        log("built", key, || format!("{name}\t{}", if loaded { "loaded" } else { "made" }));
    }
}

/// `VICTORY_PIPELINE_LOG=<file>` (diagnostics): a line per title pipeline
/// key created (`created`, its shaders, its state as described) and first
/// drawn or dispatched with (`used`), and its host pipeline's library name
/// once made or loaded (`built`, also from the pre-build), to compare what
/// a run creates with what the pre-build makes.
#[cfg(feature = "diagnostics")]
fn log(what: &str, key: &str, rest: impl FnOnce() -> String) {
    use std::collections::HashSet;
    use std::io::Write;
    static LOG: Mutex<Option<(std::fs::File, HashSet<String>)>> = Mutex::new(None);
    static PATH: OnceLock<Option<std::ffi::OsString>> = OnceLock::new();
    let Some(path) = PATH.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_LOG")) else {
        return;
    };
    let mut log = LOG.lock().unwrap_or_else(|e| e.into_inner());
    if log.is_none() {
        *log = std::fs::File::create(path).ok().map(|f| (f, HashSet::new()));
    }
    let Some((file, seen)) = log.as_mut() else { return };
    if seen.insert(format!("{what} {key}")) {
        let _ = writeln!(file, "{what}\t{key}\t{}", rest());
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// All waiting for pipelines allowed per presented frame, over all threads
/// and classes, well under the title's 30 s watchdog: what else a frame
/// waits for (page tables, copies) keeps room.
pub(crate) const ALLOWANCE: Duration = Duration::from_secs(10);

/// Of it, the most for pipelines that are not urgent.
const BUDGET: Duration = Duration::from_millis(500);

/// Of it, what is kept for passes over the screen and the builder: other
/// compute pipelines leave it.
const SCREEN_RESERVE: Duration = Duration::from_secs(3);

/// The most one pipeline is waited for, over all frames: past it, only
/// checked (it may still come, and is drawn with then). Not for the
/// builder's, whose frames run whole or wait (a dropped builder dispatch
/// leaves the title's later work waiting on work that never ran, gpu307).
const ONE_MOST: Duration = ALLOWANCE;

/// Which share of the allowance a wait takes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Class {
    Screen,
    /// The acceleration-structure builder's (`super::builder`).
    Builder,
    /// Other compute pipelines.
    Urgent,
    Other,
}

/// Time some title thread waited: waits on several recording threads at
/// once count once (summed at their ends, two waits begun together each had
/// all of the allowance: 2x with no present).
#[derive(Clone, Copy, Debug)]
struct Waited {
    /// Closed so far, and since when threads wait now, how many.
    closed: Duration,
    since: Option<Instant>,
    waiters: u32,
}

impl Waited {
    const fn new() -> Waited {
        Waited { closed: Duration::ZERO, since: None, waiters: 0 }
    }

    fn now(&self) -> Duration {
        self.closed + self.since.map_or(Duration::ZERO, |t| t.elapsed())
    }

    fn begin(&mut self) {
        if self.waiters == 0 {
            self.since = Some(Instant::now());
        }
        self.waiters += 1;
    }

    fn end(&mut self) {
        self.waiters = self.waiters.saturating_sub(1);
        if self.waiters == 0
            && let Some(since) = self.since.take()
        {
            self.closed += since.elapsed();
        }
    }
}

/// What waits used of a presented frame's allowance: in all, and by
/// pipelines that are not urgent.
#[derive(Clone, Copy, Debug)]
struct Spent {
    frame: u64,
    all: Waited,
    other: Waited,
}

impl Spent {
    const fn new() -> Spent {
        Spent { frame: u64::MAX, all: Waited::new(), other: Waited::new() }
    }

    /// What `class` may still wait in `frame` (a new frame starts anew).
    fn left(&mut self, frame: u64, class: Class) -> Duration {
        if self.frame != frame {
            *self = Spent { frame, ..Spent::new() };
        }
        let used = self.all.now();
        let all = |cap: Duration| cap.saturating_sub(used);
        match class {
            Class::Screen | Class::Builder => all(ALLOWANCE),
            Class::Urgent => all(ALLOWANCE - SCREEN_RESERVE),
            Class::Other => all(ALLOWANCE - SCREEN_RESERVE).min(BUDGET.saturating_sub(self.other.now())),
        }
    }

    /// A wait of `class` begins in `frame`: how long it may last, at most
    /// `most` (from now). [`Spent::done`] after.
    fn take(&mut self, frame: u64, class: Class, most: Duration) -> Duration {
        let took = self.left(frame, class).min(most);
        if !took.is_zero() {
            self.all.begin();
            if class == Class::Other {
                self.other.begin();
            }
        }
        took
    }

    /// A wait [`Spent::take`] gave `took` ended (none if it gave nothing,
    /// or a present came meanwhile).
    fn done(&mut self, frame: u64, class: Class, took: Duration) {
        if self.frame != frame || took.is_zero() {
            return;
        }
        self.all.end();
        if class == Class::Other {
            self.other.end();
        }
    }

    /// `class` waited `waited` in `frame` (counted in that frame only).
    #[cfg(test)]
    fn spend(&mut self, frame: u64, class: Class, waited: Duration) {
        if self.frame == frame {
            self.all.closed += waited;
            if class == Class::Other {
                self.other.closed += waited;
            }
        }
    }
}

static SPENT: Mutex<Spent> = Mutex::new(Spent::new());

fn spent() -> std::sync::MutexGuard<'static, Spent> {
    SPENT.lock().unwrap_or_else(|e| e.into_inner())
}

/// The frame the allowance belongs to: frames presented. The title's own
/// frame count goes on while a submission waits, so a budget kept by it
/// started anew while no frame was shown.
fn frame() -> u64 {
    crate::queue::presented()
}

/// Whether compute pipelines are waited for like screen passes (unless
/// `VICTORY_COMPUTE_WAIT` is 0).
fn waits_compute(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_none_or(|v| v != "0")
}

/// The queue a pipeline goes into, the first served first: passes over the
/// screen a draw waits for (the only queue [`SCREEN_WORKERS`] take), other
/// urgent pipelines, other draws' pipelines, early starts.
fn lane(class: Class, now: bool) -> usize {
    match (now, class) {
        (true, Class::Screen | Class::Builder) => 0,
        (true, Class::Urgent) => 1,
        (true, Class::Other) => 2,
        (false, _) => 3,
    }
}

/// Workers that make passes over the screen alone, besides the others.
const SCREEN_WORKERS: usize = 2;

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    // Listed for an early start in later runs from its first use, made or
    // not: one a run never got made is the one it should start early.
    if !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
        #[cfg(feature = "diagnostics")]
        log("used", &p.key, String::new);
    }
    if p.made.get().is_none() {
        start(p, true);
        wait(p);
    }
    p.made.get()?.as_ref()
}

/// The builder created `p`: made at once, ahead of every other pipeline
/// (gpu315: its 28 pipelines, queued behind 1,900 early starts, were still
/// being made when its first frame came, and its dispatches dropped).
pub(crate) fn begin_builder(p: &Arc<Pending>) {
    p.builder.store(true, Ordering::Relaxed);
    if !p.shaders.is_empty() {
        start(p, true);
        to_front(p);
    }
}

/// Queues each of `ps` (the builder's) at the front of the queue and waits
/// until all are made or refused, within what is left of the frame's
/// allowance: the builder's frames run whole or wait (`super::builder`), so
/// [`ONE_MOST`] does not hold for them. What was waited.
pub(crate) fn wait_all(ps: &[Arc<Pending>]) -> Duration {
    let frame = frame();
    let ps: Vec<&Arc<Pending>> = ps.iter().filter(|p| p.made.get().is_none() && !p.shaders.is_empty()).collect();
    // Queued first whatever is left: made sooner for a later frame.
    for p in &ps {
        start(p, true);
        to_front(p);
    }
    if ps.is_empty() {
        return Duration::ZERO;
    }
    let left = spent().take(frame, Class::Builder, ALLOWANCE);
    if left.is_zero() {
        return Duration::ZERO;
    }
    let started = Instant::now();
    WAITERS.fetch_add(1, Ordering::Relaxed);
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while ps.iter().any(|p| p.made.get().is_none()) {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    drop(held);
    WAITERS.fetch_sub(1, Ordering::Relaxed);
    let waited = started.elapsed().min(left);
    for p in &ps {
        p.waited_us.fetch_add(waited.as_micros() as u64, Ordering::Relaxed);
    }
    spent().done(frame, Class::Builder, left);
    waited
}
/// Whether `p` has shaders (one with none is never made).
pub(crate) fn has_shaders(p: &Pending) -> bool {
    !p.shaders.is_empty()
}

/// Whether `p`'s host pipeline is made (`Some(true)`), refused or not
/// translated (`Some(false)`), or not yet (`None`); never waits.
pub(crate) fn state(p: &Pending) -> Option<bool> {
    p.made.get().map(Option::is_some)
}

/// Queues `p` for the workers: once, and once more ahead of the early
/// starts if a draw waits for it (`now`) while it is still queued.
fn start(p: &Arc<Pending>, now: bool) {
    if let Some(now) = enqueue(p, now) {
        queue(p.clone(), now);
    }
}

/// Which queue `p` goes into now (true: the one draws wait for), if any.
fn enqueue(p: &Pending, now: bool) -> Option<bool> {
    let first = !p.queued.swap(true, Ordering::Relaxed);
    if first {
        PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
    }
    let ahead = now && !p.now.swap(true, Ordering::Relaxed);
    (first || ahead && !p.taken.load(Ordering::Relaxed)).then_some(ahead)
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    super::library::note_created(&p.key);
    if p.made.get().is_none()
        && !p.shaders.is_empty()
        && let Some(place) = super::library::early_place(&p.key)
    {
        p.place.store(place, Ordering::Relaxed);
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, false);
    }
}

/// Waits for `p` while the frame has allowance left for its class, and `p`
/// has some of [`ONE_MOST`] left; at the front of its queue meanwhile.
fn wait(p: &Arc<Pending>) {
    let (frame, class) = (frame(), p.class());
    let own = match class {
        Class::Builder => ALLOWANCE,
        _ => ONE_MOST.saturating_sub(Duration::from_micros(p.waited_us.load(Ordering::Relaxed))),
    };
    let left = spent().take(frame, class, own);
    if left.is_zero() {
        to_front(p);
        return;
    }
    to_front(p);
    let started = Instant::now();
    WAITERS.fetch_add(1, Ordering::Relaxed);
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    drop(held);
    WAITERS.fetch_sub(1, Ordering::Relaxed);
    let waited = started.elapsed().min(left);
    let us = waited.as_micros() as u64;
    p.waited_us.fetch_add(us, Ordering::Relaxed);
    spent().done(frame, class, left);
    #[cfg(feature = "diagnostics")]
    if class == Class::Urgent {
        let dropped = p.made.get().is_none();
        COMPUTE_WAITS[0].fetch_add(1, Ordering::Relaxed);
        COMPUTE_WAITS[1].fetch_add(us, Ordering::Relaxed);
        COMPUTE_WAITS[2].fetch_add(u64::from(dropped), Ordering::Relaxed);
        static LOGGED: AtomicU64 = AtomicU64::new(0);
        if dropped && LOGGED.fetch_add(1, Ordering::Relaxed) < 32 {
            let shaders: Vec<&str> = p.shaders.iter().map(|s| s.hash.as_str()).collect();
            tracing::warn!(key = %p.key, ?shaders, waited_ms = us / 1000, "a dispatch's compute pipeline was not made in time: dropped");
        }
    }
}

/// Compute pipelines waited for, microseconds waited, and of them those not
/// made in time all the same, since the last take.
#[cfg(feature = "diagnostics")]
static COMPUTE_WAITS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// [`COMPUTE_WAITS`], taken (for the frame report).
#[cfg(feature = "diagnostics")]
pub(crate) fn take_compute_waits() -> [u64; 3] {
    COMPUTE_WAITS.each_ref().map(|c| c.swap(0, Ordering::Relaxed))
}

/// Title threads waiting for a pipeline now.
static WAITERS: AtomicU64 = AtomicU64::new(0);

/// Saves the cache if one is due ([`Save`]), unless a title thread waits
/// for a pipeline: the next pipeline made, or the workers' idle check,
/// tries again. With `VICTORY_PIPELINE_SAVE_EVERY_SECS`, by the timer and
/// waits or not, as before. Best effort: a wait may start during a save,
/// but blocks no draw, since pipelines made meanwhile queue to be stored.
fn save(why: Save) {
    if super::library::save_every().is_some() || WAITERS.load(Ordering::Relaxed) == 0 {
        super::library::save(why);
    }
}

/// Pipelines for the workers, by [`lane`].
static QUEUES: (Mutex<[VecDeque<Arc<Pending>>; 4]>, Condvar) =
    (Mutex::new([const { VecDeque::new() }; 4]), Condvar::new());

/// A title thread waits for `p` now: it goes to the front of its draws'
/// queue, ahead of pipelines draws gave up on (a worker takes the first
/// entry it finds; the others are passed by).
fn to_front(p: &Arc<Pending>) {
    if p.taken.load(Ordering::Relaxed) || p.made.get().is_some() {
        return;
    }
    let (lock, work) = &QUEUES;
    let mut queues = lock.lock().unwrap_or_else(|e| e.into_inner());
    front(&mut queues, p);
    drop(queues);
    work.notify_all();
}

/// [`to_front`] on `queues`: once, if not first already.
fn front(queues: &mut [VecDeque<Arc<Pending>>; 4], p: &Arc<Pending>) {
    let lane = &mut queues[lane(p.class(), true)];
    if !lane.front().is_some_and(|q| Arc::ptr_eq(q, p)) {
        lane.push_front(p.clone());
    }
}

/// Hands `p` to the workers (half the logical processors, 2 to 8, and
/// [`SCREEN_WORKERS`]), started on first use.
fn queue(p: Arc<Pending>, now: bool) {
    static WORKERS: OnceLock<usize> = OnceLock::new();
    let workers = *WORKERS.get_or_init(|| {
        let n = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8));
        (0..n + SCREEN_WORKERS)
            .filter(|&i| {
                let screens_only = i >= n;
                let spawned = std::thread::Builder::new()
                    .name(format!("victory-pipelines-{i}"))
                    .spawn(move || work(screens_only));
                if let Err(e) = &spawned {
                    tracing::warn!(error = %e, "no pipeline worker");
                }
                spawned.is_ok()
            })
            .count()
    });
    if workers == 0 {
        // No worker: made here instead.
        if !p.taken.swap(true, Ordering::Relaxed) {
            p.make();
            WAITING.fetch_sub(1, Ordering::Relaxed);
        }
        return;
    }
    let (lock, work) = &QUEUES;
    let lane = lane(p.class(), now);
    push(&mut lock.lock().unwrap_or_else(|e| e.into_inner())[lane], p, lane == 3);
    // All: a worker for screen passes alone may be the one woken.
    work.notify_all();
}

/// Adds `p`; in the early starts' queue (`by_place`) after its equals by
/// place in the early list, so they go in the order of first use. The draws'
/// queues stay first come.
fn push(queue: &mut VecDeque<Arc<Pending>>, p: Arc<Pending>, by_place: bool) {
    let place = |p: &Pending| p.place.load(Ordering::Relaxed);
    let at = match by_place {
        true => queue.partition_point(|q| place(q) <= place(&p)),
        false => queue.len(),
    };
    queue.insert(at, p);
}

/// A worker: makes queued pipelines, those draws wait for first, and saves
/// the cache when idle (first uses of pipelines made before);
/// `screens_only`: passes over the screen alone.
fn work(screens_only: bool) {
    // Below the title's own threads (loading, recording), but for the
    // screen passes' workers.
    // SAFETY: a plain call on our own thread.
    unsafe {
        use windows::Win32::System::Threading::*;
        let priority = if screens_only { THREAD_PRIORITY_NORMAL } else { THREAD_PRIORITY_BELOW_NORMAL };
        let _ = SetThreadPriority(GetCurrentThread(), priority);
    }
    let (lock, work) = &QUEUES;
    let lanes = if screens_only { 1 } else { 4 };
    loop {
        let p = {
            let mut queues = lock.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(p) = queues[..lanes].iter_mut().find_map(VecDeque::pop_front) {
                    break p;
                }
                let (held, waited) = work
                    .wait_timeout(queues, Duration::from_secs(20))
                    .unwrap_or_else(|e| e.into_inner());
                queues = held;
                if waited.timed_out() {
                    drop(queues);
                    save(Save::Drained);
                    queues = lock.lock().unwrap_or_else(|e| e.into_inner());
                }
            }
        };
        if p.taken.swap(true, Ordering::Relaxed) {
            continue; // its other queue entry
        }
        p.make();
        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
            drained();
        } else {
            save(Save::Made);
        }
    }
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved.
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    save(Save::Drained);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An early start is queued once; a draw waiting for it puts it ahead,
    /// once, unless a worker has taken it.
    #[test]
    fn draws_go_ahead_of_early_starts() {
        let p = Pending::new(
            Vec::new(),
            None,
            #[cfg(feature = "diagnostics")]
            &[],
        );
        assert_eq!(enqueue(&p, false), Some(false));
        assert_eq!(enqueue(&p, false), None);
        assert_eq!(enqueue(&p, true), Some(true));
        assert_eq!(enqueue(&p, true), None);
        let q = Pending::new(
            Vec::new(),
            None,
            #[cfg(feature = "diagnostics")]
            &[],
        );
        assert_eq!(enqueue(&q, false), Some(false));
        q.taken.store(true, Ordering::Relaxed);
        assert_eq!(enqueue(&q, true), None);
        let r = Pending::new(
            Vec::new(),
            None,
            #[cfg(feature = "diagnostics")]
            &[],
        );
        assert_eq!(enqueue(&r, true), Some(true));
        assert_eq!(enqueue(&r, false), None);
        // Counted as waiting by `enqueue`, never handed to a worker.
        WAITING.fetch_sub(3, Ordering::Relaxed);
    }

    /// A draw's screen pass is served before other draws' pipelines, those
    /// before early starts.
    #[test]
    fn screen_passes_go_first() {
        assert!(lane(Class::Screen, true) < lane(Class::Urgent, true));
        assert!(lane(Class::Urgent, true) < lane(Class::Other, true));
        assert!(lane(Class::Other, true) < lane(Class::Screen, false));
        assert_eq!(lane(Class::Screen, false), lane(Class::Other, false));
        assert_eq!(lane(Class::Screen, true), 0, "the screen workers' only queue");
        let p = Pending::new(
            Vec::new(),
            None,
            #[cfg(feature = "diagnostics")]
            &[],
        );
        assert!(!p.screen());
    }

    /// One allowance a presented frame for every wait: urgent ones (compute,
    /// the builder) and others leave the screen passes' reserve, others take
    /// half a second at most, and a new presented frame starts anew. The
    /// waits gpu313 added up (20 s compute, 20 s builder) stay below it.
    #[test]
    fn a_frame_s_waits_share_one_allowance() {
        let mut s = Spent::new();
        assert_eq!(s.left(7, Class::Screen), ALLOWANCE);
        assert_eq!(s.left(7, Class::Urgent), ALLOWANCE - SCREEN_RESERVE);
        assert_eq!(s.left(7, Class::Other), BUDGET);
        s.spend(7, Class::Other, BUDGET);
        assert_eq!(s.left(7, Class::Other), Duration::ZERO);
        assert_eq!(s.left(7, Class::Urgent), ALLOWANCE - SCREEN_RESERVE - BUDGET);
        s.spend(7, Class::Urgent, Duration::from_secs(20));
        assert_eq!(s.left(7, Class::Urgent), Duration::ZERO, "the builder after compute");
        assert_eq!(s.left(7, Class::Screen), Duration::ZERO, "nothing past the allowance");
        assert!(ALLOWANCE < Duration::from_secs(30) / 2, "well under the watchdog");
        // A wait that ends after the next present counts in no frame.
        s.spend(6, Class::Screen, Duration::from_secs(5));
        assert_eq!(s.left(8, Class::Screen), ALLOWANCE, "a new presented frame");
        let mut s = Spent::new();
        s.spend(u64::MAX, Class::Urgent, ALLOWANCE - SCREEN_RESERVE);
        assert_eq!(s.left(u64::MAX, Class::Screen), SCREEN_RESERVE, "kept for screen passes");
        assert_eq!(s.left(u64::MAX, Class::Builder), SCREEN_RESERVE, "and for the builder");
    }

    /// Waits on two recording threads at once count their overlap once,
    /// and a wait begun while another runs has only what is left of the
    /// allowance from then (counted at their ends, both had all of it: 2x
    /// with no present).
    #[test]
    fn waits_at_once_count_once() {
        let mut s = Spent::new();
        let first = s.take(3, Class::Screen, ALLOWANCE);
        std::thread::sleep(Duration::from_millis(30));
        let second = s.take(3, Class::Screen, ALLOWANCE);
        assert_eq!(first, ALLOWANCE);
        assert!(second <= ALLOWANCE - Duration::from_millis(30), "the time waited since the first");
        std::thread::sleep(Duration::from_millis(30));
        s.done(3, Class::Screen, first);
        s.done(3, Class::Screen, second);
        let used = ALLOWANCE - s.left(3, Class::Screen);
        assert!(used >= Duration::from_millis(60) && used < Duration::from_millis(500), "once: {used:?}");
        // Other pipelines: what a wait did not use goes back.
        let mut s = Spent::new();
        let took = s.take(3, Class::Other, ALLOWANCE);
        assert_eq!(took, BUDGET);
        assert!(s.take(3, Class::Other, ALLOWANCE) <= BUDGET, "what is left of it");
        s.done(3, Class::Other, took);
        assert!(s.left(3, Class::Other) > BUDGET - Duration::from_millis(50), "what it did not use goes back");
    }

    /// The builder's pipelines go to the front of the screen passes' queue
    /// when created, and its waits ignore [`ONE_MOST`].
    #[test]
    fn builder_pipelines_are_made_first() {
        let p = Arc::new(Pending::new(
            Vec::new(),
            None,
            #[cfg(feature = "diagnostics")]
            &[],
        ));
        begin_builder(&p);
        assert_eq!(p.class(), Class::Builder);
        assert_eq!(lane(p.class(), true), 0, "served by the screen workers too");
    }

    /// A pipeline waited for goes to the front of its queue, once, ahead of
    /// those queued before that no one waits for now.
    #[test]
    fn a_pipeline_waited_for_goes_first() {
        let new = || {
            Arc::new(Pending::new(
                Vec::new(),
                None,
                #[cfg(feature = "diagnostics")]
                &[],
            ))
        };
        let mut queues = [const { VecDeque::new() }; 4];
        let (given_up, waited) = (new(), new());
        let lane = lane(given_up.class(), true);
        push(&mut queues[lane], given_up.clone(), false);
        push(&mut queues[lane], waited.clone(), false);
        front(&mut queues, &waited);
        front(&mut queues, &waited);
        let order: Vec<bool> = queues[lane].iter().map(|p| Arc::ptr_eq(p, &waited)).collect();
        assert_eq!(order, [true, false, true], "first, once; its later entry is passed by");
    }

    /// A pipeline that never comes is waited for [`ONE_MOST`] in all, not
    /// again every frame; one refused is not waited for.
    #[test]
    fn a_pipeline_that_never_comes_is_not_waited_for_again() {
        let shader = Arc::new(Shader {
            stage: "compute",
            hash: "never made".into(),
            container: Vec::new(),
            translated: OnceLock::new(),
            partial: OnceLock::new(),
        });
        let p = Arc::new(Pending::new(
            vec![shader],
            None,
            #[cfg(feature = "diagnostics")]
            &[],
        ));
        // Never handed to a worker: taken, so no worker makes it here.
        p.taken.store(true, Ordering::Relaxed);
        p.waited_us.store(ONE_MOST.as_micros() as u64, Ordering::Relaxed);
        let started = Instant::now();
        wait(&p);
        assert!(started.elapsed() < Duration::from_millis(100), "{:?}", started.elapsed());
        let refused = Arc::new(Pending::new(
            Vec::new(),
            None,
            #[cfg(feature = "diagnostics")]
            &[],
        ));
        assert_eq!(wait_all(&[refused]), Duration::ZERO, "nothing to make");
        WAITING.fetch_sub(u64::from(p.queued.load(Ordering::Relaxed)), Ordering::Relaxed);
    }

    /// Compute pipelines wait like screen passes unless told not to.
    #[test]
    fn compute_waits_unless_told_not_to() {
        assert!(waits_compute(None));
        assert!(!waits_compute(Some("0".as_ref())));
        assert!(waits_compute(Some("1".as_ref())));
    }

    /// Early starts are taken in the early list's order, not creation's.
    #[test]
    fn early_starts_go_by_first_use() {
        let mut queue = VecDeque::new();
        for place in [2, 0, 1] {
            let p = Pending::new(
                Vec::new(),
                None,
                #[cfg(feature = "diagnostics")]
                &[],
            );
            p.place.store(place, Ordering::Relaxed);
            push(&mut queue, Arc::new(p), true);
        }
        let places: Vec<_> = queue.iter().map(|p| p.place.load(Ordering::Relaxed)).collect();
        assert_eq!(places, [0, 1, 2]);
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(
            Vec::new(),
            None,
            #[cfg(feature = "diagnostics")]
            &[],
        ));
        assert!(ready(&p).is_none());
        assert!(p.queued.load(Ordering::Relaxed));
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn skipped_pipeline_stays_off_the_queue() {
        let hashes = crate::pipeline::parse_skip_hashes(" 0123456789ABCDEF ");
        let shader = Arc::new(Shader {
            stage: "compute",
            hash: "0123456789abcdef".into(),
            container: Vec::new(),
            translated: OnceLock::new(),
            partial: OnceLock::new(),
        });
        let cold = Arc::new(Pending::new(vec![shader.clone()], None, &hashes));
        let warm = Arc::new(Pending::new(vec![shader.clone()], None, &hashes));
        assert!(ready(&cold).is_none());
        crate::pipeline::library::note_used(&warm.key);
        early(&warm);
        assert_eq!(
            (
                cold.queued.load(Ordering::Relaxed),
                warm.queued.load(Ordering::Relaxed)
            ),
            (false, false)
        );
        assert!(matches!(cold.made.get(), Some(None)));
        assert!(matches!(warm.made.get(), Some(None)));
        assert!(shader.translated.get().is_none());
    }
}
