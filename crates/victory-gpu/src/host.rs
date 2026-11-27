//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap;
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WAIT_TIMEOUT, WPARAM};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{
    D3D_FEATURE_LEVEL_12_0, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob,
};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCSTR, PCWSTR, s, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    pub(crate) depth_bounds_supported: bool,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// Timestamps around each submission (diagnostics, timing off).
    #[cfg(feature = "diagnostics")]
    clock: OnceLock<Option<crate::gpumeasure::Clock>>,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// A ring of CPU-only views, for the CPU handles of unordered access
    /// clears, used as `rtvs`.
    cpu_views: ID3D12DescriptorHeap,
    cpu_view_size: u32,
    next_cpu_view: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Presentation>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`,
// or by the retire thread while no submission may start (`lock_idle`).
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Copy regions (destination, offset, source, offset, bytes).
type Regions = Vec<(ID3D12Resource, u64, ID3D12Resource, u64, u64)>;

thread_local! {
    /// While set, [`Host::copy_now`] adds its regions here instead of
    /// copying ([`Host::pull_riding`]).
    static RIDING: std::cell::RefCell<Option<Regions>> = const { std::cell::RefCell::new(None) };
}

/// Whether this thread is recording a riding pull ([`Host::pull_riding`]).
pub(crate) fn riding_active() -> bool {
    RIDING.with(|r| r.borrow().is_some())
}

/// The list a submission's pull rides in ([`Host::pull_riding`]).
struct PullList {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    /// Executed and not yet waited for.
    busy: bool,
    /// The submission fence value signaled after its last run.
    value: u64,
}
// SAFETY: D3D12 objects are free-threaded; used under its mutex only.
unsafe impl Send for PullList {}

/// A pull recorded to run after a submission's lists: its list (busy
/// until the wait), the buffers it copies between, and the pins of the
/// stale pages it brings back (held until the wait, as `copy_now` holds
/// them: a reader meanwhile would take the mirror for current).
struct Riding {
    pull: &'static Mutex<PullList>,
    list: ID3D12CommandList,
    _regions: Regions,
    stale: crate::shadow::Held,
}

impl Riding {
    fn pull(&self) -> std::sync::MutexGuard<'static, PullList> {
        self.pull.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Whether a submission's pull runs right after its lists, under their
/// fence wait (`VICTORY_PULL_SAME_EXECUTE=0`: in a submission and wait of
/// its own after them).
fn pull_same_execute() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::switch::on_by_default("VICTORY_PULL_SAME_EXECUTE");
        tracing::info!(on, "pulls ride their submission (VICTORY_PULL_SAME_EXECUTE)");
        on
    })
}

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// A submission sent to the host GPU, its last group not yet waited for
/// and nothing settled ([`Host::submit`], [`Host::retire`]). It owns what
/// it needs, its lists as `L` (references, or `Arc`s to retire elsewhere).
#[must_use = "a sent submission is settled only by Host::retire"]
pub(crate) struct Ticket<'h, L: std::ops::Deref<Target = HostList>>(Option<Sent<'h, L>>);

impl<L: std::ops::Deref<Target = HostList>> Drop for Ticket<'_, L> {
    /// Not retired (a bug, or an unwind): what it locks and predicts goes,
    /// its writes to title memory are lost.
    fn drop(&mut self) {
        let Some(mut t) = self.0.take() else { return };
        tracing::error!(value = t.value, "a host submission was dropped without being retired");
        if let Some(Some(r)) = &mut t.riding {
            r.stale.restale();
            r.stale.release();
        }
        crate::queue::labels_retired(t.value);
        if !std::thread::panicking() {
            debug_assert!(false, "host submission {} dropped without being retired", t.value);
        }
    }
}

/// A ticket's lists in the order they ran: the upload list first.
fn lists_of<'s, L: std::ops::Deref<Target = HostList>>(
    uploads: &'s Option<crate::bindless::Uploads>,
    runs: &'s [L],
) -> Vec<&'s HostList> {
    uploads.iter().map(|u| u.list()).chain(runs.iter().map(|l| &**l)).collect()
}

/// What a [`Ticket`] holds.
pub(crate) struct Sent<'h, L: std::ops::Deref<Target = HostList>> {
    _host: std::marker::PhantomData<&'h Host>,
    /// The fence value of its last group.
    value: u64,
    started: std::time::Instant,
    #[cfg(feature = "diagnostics")]
    flight: Option<crate::copystat::Flight>,
    #[cfg(feature = "diagnostics")]
    sync_heaps_ns: u64,
    #[cfg(feature = "diagnostics")]
    traced: bool,
    #[cfg(feature = "diagnostics")]
    clock: Option<&'h crate::gpumeasure::Clock>,
    /// Flight study: when the first lists went to the card.
    #[cfg(feature = "diagnostics")]
    sent: Option<std::time::Instant>,
    heaps: Vec<std::sync::Arc<crate::bindless::Viewed>>,
    /// The texture uploads run first, kept until they ran.
    uploads: Option<crate::bindless::Uploads>,
    runs: Vec<L>,
    /// Kept until they have run.
    lists: Vec<Option<ID3D12CommandList>>,
    groups: usize,
    snapshots: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    batch: Batch,
    prepared: std::time::Instant,
    riding: Option<Option<Riding>>,
    /// The pull ran and was waited for with the last group.
    pulled: bool,
    removed_in: Option<usize>,
    /// A wait that failed: query results may not be resolved yet.
    unwaited: bool,
    /// The last group sent: its index, when, and its fence wait's setup.
    last: Option<(usize, std::time::Instant, windows::core::Result<()>)>,
    /// Its shaders' write marks, taken on the GPU behind its lists
    /// (`crate::memsys::marks`); none: read from the bitmap at the retire.
    marks: Option<crate::memsys::marks::Snap>,
}

thread_local! {
    static SUBMITTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static RETIRE_THREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the title's submissions retire on a thread of their own
/// (`VICTORY_PIPELINE=1`, off by default, [`Host::execute_later`]).
pub(crate) fn pipelining() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::switch::on("VICTORY_PIPELINE");
        tracing::info!(on, "title submissions retire on their own thread (VICTORY_PIPELINE)");
        on
    })
}

/// Work to run after the retire in flight: `Some` while one is.
type Later = Option<Vec<Box<dyn FnOnce() + Send>>>;
static FLIGHT: Mutex<Later> = Mutex::new(None);
/// Signalled when no retire is in flight any more.
static IDLE: std::sync::Condvar = std::sync::Condvar::new();

fn flight() -> std::sync::MutexGuard<'static, Later> {
    FLIGHT.lock().unwrap_or_else(|e| e.into_inner())
}

/// Runs `then` once every host submission sent so far is retired: at once
/// when none is in flight, else on the retire thread after it.
pub(crate) fn after_retired(then: Box<dyn FnOnce() + Send>) {
    let mut f = flight();
    match f.as_mut() {
        Some(later) => later.push(then),
        None => {
            drop(f);
            then();
        }
    }
}

/// Runs `then` on the retire thread after the retire in flight, before any
/// other submission prepares (no submission runs then); gives it back if
/// none is in flight.
pub(crate) fn after_retire_in_flight(then: Box<dyn FnOnce() + Send>) -> Result<(), Box<dyn FnOnce() + Send>> {
    match flight().as_mut() {
        Some(later) => {
            later.push(then);
            Ok(())
        }
        None => Err(then),
    }
}

/// Whether this is the retire thread: work it waits for runs after it.
pub(crate) fn on_retire_thread() -> bool {
    RETIRE_THREAD.get()
}

/// The retire thread's queue.
fn retirer() -> &'static std::sync::mpsc::Sender<Box<dyn FnOnce() + Send>> {
    static TX: OnceLock<std::sync::mpsc::Sender<Box<dyn FnOnce() + Send>>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Box<dyn FnOnce() + Send>>();
        let _ = std::thread::Builder::new().name("victory retire".into()).spawn(move || {
            RETIRE_THREAD.set(true);
            for job in rx {
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).is_err() {
                    // Not idle again otherwise: every submission would wait.
                    *flight() = None;
                    IDLE.notify_all();
                    fail(format_args!("a host submission's retire panicked"));
                }
            }
        });
        tx
    })
}

/// A ticket on its way to the retire thread.
struct Retiring(Ticket<'static, std::sync::Arc<HostList>>);
// SAFETY: D3D12 objects are free-threaded, and a ticket holds no guard
// bound to its thread (stale pages are pins, the pull list is marked
// busy, the upload list is owned); only the retire thread uses it after.
unsafe impl Send for Retiring {}

impl Retiring {
    /// Taken whole, so that a closure moves the wrapper, not its field.
    fn into_ticket(self) -> Ticket<'static, std::sync::Arc<HostList>> {
        self.0
    }
}

/// This thread holds the submission lock in [`Host::execute`].
struct Submitting(bool);

impl Submitting {
    fn enter() -> Self {
        Submitting(SUBMITTING.replace(true))
    }
}

impl Drop for Submitting {
    fn drop(&mut self) {
        SUBMITTING.set(self.0);
    }
}

/// Whether this thread is inside a host submission: a blocking fence wait
/// here would never end.
pub(crate) fn submitting_here() -> bool {
    SUBMITTING.get()
}

/// How a wait for a submission ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Waited {
    /// The host signalled the submission's fence.
    Done,
    /// The device was removed while the submission ran.
    Removed,
    /// Neither after [`STALL_SECS`]: the GPU has faulted or hung and the
    /// host has not said so.
    Stalled,
}

/// How long one wait for a submission lasts before the device is asked
/// whether it is still there.
const WAIT_SLICE_MS: u32 = 250;

/// A submission still running after this many seconds is said so, and again
/// each time that time has doubled.
const LONG_WAIT_SECS: u64 = 10;

/// A submission still running after this many seconds is given up. A host
/// whose GPU faults may neither finish the work nor report the device
/// removed: vkd3d-proton waits on the driver without an end unless
/// `VKD3D_CONFIG=fault` or `breadcrumbs` is set, and says itself that such
/// a wait of over 5 s is a lost device. Windows resets a device whose
/// submission runs for about 2 s. The title ends its own run when its main
/// thread has waited 30 s for the render thread, which a submission that
/// never finishes brings about; this is below that, so the run ends with
/// our report of the cause and not with the title's.
const STALL_SECS: u64 = 25;

/// Waits for a submission: `signalled` waits one slice ([`WAIT_SLICE_MS`])
/// for its fence and says whether it came. Between slices `removed` asks
/// the device whether it was removed, because a host that loses its device
/// need not signal the fences of the work it dropped (vkd3d-proton after a
/// GPU fault does not), and a wait without an end then holds the title's
/// submit thread until the title ends the run, with no report of ours.
/// `long` is told the seconds waited so far, at [`LONG_WAIT_SECS`] and each
/// doubling of it; at [`STALL_SECS`] the wait is given up.
fn wait_for_submission(
    mut signalled: impl FnMut() -> bool,
    mut removed: impl FnMut() -> bool,
    mut long: impl FnMut(u64),
) -> Waited {
    let mut slices = 0u64;
    let mut report_at = LONG_WAIT_SECS;
    loop {
        if signalled() {
            return Waited::Done;
        }
        if removed() {
            return Waited::Removed;
        }
        slices += 1;
        let secs = slices * u64::from(WAIT_SLICE_MS) / 1000;
        if secs >= STALL_SECS {
            return Waited::Stalled;
        }
        if secs >= report_at {
            long(secs);
            report_at *= 2;
        }
    }
}

/// The view heap size the device takes ([`probe_view_heap`]).
static VIEW_HEAP: OnceLock<u32> = OnceLock::new();

/// The slots of every host view heap:
/// [`victory_shader::emit::HEAP_VIEWS`] where the device makes a
/// shader-visible heap that large, else
/// [`victory_shader::emit::PORTABLE_VIEWS`] (NVIDIA).
pub(crate) fn heap_views() -> u32 {
    host();
    VIEW_HEAP.get().copied().unwrap_or(victory_shader::emit::HEAP_VIEWS)
}

/// Chooses the view heap size ([`heap_views`]) and tells the translator.
fn probe_view_heap(device: &ID3D12Device) {
    use victory_shader::emit::{HEAP_VIEWS, PORTABLE_VIEWS};
    let desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        NumDescriptors: HEAP_VIEWS,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
        NodeMask: 0,
    };
    // SAFETY: plain heap creation, dropped at once.
    let large = unsafe { device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&desc) }.is_ok();
    let takes = *VIEW_HEAP.get_or_init(|| if large { HEAP_VIEWS } else { PORTABLE_VIEWS });
    let translated = victory_shader::emit::set_heap_views(takes);
    if translated != takes {
        tracing::warn!(translated, takes, "shaders translated before the view heap size was known");
    } else if takes != HEAP_VIEWS {
        tracing::info!(views = takes, "the host caps view heaps: shader heap indices clamped below it");
    }
    let map = crate::descriptor::host::share_mode() == crate::descriptor::host::Share::On;
    if victory_shader::emit::set_slot_map(map) != map {
        // Shaders and heaps would disagree on what a heap index names.
        tracing::error!(map, "shaders translated before the slot map was set: VICTORY_DESC_MAP ignored");
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DEBUG");
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if crate::switch::on("VICTORY_D3D12_GBV")
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP") {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            // Whether a GPU hang ends the run or freezes the PC (#1054).
            crate::tdr::log();
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        probe_view_heap(&device);
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
        let depth_bounds_supported = device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS2,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
            .is_ok()
            && options.DepthBoundsTestSupported.as_bool();
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        let cpu_views: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let cpu_view_size =
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        tracing::info!("host D3D12 device ready");
        // The console fuses its multiply-adds (`v_fma_f32`, `v_fmac_f32`,
        // `v_fma_mix*`); the lowering's DXIL `FMad` is fused only where the
        // host chooses (the RX 6700 XT does, WARP does not).
        match victory_shader::emit::host::Host::from_device(device.clone()).fuses_fmad()
        {
            Ok(true) => {}
            Ok(false) => tracing::warn!(
                "the host GPU does not fuse multiply-adds: results of the console's fused v_fma and v_fmac can differ in kind here"
            ),
            Err(e) => tracing::debug!(error = %e, "multiply-add fusing not checked"),
        }
        Ok(Host {
            device,
            depth_bounds_supported,
            queue,
            fence,
            submitted: Mutex::new(0),
            event,
            #[cfg(feature = "diagnostics")]
            clock: OnceLock::new(),
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            cpu_views,
            cpu_view_size,
            next_cpu_view: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(Presentation::default()),
        })
    }
}

/// Diagnostics: our process's private commit and working set, bytes.
#[cfg(feature = "diagnostics")]
fn process_memory() -> (u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    (c.PagefileUsage as u64, c.WorkingSetSize as u64)
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

/// Diagnostics, `VICTORY_GPU_TIMING`: GPU time per kind of command and
/// pipeline (ms, commands) and lists' wall time against the sum of their
/// commands' (ms), since the last frame-rate line.
#[cfg(feature = "diagnostics")]
type GpuTimes = (std::collections::HashMap<String, (f64, u64)>, f64, f64);
#[cfg(feature = "diagnostics")]
static GPU_TIMES: Mutex<Option<GpuTimes>> = Mutex::new(None);

/// Adds a timed list's commands (from [`HostList::slowest`]) to [`GPU_TIMES`],
/// by the command's kind and pipeline: its note before the first ` | `,
/// without counts and addresses.
#[cfg(feature = "diagnostics")]
fn gpu_times(wall: Duration, commands: &[(f64, String)]) {
    let mut times = GPU_TIMES.lock().unwrap_or_else(|e| e.into_inner());
    let (by, wall_ms, sum) = times.get_or_insert_with(Default::default);
    *wall_ms += wall.as_secs_f64() * 1e3;
    for (ms, note) in commands {
        // "draw 3x1 [mesh 1f.., pixel 2e..] | ...": the kind and the pipeline.
        let head = note.split(" | ").next().unwrap_or_default();
        let kind = head.split(' ').next().unwrap_or_default();
        let key = match head.find('[') {
            Some(at) => format!("{kind} {}", &head[at..]),
            None => kind.to_owned(),
        };
        let e = by.entry(key).or_default();
        (e.0, e.1) = (e.0 + ms, e.1 + 1);
        *sum += ms;
    }
}

/// Commands past their list's stamp cap, since the last frame-rate line.
#[cfg(feature = "diagnostics")]
static UNTIMED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Logs [`GPU_TIMES`] per frame over `frames` frames: the 25 costliest, then
/// totals by kind.
#[cfg(feature = "diagnostics")]
fn log_gpu_times(frames: f64) {
    let untimed = UNTIMED.swap(0, Ordering::Relaxed);
    if untimed > 0 {
        tracing::warn!(
            "GPU timing: {:.1} commands/frame past the stamp cap, timed together as untimed-tail",
            untimed as f64 / frames
        );
    }
    let Some((by, wall, sum)) = GPU_TIMES.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    tracing::info!(
        "GPU timing: lists {:.1} ms/frame of wall time, their commands {:.1} ms/frame",
        wall / frames,
        sum / frames
    );
    let mut by: Vec<_> = by.into_iter().collect();
    by.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    for (key, (ms, n)) in by.iter().take(25) {
        tracing::info!("  GPU {:.2} ms/frame in {:.1} commands/frame: {key}", ms / frames, *n as f64 / frames);
    }
    // The 25 above leave a long tail; totals by kind (the key's first word) cover all of it.
    let mut kinds: Vec<(&str, f64, u64)> = Vec::new();
    for (key, (ms, n)) in &by {
        let kind = key.split([' ', '[']).next().unwrap_or("");
        match kinds.iter_mut().find(|k| k.0 == kind) {
            Some(k) => (k.1, k.2) = (k.1 + ms, k.2 + n),
            None => kinds.push((kind, *ms, *n)),
        }
    }
    kinds.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (kind, ms, n) in kinds {
        tracing::info!("  GPU kind {:.2} ms/frame in {:.1} commands/frame: {kind}", ms / frames, n as f64 / frames);
    }
}

/// Diagnostics: the host device's video memory use and budget, in MB, local
/// (on the card) and non-local (system memory the GPU reads), as the
/// system's memory manager sees them: use over budget makes it page
/// allocations at submissions.
pub(crate) fn video_memory() -> Option<[(f64, f64); 2]> {
    static ADAPTER: OnceLock<Option<IDXGIAdapter3>> = OnceLock::new();
    let adapter = ADAPTER
        .get_or_init(|| {
            let device = &host()?.device;
            // SAFETY: plain device and DXGI queries.
            unsafe {
                let luid = device.GetAdapterLuid();
                let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                factory.EnumAdapterByLuid(luid).ok()
            }
        })
        .as_ref()?;
    let info = |group| {
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: a valid out pointer.
        unsafe { adapter.QueryVideoMemoryInfo(0, group, &mut info) }.ok()?;
        Some((info.CurrentUsage as f64 / 1e6, info.Budget as f64 / 1e6))
    };
    Some([info(DXGI_MEMORY_SEGMENT_GROUP_LOCAL)?, info(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)?])
}

#[cfg(test)]
pub(crate) mod tests {
    /// A submission is noted as sent when it goes to the GPU and as done
    /// once it was waited for (`crate::memsys::sent`, `done`): pages of
    /// destroyed pooled buffers wait for that.
    #[test]
    fn a_submission_is_noted_as_sent_and_then_as_done() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        list.close();
        let before = crate::memsys::done();
        host.execute(&[&list]);
        // With submissions held off, the last one sent has run.
        let submitted = no_submissions().unwrap();
        assert!(*submitted >= 1);
        assert_eq!(crate::memsys::sent(), *submitted, "the last submission sent");
        assert_eq!(crate::memsys::done(), *submitted, "and waited for");
        assert!(crate::memsys::done() > before, "ours among them");
    }

    /// A list of our own run at once is waited for: what it copied is
    /// there when `run_now` returns.
    #[test]
    fn a_list_run_now_has_run_when_it_returns() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (src, s) = host.shared_buffer(4).unwrap();
        let (dst, d) = host.shared_buffer(4).unwrap();
        // SAFETY: four mapped bytes each.
        unsafe { s.write_bytes(0xAB, 4) };
        // SAFETY: an open list of ours and two live buffers.
        let ran = host.run_now(&mut |list| unsafe { list.CopyBufferRegion(&dst, 0, &src, 0, 4) });
        assert!(ran);
        // SAFETY: as above, the copy waited for.
        assert_eq!(unsafe { std::slice::from_raw_parts(d, 4) }, [0xAB; 4]);
    }

    #[test]
    fn legal_states_are_texture_states_with_writes_alone() {
        assert_eq!(super::legal_for(0x2, 0x10 | 0x80), 0x10);
        assert_eq!(super::legal_for(0, 0x40_4080), 0x80);
        assert_eq!(super::legal_for(0x1, 0x4 | 0x80), 0x4);
        for (flags, state) in [(0x2, 0x10 | 0x80), (0, 0x40_4080), (0x1, 0x4 | 0x80), (0, 0x100_0080)] {
            let legal = super::legal_for(flags, state);
            assert_eq!(super::legal_for(flags, legal), legal);
        }
    }

    /// While a pull rides a submission, a host copy is recorded for its
    /// list and not run; after, copies run at once again.
    #[test]
    fn host_copies_are_recorded_while_a_pull_rides() {
        let Some(host) = super::host() else {
            return;
        };
        let (src, s) = host.shared_buffer(4).unwrap();
        let (dst, d) = host.shared_buffer(4).unwrap();
        // SAFETY: our mapped buffers, 4 bytes each.
        unsafe {
            std::ptr::write_bytes(s, 0xAB, 4);
            std::ptr::write_bytes(d, 0, 4);
        }
        super::RIDING.with(|r| *r.borrow_mut() = Some(Vec::new()));
        host.copy_now(&[(&dst, 0, &src, 0, 4)]);
        let recorded = super::RIDING.with(|r| r.borrow_mut().take()).unwrap();
        assert_eq!(recorded.len(), 1);
        // SAFETY: as above.
        assert_eq!(unsafe { std::slice::from_raw_parts(d, 4) }, [0; 4], "not copied yet");
        host.copy_now(&[(&dst, 0, &src, 0, 4)]);
        // SAFETY: as above, the copy waited for.
        assert_eq!(unsafe { std::slice::from_raw_parts(d, 4) }, [0xAB; 4]);
    }

    /// A sent submission holds the submission lock and leaves its
    /// bottom-of-pipe write undone until it is retired.
    #[test]
    fn a_sent_submission_settles_when_retired() {
        let Some(host) = super::host() else {
            return;
        };
        let word = Box::new(std::cell::Cell::new(0u64));
        let list = host.new_list().unwrap();
        list.defer_write(word.as_ptr().cast(), crate::queue::Written::Value(7), 8);
        list.close();
        let mut lock = host.lock_idle();
        let ticket = host.submit(&mut lock, vec![&list]);
        let value = *lock;
        assert_eq!(ticket.0.as_ref().map(|t| t.value), Some(value));
        assert_eq!(word.get(), 0, "written when settled");
        host.retire(ticket);
        assert_eq!(word.get(), 7);
        // SAFETY: a plain query.
        assert!(unsafe { host.fence.GetCompletedValue() } >= value);
    }

    /// A submission retired on the retire thread is settled, and what ran
    /// after it, before the next host submission starts.
    #[test]
    fn a_retire_on_its_thread_settles_before_the_next_submission() {
        let Some(host) = super::host() else {
            return;
        };
        let word = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let list = host.new_list().unwrap();
        list.defer_write(word.as_ptr().cast(), crate::queue::Written::Value(7), 8);
        list.close();
        let (tx, rx) = std::sync::mpsc::channel();
        let after = tx.clone();
        host.execute_later(vec![std::sync::Arc::new(list)], Box::new(move || tx.send("retired").unwrap()));
        super::after_retired(Box::new(move || after.send("after").unwrap()));
        let next = host.new_list().unwrap();
        next.close();
        host.execute(&[&next]);
        assert_eq!(word.load(std::sync::atomic::Ordering::SeqCst), 7);
        assert_eq!(rx.try_iter().collect::<Vec<_>>(), ["retired", "after"]);
    }

    /// A ticket dropped unretired gives its pending labels up (and says so
    /// in debug builds).
    #[test]
    fn a_dropped_ticket_gives_its_labels_up() {
        let Some(host) = super::host() else {
            return;
        };
        let list = host.new_list().unwrap();
        list.close();
        let mut lock = host.lock_idle();
        let ticket = host.submit(&mut lock, vec![&list]);
        let dropped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(ticket)));
        assert_eq!(dropped.is_err(), cfg!(debug_assertions));
        // The GPU is done with the list before it goes.
        // SAFETY: the fence and event are ours.
        unsafe {
            let _ = host.fence.SetEventOnCompletion(*lock, host.event);
            super::WaitForSingleObject(host.event, 5000);
        }
    }

    /// Keeps submissions off while held: for tests of what one takes from
    /// state all share, such as the shaders' marks in the bitmaps
    /// (`crate::shadow::copy_back_marks` takes every mark set), which a
    /// test submitting alongside would otherwise take first. `None` with no
    /// host GPU.
    pub(crate) fn no_submissions() -> Option<std::sync::MutexGuard<'static, u64>> {
        super::host().map(|h| h.lock_idle())
    }

    /// Work between submissions is put off while one runs, also when the
    /// thread asking holds the lock itself: never a deadlock.
    #[test]
    fn work_between_submissions_waits_for_none() {
        let Some(host) = super::host() else {
            return;
        };
        let held = no_submissions();
        assert!(host.between_submissions(|| ()).is_none(), "put off, not waited for");
        drop(held);
        // Other tests may be submitting meanwhile.
        let mut got = None;
        for _ in 0..2000 {
            got = host.between_submissions(|| 7);
            if got.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(got, Some(7));
    }

    thread_local! {
        /// Every transition barrier built on this thread (resource, before,
        /// after), in order.
        pub(crate) static ISSUED: std::cell::RefCell<Vec<(usize, u32, u32)>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// The barriers built on this thread since the last call.
    pub(crate) fn issued() -> Vec<(usize, u32, u32)> {
        ISSUED.with(|l| std::mem::take(&mut *l.borrow_mut()))
    }

    /// Replays transitions in the order the GPU runs them from the states
    /// in `states` (resources not in it are not checked), by the debug
    /// layer's rule: each barrier's before-state is the resource's current
    /// one. The first that is not, else the states after.
    pub(crate) fn replay(issued: &[(usize, u32, u32)], states: &mut std::collections::HashMap<usize, u32>) -> Result<(), String> {
        for (k, &(resource, before, after)) in issued.iter().enumerate() {
            if let Some(state) = states.get_mut(&resource) {
                if *state != before {
                    return Err(format!("barrier {k} on {resource:#x}: before {before:#x}, but the resource is in {state:#x}"));
                }
                *state = after;
            }
        }
        Ok(())
    }

    #[test]
    fn swap_chain_failures_retry_once_per_second_and_success_clears_the_wait() {
        use super::{Duration, Instant, SwapChainRetry};

        let start = Instant::now();
        let mut retry = SwapChainRetry::default();
        let mut attempts = 0;
        // Two seconds at 60 Hz, with a failure on every eligible attempt.
        for frame in 0..120 {
            retry.attempt(start + Duration::from_millis(frame * 1000 / 60), || {
                attempts += 1;
                Err::<(), _>("desktop unavailable")
            });
        }
        assert_eq!(attempts, 2);
        let recovered = start + Duration::from_secs(2);
        assert_eq!(retry.attempt(recovered, || Ok::<_, ()>(42)), Some(Ok(42)));
        assert_eq!(retry.attempt(recovered, || Ok::<_, ()>(43)), Some(Ok(43)));
    }

    /// A submission's wait ends when its fence comes, or when the device is
    /// gone and the fence never will.
    #[test]
    fn a_wait_for_a_submission_ends_when_the_device_is_removed() {
        use super::{LONG_WAIT_SECS, STALL_SECS, WAIT_SLICE_MS, Waited, wait_for_submission};
        let slices_per_sec = 1000 / u64::from(WAIT_SLICE_MS);
        // The fence comes in the third slice: the device is asked twice.
        let (mut waits, mut asked) = (0, 0);
        let done = wait_for_submission(
            || {
                waits += 1;
                waits == 3
            },
            || {
                asked += 1;
                false
            },
            |_| panic!("not a long wait"),
        );
        assert_eq!((done, waits, asked), (Waited::Done, 3, 2));
        // The fence never comes and the device is removed in the fifth slice.
        let mut asked = 0;
        let lost = wait_for_submission(
            || false,
            || {
                asked += 1;
                asked == 5
            },
            |_| panic!("not a long wait"),
        );
        assert_eq!((lost, asked), (Waited::Removed, 5));
        // A fence signalled together with the removal is a finished wait:
        // the caller's check after the submission reports the removal.
        assert_eq!(wait_for_submission(|| true, || true, |_| ()), Waited::Done);
        // A long submission on a live device is waited for, and said so at
        // 10 s and 20 s.
        let (mut waits, mut said) = (0, Vec::new());
        let done = wait_for_submission(
            || {
                waits += 1;
                waits > 22 * slices_per_sec
            },
            || false,
            |secs| said.push(secs),
        );
        assert_eq!(done, Waited::Done);
        assert_eq!(said, [LONG_WAIT_SECS, 2 * LONG_WAIT_SECS]);
        // One that neither finishes nor loses its device is given up, after
        // every slice of the time allowed and no sooner.
        let mut waits = 0;
        let stalled = wait_for_submission(
            || {
                waits += 1;
                false
            },
            || false,
            |_| (),
        );
        assert_eq!(
            (stalled, waits),
            (Waited::Stalled, STALL_SECS * slices_per_sec)
        );
    }

    use super::*;

    /// Shaders that use title memory mark their reads in their own bitmap:
    /// read marks of both bitmaps are taken, each slot once until it is
    /// forgotten, and a forgotten slot's next read is taken again.
    #[test]
    fn read_marks_of_both_bitmaps_are_taken() {
        let Some(host) = host() else { return };
        // No submission takes the marks meanwhile.
        let _one = host.lock_idle();
        let (plain, memory) = (victory_shader::emit::HEAP_VIEWS - 2, victory_shader::emit::HEAP_VIEWS - 1);
        crate::shadow::tests::mark_read(false, plain);
        crate::shadow::tests::mark_read(true, memory);
        let taken = crate::shadow::take_reads();
        assert!(taken.contains(&plain), "a plain shader's read");
        assert!(taken.contains(&memory), "a read by a shader that uses title memory");
        let again = crate::shadow::take_reads();
        assert!(!again.contains(&plain) && !again.contains(&memory), "each once");
        crate::shadow::forget_read(memory);
        assert!(!crate::shadow::take_reads().contains(&memory), "forgotten: its mark cleared");
        crate::shadow::tests::mark_read(true, memory);
        assert!(crate::shadow::take_reads().contains(&memory), "read again after it was forgotten");
    }

    /// A slot left null (a placed texture written into the heap, or one given
    /// up) has its forgotten mark cleared before the next submission runs, so
    /// a read in that very submission makes the texture instead of having its
    /// mark wiped out after it.
    #[test]
    fn forgotten_slot_read_in_the_next_submission_is_made() {
        let Some(host) = host() else { return };
        let (null, mut heap) = (std::ptr::null(), std::ptr::null());
        let desc: [u32; 8] = [0, 800, 1, 0, 0, 0, 0, 0];
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        let hr = crate::descriptor::create_heap(null, desc.as_ptr().cast(), &iid, &mut heap);
        assert_eq!(hr, 0);
        let title_heap = crate::descriptor::heap(heap);
        let (slot, memory) = (768u32, vec![0u8; 1 << 16]);
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        let at = title_heap.start + u64::from(slot) * 32;
        crate::descriptor::create_placed_srv(null, &words, &view, at);
        let heap_host = title_heap.host().expect("a host view heap");
        assert!(!heap_host.kept(slot), "placed texture left null until read");
        let list = host.new_list().unwrap();
        crate::shadow::tests::record_mark_read(&list, false, slot);
        list.close();
        host.execute(&[&list]);
        assert!(heap_host.kept(slot), "made after its first read");
        let texture = crate::resource::texture_at(address, &words).expect("its resource");
        assert!(crate::resource::give_up_now(texture));
        assert!(!heap_host.kept(slot), "given up");
        list.reset();
        crate::shadow::tests::record_mark_read(&list, false, slot);
        list.close();
        host.execute(&[&list]);
        assert!(heap_host.kept(slot), "made again after its first read");
        type Release = extern "system" fn(crate::object::This) -> u32;
        assert_eq!(crate::object::slot::<Release>(heap, 0x10)(heap), 0);
    }

    /// The textures shaders read through the heap since the last eviction
    /// pass are named again at each pass (a slot's read is reported as new
    /// only once), and the marks are cleared for the next.
    #[test]
    fn heap_reads_since_the_last_pass_name_their_textures() {
        let Some(host) = host() else { return };
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        crate::descriptor::create_placed_srv(null, &words, &view, title_heap.start);
        let heap_host = title_heap.host().expect("a host view heap");
        assert!(heap_host.mirror(0, title_heap.start, true).is_some());
        let texture = crate::resource::texture_at(address, &words).expect("its resource");
        let raw = windows::core::Interface::as_raw(&crate::resource::made_copy(texture).unwrap().texture) as usize;
        // No submission takes the marks meanwhile.
        let _one = host.lock_idle();
        for _ in 0..2 {
            crate::shadow::tests::mark_read(false, 0);
            let slots = crate::shadow::reads_since();
            assert!(slots.contains(&0), "the slot read");
            assert!(crate::descriptor::read_textures(&slots).contains(&raw), "its texture named");
            assert!(!crate::shadow::reads_since().contains(&0), "the marks cleared");
        }
    }

    /// The idle rule against the heap reads that thrashed (run evict1): a
    /// texture shaders read through the heap in every frame stays over more
    /// frames and time than the idle limits; one read once is given up. A
    /// list of the title's naming a texture (a table read, a copy) stamps it
    /// as used; a list of ours (uploads) does not.
    #[test]
    fn textures_read_every_frame_stay_and_title_lists_stamp_theirs() {
        let Some(host) = host() else { return };
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let step = 32;
        // Two 4x1 RGBA8 textures, linear, in memory of their own, viewed at
        // slots 5 and 6.
        let memory = vec![0u8; 1 << 17];
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let heap_host = title_heap.host().expect("a host view heap");
        let textures: Vec<crate::object::This> = [5u32, 6]
            .iter()
            .map(|&slot| {
                let address = memory.as_ptr() as u64 + u64::from(slot) * (1 << 13);
                let mut view = [0u32; 16];
                view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
                (view[10], view[11]) = (address as u32, (address >> 32) as u32);
                let at = title_heap.start + u64::from(slot) * step;
                crate::descriptor::create_placed_srv(null, &words, &view, at);
                assert!(heap_host.mirror(slot, at, true).is_some());
                crate::resource::texture_at(address, &words).expect("its resource")
            })
            .collect();
        let owned: Vec<_> = textures.iter().map(|&t| crate::resource::owned(t).expect("may be given up")).collect();
        let (idle_frames, idle_ms) = (crate::resource::IDLE_FRAMES, crate::resource::IDLE_MS);
        {
            // No submission takes the marks meanwhile.
            let _one = host.lock_idle();
            // Passes a third of the idle limits apart, over three times them.
            for k in 0..10 {
                crate::shadow::tests::mark_read(false, 5);
                if k == 0 {
                    crate::shadow::tests::mark_read(false, 6);
                }
                let read = crate::descriptor::read_textures(&crate::shadow::reads_since());
                crate::resource::evict_among(&owned, &read, 1000 + k * idle_frames / 3, 1_000_000 + k * idle_ms / 3, 0);
                assert!(crate::resource::made_copy(textures[0]).is_some(), "read every frame: stays (frame {k})");
            }
        }
        assert!(crate::resource::made_copy(textures[1]).is_none(), "read once: given up");
        drop(owned);
        // A title list naming the texture stamps it, an upload list does not.
        let copy = crate::resource::host_copy(textures[0]).unwrap();
        for (ours, stamped) in [(true, false), (false, true)] {
            copy.unstamp();
            let list = host.new_list().unwrap();
            if ours {
                list.set_ours();
            }
            let _ = list.transition(textures[0], 0x40);
            assert_eq!(copy.last_used().0 != u64::MAX, stamped, "ours: {ours}");
        }
    }

    /// A list twice in one submission gets a prelude for each place: the
    /// second must not record over the first's barriers.
    #[test]
    fn a_list_run_twice_keeps_both_preludes() {
        let Some(host) = host() else { return };
        let address = crate::watch::tests::pool(1) as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 64, 0, 64, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let mut object = std::ptr::null();
        assert_eq!(crate::resource::create_at_address(std::ptr::null(), address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut object), 0);
        let copy = crate::resource::host_copy(object).expect("a host texture");
        let (common, copy_dest, generic_read) = (0, 0x400, 0xac3);
        let real = copy.state.swap(common, Ordering::Relaxed);
        let list = host.new_list().unwrap();
        list.track(&copy, copy_dest);
        list.track(&copy, generic_read);
        list.close();
        let _ = issued();
        let first = list.prelude(host, 0).expect("from the common state");
        let second = list.prelude(host, 1).expect("from the state the first run leaves");
        assert_ne!(first.as_raw(), second.as_raw());
        let texture = copy.texture.as_raw() as usize;
        assert_eq!(issued(), [(texture, common, copy_dest), (texture, generic_read, copy_dest)]);
        // Neither prelude ran.
        copy.state.store(real, Ordering::Relaxed);
    }

    /// Textures the heaps view are brought up to date only if the title used
    /// them lately: one whose pages changed but that no shader read is not
    /// uploaded; read in the last pass, it is.
    #[test]
    fn heap_textures_unread_lately_are_not_brought_up() {
        let Some(host) = host() else { return };
        let address = crate::watch::tests::pool(1) as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 64, 0, 64, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let mut object = std::ptr::null();
        let null = std::ptr::null();
        assert_eq!(crate::resource::create_at_address(null, address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut object), 0);
        let copy = crate::resource::host_copy(object).expect("a host texture");
        let mut desc = [0u32; 10];
        desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        let viewed = std::sync::Arc::new(crate::bindless::Viewed::default());
        viewed.set(0, Some(&crate::descriptor::View::Shader(object as usize, Some(desc))), true);
        // No submission uploads meanwhile.
        let _one = host.lock_idle();
        // SAFETY: the texture's memory, 256-byte rows of 64 texels.
        unsafe { std::ptr::write_bytes(address as *mut u8, 0x40, 64 * 64 * 4) };
        copy.unstamp();
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[]);
        assert_eq!(crate::texture::uploads(object), 0, "not read lately: not brought up");
        copy.stamp();
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[]);
        assert_eq!(crate::texture::uploads(object), 1, "read in the last pass: brought up");
    }

    /// Heap textures brought up in one submission are refreshed side by
    /// side, each into its own staging and its own upload.
    #[test]
    fn heap_textures_brought_up_together_each_get_their_upload() {
        let Some(host) = host() else { return };
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let pool = crate::watch::tests::pool(6);
        let viewed = std::sync::Arc::new(crate::bindless::Viewed::default());
        let objects: Vec<crate::object::This> = (0..6u32)
            .map(|k| {
                // RGBA8 128x128 (one page each), linear, each its own bytes.
                let address = (pool + k as usize * crate::watch::PAGE) as u64;
                // SAFETY: the texture's own page of the pool.
                unsafe { std::ptr::write_bytes(address as *mut u8, 0x10 * k as u8, crate::watch::PAGE) };
                let words: [u32; 14] = [3, 0, 0, 0, 128, 0, 128, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
                let mut object = std::ptr::null();
                assert_eq!(crate::resource::create_at_address(std::ptr::null(), address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut object), 0);
                crate::resource::host_copy(object).expect("a host texture").stamp();
                let mut desc = [0u32; 10];
                desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
                viewed.set(k, Some(&crate::descriptor::View::Shader(object as usize, Some(desc))), true);
                object
            })
            .collect();
        // No submission uploads meanwhile.
        let _one = host.lock_idle();
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[]);
        for &object in &objects {
            assert_eq!(crate::texture::uploads(object), 1, "each brought up once");
        }
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[]);
        for &object in &objects {
            assert_eq!(crate::texture::uploads(object), 1, "unchanged: not again");
        }
    }

    #[test]
    fn list_scratch_grows_to_fit_and_reuses_fenced_chunks() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let first = list.scratch(1).unwrap();
        let second = list.scratch(1).unwrap();
        assert_eq!(second.0 - first.0, 256);
        assert_eq!(second.1 as usize - first.1 as usize, 256);
        assert_eq!(list.scratch.lock().unwrap().chunks[0].capacity, FIRST_SCRATCH_CHUNK);
        let large = list.scratch(96 << 10).unwrap();
        assert_eq!(list.scratch.lock().unwrap().chunks[1].capacity, 128 << 10);
        assert!(list.scratch(u64::MAX).is_none());
        assert!(list.scratch(SCRATCH_CHUNK + 1).is_some());
        list.close();
        host.execute(&[&list]);
        list.reset();
        assert_eq!(list.scratch(96 << 10).unwrap(), large);
        assert_eq!(list.scratch.lock().unwrap().chunks.len(), 2);
        list.close();
        host.execute(&[&list]);
        list.reset();
        assert_eq!(list.scratch(512).unwrap(), first);
        list.close();
    }

    #[test]
    fn present_compositor_builds_both_pipeline_states() {
        let Some(host) = host() else {
            eprintln!("skipping compositor initialization: no D3D12 device");
            return;
        };
        Compositor::new(&host.device, DXGI_FORMAT_R8G8B8A8_UNORM)
            .expect("create compositor root signature and both pipeline states");
        eprintln!("PresentX compositor initialized");
    }

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }

    /// A node of the device-removed data is a list by pointer or by name.
    /// The name is what still matches when the runtime records another
    /// object than the one we hold (the debug layer wraps ours), and a
    /// different list or an unnamed node never matches.
    #[test]
    fn a_breadcrumb_node_is_found_by_pointer_or_by_name() {
        let ours = [0x1000, 0x1008];
        let name = list_name(7);
        let is = |pointer, text| node_is_list(pointer, text, &ours, &name);
        assert_eq!(name, "victory host list 7");
        assert!(is(Some(0x1000), None));
        assert!(is(Some(0x1008), Some("other")));
        assert!(is(Some(0x9999), Some("victory host list 7")));
        assert!(is(None, Some("victory host list 7")));
        assert!(!is(Some(0x9999), Some("victory host list 70")));
        assert!(!is(Some(0x9999), None));
        assert!(!is(None, None));
        // The debug layer appends its own note to the name we gave.
        assert!(is(None, Some("victory host list 7 (debug layer indirect)")));
        assert!(is(None, Some("victory host list 7 (x)")));
        assert!(!is(
            None,
            Some("victory host list 70 (debug layer indirect)")
        ));
        assert!(!is(None, Some("victory host list 7 debug layer")));
        assert!(!is(None, Some("victory host list 7(")));
        assert!(is(None, Some("victory host list 7")));
        assert_eq!(list_name(70), "victory host list 70");
    }

    /// The pointers of one list: the interface pointers it answers with
    /// are all there, and another list's are not.
    #[test]
    fn a_list_knows_every_pointer_it_answers_with() {
        // SAFETY: plain device and list creation on the software adapter.
        unsafe {
            let Ok(adapter) = warp_adapter() else { return };
            let mut device: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_err() {
                return;
            }
            let device = device.unwrap();
            // An open list holds its allocator: one each.
            let make = || -> ID3D12GraphicsCommandList {
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .unwrap()
            };
            let (a, b) = (make(), make());
            let pointers = identities(&a);
            assert!(pointers.contains(&(a.as_raw() as usize)));
            let unknown = a.cast::<windows::core::IUnknown>().unwrap();
            assert!(pointers.contains(&(unknown.as_raw() as usize)));
            let command_list = a.cast::<ID3D12CommandList>().unwrap();
            assert!(pointers.contains(&(command_list.as_raw() as usize)));
            let other = identities(&b);
            assert!(pointers.iter().all(|p| !other.contains(p)));
            // As a node: the same object by any of its pointers; a node
            // naming the other list by pointer and our name still matches.
            let name = list_name(3);
            let is = |pointer, text| node_is_list(pointer, text, &pointers, &name);
            assert!(is(Some(command_list.as_raw() as usize), None));
            assert!(!is(Some(b.as_raw() as usize), None));
            assert!(is(Some(b.as_raw() as usize), Some(&name)));
        }
    }

    /// A node as the runtime gives it, built by hand: `HostList::is_node`
    /// reads the pointer and the copied name, and the name alone finds a
    /// list the runtime records under another object.
    #[test]
    fn is_node_reads_the_name_the_runtime_copied() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let mut wide: Vec<u16> = list_name(list.id).encode_utf16().collect();
        wide.push(0);
        let mut node = D3D12_AUTO_BREADCRUMB_NODE::default();
        assert!(!list.is_node(&node), "an empty node is no list");
        node.pCommandListDebugNameW = windows::core::PCWSTR(wide.as_ptr());
        assert!(list.is_node(&node), "found by its name alone");
        let mut other: Vec<u16> = list_name(list.id + 1).encode_utf16().collect();
        other.push(0);
        node.pCommandListDebugNameW = windows::core::PCWSTR(other.as_ptr());
        assert!(!list.is_node(&node), "another list's name is no match");
        // With the debug layer: our name and the layer's note, in either
        // name field; and a list whose number only starts with ours is not it.
        let note = std::ffi::CString::new(format!("{} (debug layer indirect)", list_name(list.id)))
            .unwrap();
        node.pCommandListDebugNameW = windows::core::PCWSTR::null();
        node.pCommandListDebugNameA = note.as_ptr().cast();
        assert!(
            list.is_node(&node),
            "found by its ANSI name with the layer's note"
        );
        let longer =
            std::ffi::CString::new(format!("{}0 (debug layer indirect)", list_name(list.id)))
                .unwrap();
        node.pCommandListDebugNameA = longer.as_ptr().cast();
        assert!(!list.is_node(&node), "a number that only starts with ours is no match");
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) depth_bounds: Option<ID3D12GraphicsCommandList1>,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Occlusion queries begun in the list and not ended yet: host heap,
    /// type, index, and whether a draw since was not run on the host.
    queries: Mutex<Vec<(ID3D12QueryHeap, D3D12_QUERY_TYPE, u32, bool)>>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `preludes`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// Depth textures with a writable twin the list uses (`crate::twin`),
    /// by host copy: the side its first use needs and the side it leaves
    /// newer.
    twins: Mutex<std::collections::HashMap<usize, crate::twin::Use>>,
    /// A copy of ours replaced the list's compute root signature and
    /// pipeline: draws and dispatches set theirs again.
    rebind: std::sync::atomic::AtomicBool,
    /// A list of ours (uploads), whose uses are no use of the title's
    /// (`crate::resource::evict`).
    ours: std::sync::atomic::AtomicBool,
    /// The draw or dispatch being recorded runs a program using title
    /// memory ([`HostList::writer_uses_memory`]).
    memory_writer: std::sync::atomic::AtomicBool,
    /// A pipeline sync's barrier is the last thing the title recorded
    /// (`crate::list`).
    pub(crate) synced: std::sync::atomic::AtomicBool,
    /// The lists that run those transitions just before this one, one per
    /// place the list has in a submission: each must keep its barriers until
    /// the submission runs.
    preludes: Mutex<Vec<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>>,
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// Which list this is, for the name that tells it apart in device-removed
    /// data (see [`HostList::is_node`]).
    id: u64,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// Clamped indirect arguments (`crate::indirect`), reused by each
    /// execution in turn; replaced by a bigger one when too small.
    arguments: Mutex<Option<ID3D12Resource>>,
    /// The open window of indirect draws clamped together (`indirect`,
    /// `VICTORY_CLAMP_WINDOW`).
    clamp_window: Mutex<Option<crate::indirect::Window>>,
    /// The buffers of the window check (`VICTORY_CLAMP_WINDOW_CHECK`).
    #[cfg(feature = "diagnostics")]
    window_check: OnceLock<Option<crate::indirect::WindowCheck>>,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// In a frame `crate::capture` lists: the textures the list renders to
    /// or writes, and the pipeline that first wrote each (diagnostics).
    #[cfg(feature = "diagnostics")]
    targets: Mutex<Vec<(usize, String)>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
    /// With `VICTORY_PIPELINE_STATS`: the list's statistics query.
    #[cfg(feature = "diagnostics")]
    stats: OnceLock<Option<crate::gpumeasure::ListStats>>,
}

// SAFETY: D3D12 objects are free-threaded; a host list is recorded by the
// title list's one recording thread, then run by one submission at a time
// (the queues hold it between, `crate::hold`).
unsafe impl Send for HostList {}
// SAFETY: as above.
unsafe impl Sync for HostList {}

pub(crate) fn warn_depth_bounds() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| tracing::warn!("the host GPU has no depth bounds test; depth bounds are ignored"));
}

/// Timestamps per list (diagnostics, timing on); commands past them are
/// counted and timed together.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 1 << 16;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DRED"))
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_TIMING"))
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// The id of the next host list (see [`HostList::is_node`]).
static NEXT_LIST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The name a host list gets with device-removed data on.
fn list_name(id: u64) -> String {
    format!("victory host list {id}")
}

/// The pointers one of our list objects answers to: its own and the one each
/// interface a runtime may record it by gives (COM identity is the
/// `IUnknown` one). Only our live object is queried: a node of the
/// device-removed data may name a list that is long released, so its pointer
/// is compared and never dereferenced.
fn identities(list: &impl Interface) -> Vec<usize> {
    let mut found = vec![list.as_raw() as usize];
    let mut add = |pointer: Option<*mut core::ffi::c_void>| {
        if let Some(pointer) = pointer {
            found.push(pointer as usize);
        }
    };
    add(list
        .cast::<windows::core::IUnknown>()
        .ok()
        .map(|i| i.as_raw()));
    add(list.cast::<ID3D12Object>().ok().map(|i| i.as_raw()));
    add(list.cast::<ID3D12DeviceChild>().ok().map(|i| i.as_raw()));
    add(list.cast::<ID3D12CommandList>().ok().map(|i| i.as_raw()));
    found.sort_unstable();
    found.dedup();
    found
}

/// Whether a node of the device-removed data is a list: by pointer, or by the
/// name the runtime copied when the list was submitted. With the debug layer
/// on, the object we hold is its wrapper and the runtime records another
/// one, so no pointer of ours matches, and it appends its own note to our
/// name ("victory host list 158 (debug layer indirect)"). The name matches
/// exactly or followed by " (": never as the start of another number.
fn node_is_list(
    node_pointer: Option<usize>,
    node_name: Option<&str>,
    pointers: &[usize],
    name: &str,
) -> bool {
    node_pointer.is_some_and(|p| pointers.contains(&p))
        || node_name.is_some_and(|n| {
            n == name
                || n.strip_prefix(name)
                    .is_some_and(|rest| rest.starts_with(" ("))
        })
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<ScratchChunk>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 4 << 20;
const FIRST_SCRATCH_CHUNK: u64 = 64 << 10;

struct ScratchChunk {
    _buffer: ID3D12Resource,
    mapped: usize,
    gpu: u64,
    capacity: u64,
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy;
    /// whether they mark the pages they write (no snapshot then), and
    /// whether their program uses title memory (then the range is withheld
    /// when such a list misses, `crate::memory::withholding`).
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool, bool),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>, std::sync::Arc<crate::resource::HostTexture>),
    /// Brings every buffer range a view heap's views reach up to date
    /// (`crate::bindless`).
    Heap(std::sync::Arc<crate::bindless::Viewed>),
    /// Shaders read title memory through a view heap's page table (the
    /// heap's key, `crate::memory`): the table names every copy, each
    /// brought up to date.
    Memory(usize),
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: crate::fx::Set<(usize, usize, usize)>,
    /// Textures whose contents the list brings up to date.
    textures: crate::fx::Set<usize>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// The memory values the list waits for before it starts (a list the
    /// title records with a wait after commands is cut there,
    /// `crate::list`).
    waits: Vec<crate::hold::ValueWait>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
    /// An indirect clamp was recorded and no shader write since.
    #[cfg(feature = "diagnostics")]
    clamp_clean: bool,
    /// A title barrier group ordering indirect arguments was recorded and
    /// no shader write since.
    #[cfg(feature = "diagnostics")]
    args_ordered: bool,
    /// Work done once the list has run, before the bottom-of-pipe writes
    /// (title memory shaders wrote at addresses they computed goes back).
    after: Vec<Box<dyn Fn() + Send>>,
    /// Asked once the list has run, before anything goes back: whether its
    /// shaders read title memory with no copy (`crate::memory`); then what
    /// they wrote through views does not go back either.
    missed: Vec<Box<dyn Fn() -> bool + Send>>,
    /// Buffers shaders wrote, copied at the list's end into read-back
    /// buffers (from, to, bytes; [`HostList::read_back_at_close`]).
    read_backs: Vec<(ID3D12Resource, ID3D12Resource, u64)>,
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// The wave sizes the host runs (D3D12 options 1), which translated
    /// compute shaders that declare the console's wave size must fit;
    /// logged once. Unknown (the query fails): any.
    pub(crate) fn target(&self) -> victory_shader::emit::Target {
        static TARGET: OnceLock<victory_shader::emit::Target> = OnceLock::new();
        *TARGET.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
            // SAFETY: the query writes the structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS1,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let target = if ok && options.WaveOps.as_bool() && options.WaveLaneCountMin > 0 {
                victory_shader::emit::Target {
                    lanes_min: options.WaveLaneCountMin,
                    lanes_max: options.WaveLaneCountMax,
                }
            } else {
                victory_shader::emit::Target::ANY
            };
            tracing::info!(
                lanes_min = target.lanes_min,
                lanes_max = target.lanes_max,
                "host wave sizes"
            );
            if !target.runs(64) {
                tracing::warn!(
                    "the host has no 64-lane waves: console wave64 compute programs that need \
                     their whole wave run as two 32-lane waves where there is a form for \
                     them; the rest do not translate, and their dispatches are dropped"
                );
            }
            target
        })
    }

    /// Whether the host runs 64-bit atomics on typed resources in the
    /// descriptor heap (translated 64-bit image atomics); logged once.
    pub(crate) fn heap_atomics64(&self) -> bool {
        static ATOMICS: OnceLock<bool> = OnceLock::new();
        *ATOMICS.get_or_init(|| {
            let mut typed = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
            let mut heap = D3D12_FEATURE_DATA_D3D12_OPTIONS11::default();
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS9,
                        (&raw mut typed).cast(),
                        size_of_val(&typed) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_D3D12_OPTIONS11,
                            (&raw mut heap).cast(),
                            size_of_val(&heap) as u32,
                        )
                        .is_ok()
            };
            let atomics = ok
                && typed.AtomicInt64OnTypedResourceSupported.as_bool()
                && heap.AtomicInt64OnDescriptorHeapResourceSupported.as_bool();
            tracing::info!(atomics, "host 64-bit atomics on typed heap resources");
            atomics
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let id = NEXT_LIST.fetch_add(1, Ordering::Relaxed);
            name(&list, || list_name(id));
            let made = HostList {
                allocator,
                depth_bounds: self.depth_bounds_supported.then(|| list.cast()).transpose()?,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                queries: Mutex::default(),
                states: Mutex::default(),
                twins: Mutex::default(),
                rebind: false.into(),
                ours: false.into(),
                memory_writer: false.into(),
                synced: false.into(),
                preludes: Mutex::default(),
                pipelines: Mutex::default(),
                id,
                budget: OnceLock::new(),
                arguments: Mutex::new(None),
                clamp_window: Mutex::new(None),
                #[cfg(feature = "diagnostics")]
                window_check: OnceLock::new(),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                targets: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
                #[cfg(feature = "diagnostics")]
                stats: OnceLock::new(),
            };
            #[cfg(feature = "diagnostics")]
            if crate::gpumeasure::stats_on()
                && let Some(stats) = made.stats.get_or_init(|| crate::gpumeasure::ListStats::new(&self.device))
            {
                stats.begin(&made.list);
            }
            Ok(made)
        }
    }

    /// Runs `f` with no submission running, and none starting meanwhile;
    /// None, without running it, if a submission is under way (on this
    /// thread too: the lock is not reentrant, so it is never waited for),
    /// for the caller to try again later.
    pub(crate) fn between_submissions<R>(&self, f: impl FnOnce() -> R) -> Option<R> {
        let _one = match self.submitted.try_lock() {
            Ok(one) => one,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return None,
        };
        // A retire in flight: its lists may still run on the GPU.
        if flight().is_some() {
            return None;
        }
        Some(f())
    }

    /// Runs closed lists and waits for them.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = self.lock_idle();
        let _here = Submitting::enter();
        let ticket = self.submit(&mut submitted, runs.to_vec());
        #[cfg(feature = "diagnostics")]
        let traced = ticket.0.as_ref().is_some_and(|t| t.traced);
        // The upload list and snapshots go in it, before the lock.
        self.retire(ticket);
        // Idle, the lock held: runs the title's changes wait for are dropped.
        crate::memory::serve_drops();
        // A listed frame: what this submission's lists drew, as it is now
        // (the dumps submit lists of their own).
        #[cfg(feature = "diagnostics")]
        if traced {
            let sub = *submitted;
            drop(submitted);
            drop(_here);
            let mut targets: Vec<(usize, String)> = Vec::new();
            for r in runs {
                for t in r.targets.lock().unwrap_or_else(|e| e.into_inner()).iter() {
                    if !targets.iter().any(|x| x.0 == t.0) {
                        targets.push(t.clone());
                    }
                }
            }
            crate::capture::dump_submission(sub, &targets);
        }
    }

    /// The submission lock, once no retire is in flight ([`Host::execute_later`]).
    fn lock_idle(&self) -> std::sync::MutexGuard<'_, u64> {
        // It would wait for itself.
        debug_assert!(!RETIRE_THREAD.get(), "a host submission from the retire thread");
        #[cfg(feature = "diagnostics")]
        let locking = crate::copystat::TimedNs(std::time::Instant::now(), crate::copystat::C::PrepLockNs);
        let submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        drop(IDLE.wait_while(flight(), |f| f.is_some()).unwrap_or_else(|e| e.into_inner()));
        #[cfg(feature = "diagnostics")]
        {
            drop(locking);
            crate::copystat::execute_start();
        }
        submitted
    }

    /// Sends title lists to the host GPU and retires them on the retire
    /// thread ([`pipelining`]), then runs `then`; returns once they are on
    /// the GPU. The next host submission waits for the retire: only the
    /// title's own work overlaps the GPU and the settle.
    pub(crate) fn execute_later(&'static self, runs: Vec<std::sync::Arc<HostList>>, then: Box<dyn FnOnce() + Send>) {
        // A listed frame dumps its lists after the settle: at once.
        #[cfg(feature = "diagnostics")]
        if crate::capture::listed(crate::queue::presented() + 1) {
            let lists: Vec<&HostList> = runs.iter().map(|l| &**l).collect();
            self.execute(&lists);
            drop(lists);
            after_retired(then);
            return;
        }
        let mut submitted = self.lock_idle();
        let here = Submitting::enter();
        let ticket = self.submit(&mut submitted, runs);
        *flight() = Some(Vec::new());
        drop(here);
        drop(submitted);
        let retiring = Retiring(ticket);
        let job: Box<dyn FnOnce() + Send> = Box::new(move || {
            let here = Submitting::enter();
            self.retire(retiring.into_ticket());
            drop(here);
            // Idle, no submission preparing: as at `execute`'s end.
            crate::memory::serve_drops();
            then();
            // Work queued behind it, then idle.
            loop {
                let later = {
                    let mut f = flight();
                    match f.as_mut().map(std::mem::take) {
                        Some(later) if !later.is_empty() => later,
                        _ => {
                            *f = None;
                            IDLE.notify_all();
                            break;
                        }
                    }
                };
                for work in later {
                    work();
                }
            }
            // Held work may wait for what it signalled or wrote.
            crate::queue::wake_held();
        });
        if let Err(e) = retirer().send(job) {
            // No retire thread: here.
            (e.0)();
        }
    }

    /// Prepares closed lists and sends them to the host GPU; the last group
    /// is waited for and settled by [`Host::retire`]. `submitted` is the
    /// submission lock's value, held by the caller from here to the retire.
    pub(crate) fn submit<'h, L: std::ops::Deref<Target = HostList>>(&'h self, submitted: &mut u64, runs: Vec<L>) -> Ticket<'h, L> {
        let given = runs;
        let runs: Vec<&HostList> = given.iter().map(|l| &**l).collect();
        // What follows clears what lists last ran with: none may be running.
        debug_assert!(flight().is_none(), "a submission prepared while a retire is in flight");
        // Before anything is prepared: page tables let go of title memory in
        // place whose views went, if a frame's end could not, and runs the
        // title's changes wait for.
        crate::memory::drop_dead_runs_put_off();
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        // Flight study: whether this is a title submission, and what came
        // between it and the one before.
        #[cfg(feature = "diagnostics")]
        let flight = crate::copystat::flight_begin();
        // Host textures the title has not used for a while (or past the
        // card's budget) are given up first: the last submission has run.
        {
            #[cfg(feature = "diagnostics")]
            let _timed = crate::copystat::TimedNs(std::time::Instant::now(), crate::copystat::C::PrepEvictNs);
            crate::resource::evict_unused();
        }
        // Descriptors the title copied into its heaps itself, before the
        // heaps' views are brought up to date below.
        #[cfg(feature = "diagnostics")]
        let syncing = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SyncHeapsUs);
        crate::descriptor::sync_heaps();
        #[cfg(feature = "diagnostics")]
        let sync_heaps_ns = syncing.0.elapsed().as_nanos() as u64;
        #[cfg(feature = "diagnostics")]
        drop(syncing);
        // Textures heap-indexing shaders may read, uploaded first by a list
        // of ours where their title memory changed.
        let mut heaps: Vec<std::sync::Arc<crate::bindless::Viewed>> = runs
            .iter()
            .flat_map(|l| {
                l.deferred()
                    .copies
                    .iter()
                    .filter_map(|s| match s {
                        Step::Heap(v) => Some(v.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        // Each once: most lists use the same few heaps, and a walk covers
        // every texture a heap views.
        heaps.sort_unstable_by_key(std::sync::Arc::as_ptr);
        heaps.dedup_by(|a, b| std::sync::Arc::ptr_eq(a, b));
        // And textures read through tables, uploaded by the same list.
        let textures: Vec<_> = runs
            .iter()
            .flat_map(|l| {
                l.deferred()
                    .copies
                    .iter()
                    .filter_map(|s| match s {
                        Step::Texture(staging, copy) => Some((staging.clone(), copy.clone())),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        #[cfg(feature = "diagnostics")]
        let uploading = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::UploadsUs);
        let uploads = (!heaps.is_empty() || !textures.is_empty())
            .then(|| crate::bindless::uploads(&heaps, &textures))
            .flatten();
        #[cfg(feature = "diagnostics")]
        drop(uploading);
        let all: Vec<&HostList> = uploads
            .as_ref()
            .map(|u| u.list())
            .into_iter()
            .chain(runs.iter().copied())
            .collect();
        let runs = &all[..];
        // A frame `crate::capture` lists: every command of every list, in
        // the order the GPU runs them (with `VICTORY_D3D12_DRED`, which
        // notes them).
        #[cfg(feature = "diagnostics")]
        let traced = crate::capture::listed(crate::queue::presented() + 1);
        #[cfg(feature = "diagnostics")]
        if traced {
            for (i, r) in runs.iter().enumerate() {
                let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                for (k, (note, _)) in notes.iter().enumerate() {
                    tracing::info!(
                        target: "victory_gpu::trace",
                        "s{} l{i}/{} #{k}: {note}",
                        *submitted + 1,
                        runs.len()
                    );
                }
            }
        }
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        // Snapshots of copies in video memory read their mirrors: brought up
        // to date at once, and kept so until the copy-backs.
        let snapshots: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = runs
            .iter()
            .flat_map(|l| {
                l.deferred()
                    .copies
                    .iter()
                    .filter_map(|s| match s {
                        Step::GpuWrite(shadow, address, len, false, _) => {
                            Some((shadow.clone(), *address, *len))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        {
            #[cfg(feature = "diagnostics")]
            let _timed = crate::copystat::TimedNs(std::time::Instant::now(), crate::copystat::C::PrepSnapshotsNs);
            crate::shadow::snapshots_ahead(&snapshots);
        }
        let mut batch = Batch::reused();
        #[cfg(feature = "diagnostics")]
        {
            let last = crate::copystat::flight_writes(&flight);
            (batch.flight_prev, batch.flight_prev_copies, batch.flight_prev_used) = (last.writes, last.copies, last.used);
            for (staging, _) in &textures {
                let (at, len) = staging.title_range();
                batch.touch(at, len, crate::copystat::C::RangeStaging);
            }
        }
        #[cfg(feature = "diagnostics")]
        let listing = crate::copystat::TimedNs(std::time::Instant::now(), crate::copystat::C::PrepListsNs);
        for (i, list) in runs.iter().enumerate() {
            list.run_deferred(&mut batch);
            firsts.push(lists.len());
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::PreludeUs);
            let nth = runs[..i].iter().filter(|r| std::ptr::eq(**r, *list)).count();
            if let Some(prelude) = list.prelude(self, nth) {
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
        }
        #[cfg(feature = "diagnostics")]
        drop(listing);
        {
            #[cfg(feature = "diagnostics")]
            let _timed = crate::copystat::TimedNs(std::time::Instant::now(), crate::copystat::C::PrepFlushNs);
            batch.flush();
        }
        let prepared = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::flight_prepared();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::FaultsPrep, crate::copystat::page_faults().saturating_sub(faults));
        // Diagnostics, timing off: a timestamp before and after the lists.
        #[cfg(feature = "diagnostics")]
        let clock = (!timing())
            .then(|| self.clock.get_or_init(|| crate::gpumeasure::Clock::new(&self.device, &self.queue)))
            .and_then(Option::as_ref);
        #[cfg(feature = "diagnostics")]
        if let Some(clock) = clock {
            lists.insert(0, Some(clock.begin.clone()));
            lists.push(Some(clock.end.clone()));
        }
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let groups: Vec<std::ops::Range<usize>> = if timing() {
            firsts
                .iter()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|(&a, &b)| a..b)
                .collect()
        } else {
            std::iter::once(0..lists.len()).collect()
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = std::iter::once(0..lists.len());
        // Clear read marks of slots forgotten before this submission (placed
        // textures written or evicted) so reads in it are not wiped after it.
        {
            #[cfg(feature = "diagnostics")]
            let _timed = crate::copystat::TimedNs(std::time::Instant::now(), crate::copystat::C::PrepForgottenNs);
            crate::shadow::clear_forgotten_reads();
        }
        // What shaders write into copies in video memory comes back into
        // their mirrors (`crate::shadow::pull`, for the copy-backs) in an
        // ExecuteCommandLists of its own right after the lists': its end
        // waits for their writes and leaves the buffers common, so the
        // copies promote them as a submission of their own would; one
        // fence wait covers both. Recorded now: the stale bits it clears
        // are set again by any refresh after, and video memory written
        // before is seen by the copies.
        let riding = if pull_same_execute() && !snapshots.is_empty() {
            self.pull_riding(&snapshots)
        } else {
            None
        };
        // The pull ran and was waited for with the last group.
        let pulled = matches!(riding, Some(None));
        let mut t = Sent {
            _host: std::marker::PhantomData,
            groups: groups.len(),
            value: *submitted,
            started,
            #[cfg(feature = "diagnostics")]
            flight,
            #[cfg(feature = "diagnostics")]
            sync_heaps_ns,
            #[cfg(feature = "diagnostics")]
            traced,
            #[cfg(feature = "diagnostics")]
            clock,
            #[cfg(feature = "diagnostics")]
            sent: None,
            heaps,
            uploads,
            runs: given,
            lists,
            snapshots,
            batch,
            prepared,
            riding,
            pulled,
            removed_in: None,
            unwaited: false,
            last: None,
            marks: None,
        };
        for (i, group) in groups.into_iter().enumerate() {
            *submitted += 1;
            crate::memsys::submission_sent(*submitted);
            t.value = *submitted;
            // Before the lists go: what reads view heap slots from here on
            // is covered by this value ([`sent`]).
            SENT.fetch_max(t.value, Ordering::SeqCst);
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            crate::resource::life::note(0, crate::resource::life::SUBMITTED, t.value, 0);
            #[cfg(feature = "diagnostics")]
            if t.sent.is_none() {
                crate::copystat::flight_sent(&mut t.flight);
                t.sent = Some(std::time::Instant::now());
            }
            let last = i + 1 == t.groups;
            // SAFETY: closed lists; the fence and event are ours.
            let waited = unsafe {
                self.queue.ExecuteCommandLists(&t.lists[group]);
                if last && let Some(Some(r)) = &mut t.riding {
                    self.queue.ExecuteCommandLists(&[Some(r.list.clone())]);
                    r.pull().value = t.value;
                }
                // Its write marks become its own, and the bitmap clear
                // for the next, in queue order.
                if last
                    && crate::memsys::gating()
                    && let Some((list, snap)) = crate::memsys::marks::take(self, t.value)
                {
                    self.queue.ExecuteCommandLists(&[Some(list)]);
                    t.marks = Some(snap);
                }
                self.queue
                    .Signal(&self.fence, t.value)
                    .and_then(|()| self.fence.SetEventOnCompletion(t.value, self.event))
            };
            // The last group is waited for when the ticket is retired.
            if last {
                t.last = Some((i, started, waited));
                break;
            }
            if !self.end_group(&mut t, i, started, waited) {
                break;
            }
        }
        // Its last group never went: the pull list did not either.
        if t.last.is_none()
            && let Some(Some(r)) = &t.riding
        {
            r.pull().busy = false;
        }
        // Pending until retired, for queue waits to read ahead.
        crate::queue::labels_sent(t.value, lists_of(&t.uploads, &t.runs).iter().flat_map(|l| l.deferred().writes.clone()));
        Ticket(Some(t))
    }

    /// Waits for group `i` of a ticket, sent at `started`: false when the
    /// device is gone and no more of it may run.
    fn end_group<L: std::ops::Deref<Target = HostList>>(
        &self,
        t: &mut Sent<'_, L>,
        i: usize,
        started: std::time::Instant,
        waited: windows::core::Result<()>,
    ) -> bool {
        let all = lists_of(&t.uploads, &t.runs);
        let mut ended = Waited::Done;
        let last = i + 1 == t.groups;
        // SAFETY: the fence and event are ours.
        unsafe {
            match waited {
                Ok(()) => {
                    ended = wait_for_submission(
                        // The event is auto-reset and shared: an earlier wait's
                        // signal may wake this one (#2899).
                        || {
                            WaitForSingleObject(self.event, WAIT_SLICE_MS) != WAIT_TIMEOUT
                                && self.fence.GetCompletedValue() >= t.value
                        },
                        || self.device.GetDeviceRemovedReason().is_err(),
                        |secs| {
                            tracing::warn!(
                                secs,
                                lists = all.len(),
                                "host submission still on the GPU"
                            );
                        },
                    );
                }
                Err(e) => {
                    t.unwaited = true;
                    tracing::warn!(error = %e, "host submission not waited for");
                }
            }
        }
        match ended {
            // Run, unless a wait failed: then nothing says so.
            Waited::Done => {
                if !t.unwaited {
                    crate::memsys::submission_done(t.value);
                }
            }
            // Nothing more runs on a removed device; reported below.
            Waited::Removed => {
                if t.groups > 1 && t.removed_in.is_none() {
                    t.removed_in = Some(i);
                }
                return false;
            }
            Waited::Stalled => fail(format_args!(
                "host GPU has not finished a submission of {} lists in {STALL_SECS} s and reports \
                 no removal: the GPU has most likely faulted or hung (on Linux the kernel log \
                 names a fault){}",
                all.len(),
                last_notes(&all)
            )),
        }
        // `ended` stays `Done` when the wait could not be set up.
        if last
            && !t.unwaited
            && matches!(ended, Waited::Done)
            && let Some(Some(r)) = &mut t.riding
        {
            r.pull().busy = false;
            t.pulled = true;
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::PullsRiding, 1);
        }
        #[cfg(feature = "diagnostics")]
        crate::resource::life::note(0, crate::resource::life::RAN, t.value, 0);
        #[cfg(feature = "diagnostics")]
        if let Some(clock) = t.clock
            && !t.unwaited
        {
            clock.add(started.elapsed());
        }
        #[cfg(feature = "diagnostics")]
        if t.groups > 1 {
            // SAFETY: a plain query.
            let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
            gpu_times(started.elapsed(), &all[i].slowest(frequency, usize::MAX));
        }
        #[cfg(feature = "diagnostics")]
        if t.groups > 1 && started.elapsed() > std::time::Duration::from_millis(50) {
            // SAFETY: a plain query.
            let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
            let commands = all[i].notes.lock().unwrap_or_else(|e| e.into_inner()).len();
            tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
            for (ms, op) in all[i].slowest(frequency, 6) {
                tracing::warn!("slow list command, {ms:.1} ms: {op}");
            }
        }
        // SAFETY: a plain query.
        if t.groups > 1 && t.removed_in.is_none() && unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            t.removed_in = Some(i);
        }
        let _ = (i, started);
        true
    }

    /// Waits for a submission [`Host::submit`] sent and settles it: what
    /// shaders wrote goes back to title memory, then the deferred copies,
    /// work and bottom-of-pipe writes.
    pub(crate) fn retire<L: std::ops::Deref<Target = HostList>>(&self, mut ticket: Ticket<'_, L>) {
        let Some(mut t) = ticket.0.take() else { return };
        if let Some((i, started, waited)) = t.last.take() {
            self.end_group(&mut t, i, started, waited);
        }
        let Sent { value, started, heaps, uploads, runs: given, snapshots, batch, prepared, mut riding, pulled, removed_in, unwaited, marks, .. } = t;
        let lists = lists_of(&uploads, &given);
        let runs = &lists[..];
        // The riding copies ran, or never will: their pages unlock before
        // anything here reads a mirror. Not waited for, they may run later:
        // their pages stay stale.
        if let Some(Some(r)) = &mut riding {
            if unwaited {
                r.stale.restale();
            }
            r.stale.release();
        }
        #[cfg(feature = "diagnostics")]
        let waited_at = std::time::Instant::now();
        let gpu = prepared.elapsed();
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs),
                removed_during(runs, removed_in)
            ));
        }
        #[cfg(feature = "diagnostics")]
        if !unwaited {
            crate::gpumeasure::add_statistics(runs);
        }
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        // One list's lock at a time: a submission may name a list twice.
        // Copy, address, bytes, goes back, writes marked.
        let mut back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool, bool)> = Vec::new();
        let mut index: std::collections::HashMap<(usize, usize, usize), usize> = std::collections::HashMap::new();
        // Shaders that read memory with no copy computed something else than
        // the console's: it stays out of title memory, where it broke the
        // title's own data (runs c1, c4), and so does what every program
        // using title memory wrote while any misses
        // (`crate::memory::withholding`). Only lists with such programs have
        // checks; all are asked before anything goes back. Other programs in
        // those lists write through bounded views from their own inputs:
        // their ranges go back (the imposter transforms beside the BVH
        // builder in one list, run light13).
        let missed: Vec<bool> = runs
            .iter()
            .map(|list| crate::memory::withholds() && list.deferred().missed.iter().any(|check| check()))
            .collect();
        crate::memory::note_submission(missed.contains(&true));
        let withholding = crate::memory::withholding();
        #[cfg(feature = "diagnostics")]
        {
            // Collected before the log's lock.
            let lists: Vec<_> = runs
                .iter()
                .zip(&missed)
                .map(|(list, &missed)| {
                    let memory = !list.deferred().missed.is_empty();
                    let bound = list.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                    let logged = memory || bound.iter().any(|p| p.contains(crate::lightlog::RAY));
                    let pipelines = if logged { bound.clone() } else { Default::default() };
                    (memory, missed, pipelines)
                })
                .collect();
            crate::lightlog::submission(withholding, lists.into_iter());
        }
        for (list, &missed) in runs.iter().zip(&missed) {
            let deferred = list.deferred();
            let missed = missed || withholding && !deferred.missed.is_empty();
            for step in &deferred.copies {
                if let Step::GpuWrite(shadow, address, len, marked, memory) = step {
                    let key = (std::sync::Arc::as_ptr(shadow) as usize, *address, *len);
                    // A range both kinds declare is withheld whole.
                    let goes_back = !(missed && *memory);
                    match index.get(&key) {
                        Some(&i) => {
                            back[i].3 &= goes_back;
                            back[i].4 &= *marked;
                        }
                        None => {
                            index.insert(key, back.len());
                            back.push((shadow.clone(), *address, *len, goes_back, *marked));
                        }
                    }
                }
            }
        }
        // A range overlapping a withheld one of the same copy is withheld
        // too: its copy-back would take the other's bytes along (pages a
        // program using title memory and another both wrote).
        let held: Vec<(usize, usize, usize)> = back
            .iter()
            .filter(|b| !b.3)
            .map(|(s, a, n, ..)| (std::sync::Arc::as_ptr(s) as usize, *a, *n))
            .collect();
        for b in back.iter_mut().filter(|b| b.3) {
            let copy = std::sync::Arc::as_ptr(&b.0) as usize;
            b.3 = !held.iter().any(|&(c, a, n)| c == copy && a < b.1 + b.2 && b.1 < a + n);
        }
        // Pages shaders marked first: the copy-backs below end what the
        // copies keep for them.
        let withheld: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = back
            .iter()
            .filter(|b| !b.3)
            .map(|(s, a, n, ..)| (s.clone(), *a, *n))
            .collect();
        {
            #[cfg(feature = "diagnostics")]
            let _timed = crate::copystat::TimedNs(std::time::Instant::now(), crate::copystat::C::SettleMarksNs);
            // Not waited for, its marks may not be in the ring yet: the bitmap.
            crate::shadow::copy_back_marks_of(&withheld, withholding, marks.as_ref().filter(|_| !unwaited));
        }
        if let Some(snap) = marks {
            crate::memsys::marks::done(snap);
        }
        // Texture views left null that shaders read: made for what follows.
        #[cfg(feature = "diagnostics")]
        let reading = crate::copystat::TimedNs(std::time::Instant::now(), crate::copystat::C::SettleReadsNs);
        let read = crate::shadow::take_reads();
        if !read.is_empty() {
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::add(crate::copystat::C::MadeReadSlots, read.len() as u64);
                crate::copystat::add(crate::copystat::C::MadeReadSubs, 1);
            }
            crate::descriptor::make_read(&read);
        }
        #[cfg(feature = "diagnostics")]
        drop(reading);
        // What shaders wrote into copies in video memory, for the copy-backs
        // below to read from their mirrors: in the ranges with snapshots
        // (marked ones compare nothing; `copy_back_marks` pulled their pages).
        // Done with the lists when it rode them (`riding`): what ran on the
        // CPU since keeps video memory and the mirrors alike, or marks the
        // pages stale.
        #[cfg(feature = "diagnostics")]
        let pulling = crate::copystat::TimedNs(std::time::Instant::now(), crate::copystat::C::SettlePullNs);
        if !pulled {
            crate::shadow::pull(&snapshots);
            if let Some(Some(r)) = &mut riding {
                static FELL_BACK: std::sync::Once = std::sync::Once::new();
                FELL_BACK.call_once(|| tracing::warn!("a riding pull was not waited for: it pulled on its own"));
                // Done once the submission's fence passed it: the pull's own
                // copy may not have run or waited (an error, nothing to copy).
                // SAFETY: a plain query.
                r.pull().busy &= unsafe { self.fence.GetCompletedValue() } < value;
            }
        }
        #[cfg(feature = "diagnostics")]
        drop(pulling);
        /// Held only to be dropped with the submission's buffers.
        struct Kept {
            _regions: Regions,
        }
        // SAFETY: D3D12 resources are free-threaded; only released there.
        unsafe impl Send for Kept {}
        // Not waited for: its buffers go with the submission's (`back`).
        let kept = match riding.take() {
            Some(Some(r)) if unwaited => Some(Kept { _regions: r._regions }),
            _ => None,
        };
        static RETAKE_ALL: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var_os("VICTORY_RETAKE_ALL").is_some_and(|v| v != "0"));
        #[cfg(feature = "diagnostics")]
        let backing = crate::copystat::TimedNs(std::time::Instant::now(), crate::copystat::C::SettleBackNs);
        for (shadow, address, len, goes_back, marked) in &back {
            let (address, len) = (*address, *len);
            if *goes_back {
                let written = shadow.copy_back(address, len);
                COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            } else {
                shadow.keep_writes(address, len);
                crate::memory::note_withheld(len as u64);
                // Writes that mark their pages: `copy_back_marks` dropped
                // those, and the others hold nothing of the shaders'.
                // `VICTORY_RETAKE_ALL=1`: the whole range.
                if !*marked || *RETAKE_ALL {
                    crate::memory::drop_withheld(address, len);
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        drop(backing);
        crate::shadow::snapshots_done(&snapshots);
        #[cfg(feature = "diagnostics")]
        let after = crate::copystat::TimedNs(std::time::Instant::now(), crate::copystat::C::SettleAfterNs);
        // Copies kept past the lists' run: what the shaders wrote reached
        // title memory in the copy-backs above, so a deferred copy's source
        // is new now (and its destination is not overwritten by them).
        // A range the copy-back withheld is still old.
        for &(dst, src, len) in &batch.late {
            copy_mapped(dst as *mut u8, src as *const u8, len, true);
        }
        for list in runs {
            for work in &list.deferred().after {
                work();
            }
        }
        #[cfg(feature = "diagnostics")]
        drop(after);
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::FaultsBack, crate::copystat::page_faults().saturating_sub(faults));
        for list in runs {
            for &(at, what, bytes) in &list.deferred().writes {
                let what = match what {
                    // Visible: a zero not yet resolved would hide the object.
                    crate::queue::Written::HostValue(_) if unwaited => crate::queue::Written::Value(1 << 20),
                    what => what,
                };
                #[cfg(feature = "diagnostics")]
                crate::copystat::label_written(at, &what);
                what.write(at as *mut u8, bytes);
            }
        }
        crate::queue::labels_retired(value);
        #[cfg(feature = "diagnostics")]
        if let Some(f) = t.flight {
            let sent = t.sent.unwrap_or(waited_at);
            let ns = |d: std::time::Duration| d.as_nanos() as u64;
            // What the next submission would wait for: the shaders' ranges,
            // late copies' destinations, bottom-of-pipe writes.
            let writes = batch
                .gpu_writes
                .iter()
                .copied()
                .chain(batch.late.iter().map(|&(dst, _, len)| (dst, dst + len)))
                .chain(runs.iter().flat_map(|l| {
                    l.deferred().writes.iter().map(|&(at, _, bytes)| (at, at + bytes)).collect::<Vec<_>>()
                }))
                .collect();
            let copies = back.iter().map(|b| std::sync::Arc::as_ptr(&b.0) as usize).collect();
            crate::copystat::flight_end(
                f,
                batch.flight_touched,
                [ns(sent - started), t.sync_heaps_ns, ns(waited_at - sent), ns(copying.elapsed())],
                writes,
                copies,
                crate::copystat::Mine {
                    real: batch.flight_real,
                    shared: batch.flight_shared,
                    used: batch.flight_used.clone(),
                    marked: batch.flight_marked,
                    hazard: batch.flight_hazard,
                },
            );
        }
        // The submission has run and the next waits for our lock. Host
        // copies the title gave up meanwhile may have their last holder here.
        release_retired();
        release_later(Box::new((batch, back, withheld, heaps, kept)));
        #[cfg(feature = "diagnostics")]
        crate::copystat::execute_end();
        self.log_messages();
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                    #[cfg(feature = "diagnostics")]
                    for name in names.iter().filter(|_| what == "inside freed") {
                        text.push_str(&crate::resource::life::of(name));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // The nodes, oldest list first, and which run each is.
            let mut nodes = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                nodes.push(p);
                node = p.pNext;
            }
            let index: Vec<Option<usize>> = nodes
                .iter()
                .map(|p| runs.iter().position(|r| r.is_node(p)))
                .collect();
            if let Some(last) = nodes.last()
                && index.iter().all(Option::is_none)
            {
                let name = |w: windows::core::PCWSTR, a: *const u8| {
                    let w = (!w.is_null()).then(|| w.to_string().unwrap_or_default());
                    let a = (!a.is_null()).then(|| {
                        std::ffi::CStr::from_ptr(a.cast())
                            .to_string_lossy()
                            .into_owned()
                    });
                    format!("{w:?}/{a:?}")
                };
                tracing::warn!(
                    nodes = nodes.len(),
                    last_pointer = ?last.pCommandList.as_ref().map(|l| l.as_raw()),
                    last_names = %name(last.pCommandListDebugNameW, last.pCommandListDebugNameA),
                    "no breadcrumb node named one of our lists"
                );
            }
            // Every list's progress (done/count), in the runtime's order.
            let progress: Vec<String> = nodes
                .iter()
                .zip(&index)
                .map(|(p, index)| {
                    let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                    format!("{done}/{} {index:?}", p.BreadcrumbCount)
                })
                .collect();
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            for (p, index) in nodes.iter().zip(&index) {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let run = index.map(|i| runs[i]);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A slot of CPU-only views, valid until it is recorded.
    pub(crate) fn cpu_view(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_cpu_view.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a live heap.
        let mut handle = unsafe { self.cpu_views.GetCPUDescriptorHandleForHeapStart() };
        handle.ptr += (slot * self.cpu_view_size) as usize;
        handle
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let bytes = u64::from(STAMPS) * 8;
                crate::gpumeasure::queries(&host()?.device, D3D12_QUERY_HEAP_TYPE_TIMESTAMP, STAMPS, bytes)
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                let note = crate::gpumeasure::span_note(i, count, notes.len(), &notes[i].0);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, note)
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    /// Whether `node` of the device-removed data (its breadcrumbs) is this
    /// list.
    fn is_node(&self, node: &D3D12_AUTO_BREADCRUMB_NODE) -> bool {
        // SAFETY: only the pointer and the name text are read: the pointer is
        // compared, never used, and the name is the runtime's copy.
        let (pointer, wide, ansi) = unsafe {
            (
                node.pCommandList.as_ref().map(|l| l.as_raw() as usize),
                (!node.pCommandListDebugNameW.is_null())
                    .then(|| node.pCommandListDebugNameW.to_string().unwrap_or_default()),
                (!node.pCommandListDebugNameA.is_null()).then(|| {
                    std::ffi::CStr::from_ptr(node.pCommandListDebugNameA.cast())
                        .to_string_lossy()
                        .into_owned()
                }),
            )
        };
        let (pointers, name) = (identities(&self.list), list_name(self.id));
        node_is_list(pointer, wide.as_deref(), &pointers, &name)
            || node_is_list(None, ansi.as_deref(), &pointers, &name)
    }

    /// The list's statistics query, if it has one.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn statistics(&self) -> Option<&crate::gpumeasure::ListStats> {
        self.stats.get()?.as_ref()
    }

    pub(crate) fn close(&self) {
        crate::indirect::end_window(self);
        #[cfg(feature = "diagnostics")]
        self.note_host("host-close");
        // The host closes no list with queries running; their title ends,
        // in a later list, are dropped (`end_query`).
        for (heap, kind, index, _) in self.queries.lock().unwrap_or_else(|e| e.into_inner()).drain(..) {
            // SAFETY: an open list and a query begun in it.
            unsafe { self.list.EndQuery(&heap, kind, index) };
        }
        for (from, to, bytes) in &self.deferred().read_backs {
            // SAFETY: an open list; buffers of ours it holds.
            unsafe {
                self.list.ResourceBarrier(&[transition(
                    from,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                self.list.CopyBufferRegion(to, 0, from, 0, *bytes);
            }
        }
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let noted = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            let count = noted.min(STAMPS - 1);
            UNTIMED.fetch_add(u64::from(noted - count), Ordering::Relaxed);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        #[cfg(feature = "diagnostics")]
        if let Some(stats) = self.statistics() {
            stats.end(&self.list);
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        self.synced.store(false, Ordering::Relaxed);
        *self.deferred() = Deferred::default();
        *self.clamp_window() = None;
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.queries.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.targets.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
        #[cfg(feature = "diagnostics")]
        if let Some(stats) = self.statistics() {
            stats.begin(&self.list);
        }
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    /// A depth texture with a writable twin (`crate::twin`) has the side
    /// the use needs brought up to date first; unordered access moves the
    /// twin, not the depth texture (nothing returned).
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(ID3D12Resource, u32)> {
        // One host texture for the whole move (`crate::resource::evict` may
        // give it up meanwhile; this list then holds it).
        let copy = crate::resource::host_copy(resource)?;
        let texture = copy.texture.clone();
        if copy.packed().is_some() {
            crate::texture::upload(self, resource, &texture);
        }
        if let Some(Some(twin)) = copy.twin.get()
            && crate::twin::route(self, &copy, twin, wanted)
        {
            return None;
        }
        let wanted = legal_state(&texture, wanted);
        self.track(&copy, wanted).map(|before| (texture, before))
    }

    /// The list's state for host texture `copy` becomes `wanted` (a legal
    /// one): the state it leaves, if that is a change. Nothing for the
    /// list's first use (its prelude moves the texture there).
    fn track(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) -> Option<u32> {
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                copy.note_used();
                if !self.ours.load(Ordering::Relaxed) {
                    copy.stamp();
                }
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        }
    }

    /// [`HostList::transition`] for a host texture of ours (a twin's
    /// plane), without twins: the barrier to record, if one is needed.
    pub(crate) fn move_texture(
        &self,
        copy: &std::sync::Arc<crate::resource::HostTexture>,
        wanted: u32,
    ) -> Option<D3D12_RESOURCE_BARRIER> {
        let wanted = legal_state(&copy.texture, wanted);
        let before = self.track(copy, wanted)?;
        Some(transition(
            &copy.texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(wanted as i32),
        ))
    }

    /// A use of the depth texture `depth` (with a twin) needing `side`
    /// (`crate::twin::DEPTH` or `TWIN`): whether a copy must be recorded
    /// now (true: into the depth texture). The list's first use is the
    /// prelude's to prepare.
    pub(crate) fn twin_use(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        side: u8,
        writes: bool,
    ) -> Option<bool> {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        match twins.entry(std::sync::Arc::as_ptr(depth) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(crate::twin::Use::new(depth, side, writes));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => o.get_mut().again(side, writes),
        }
    }

    /// The list is one of ours (uploads): what it uses is no use of the
    /// title's.
    pub(crate) fn set_ours(&self) {
        self.ours.store(true, Ordering::Relaxed);
    }

    /// What the list's shaders write from now until the guard goes is
    /// written by a program using title memory (`uses`) or not.
    pub(crate) fn writer_uses_memory(&self, uses: bool) -> impl Drop + '_ {
        struct Reset<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Relaxed);
            }
        }
        self.memory_writer.store(uses, Ordering::Relaxed);
        Reset(&self.memory_writer)
    }

    /// A copy of ours replaced the compute root signature and pipeline.
    pub(crate) fn need_rebind(&self) {
        self.rebind.store(true, Ordering::Relaxed);
    }

    /// Whether the list's compute root signature and pipeline must be set
    /// again (once).
    pub(crate) fn take_rebind(&self) -> bool {
        self.rebind.swap(false, Ordering::Relaxed)
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with.
    /// `nth` counts the list's earlier places in the same submission.
    fn prelude(&self, host: &Host, nth: usize) -> Option<ID3D12CommandList> {
        // Depth textures and their twins first, from their real states and
        // back: the side each first use needs, made newer.
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let copies = crate::twin::at_submission(twins.values());
        drop(twins);
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        // The copies start from the real states, then the list's moves from
        // them; only after both do they become the states the list leaves.
        let moves = || -> Vec<D3D12_RESOURCE_BARRIER> {
            states
                .values()
                .filter_map(|(first, _, copy)| {
                    let real = copy.state.load(Ordering::Relaxed);
                    (real != *first).then(|| {
                        transition(&copy.texture, D3D12_RESOURCE_STATES(real as i32), D3D12_RESOURCE_STATES(*first as i32))
                    })
                })
                .collect()
        };
        let left = || {
            for (_, last, copy) in states.values() {
                copy.state.store(*last, Ordering::Relaxed);
            }
        };
        if copies.is_empty() && states.values().all(|(first, _, copy)| copy.state.load(Ordering::Relaxed) == *first) {
            left();
            return None;
        }
        let mut preludes = self.preludes.lock().unwrap_or_else(|e| e.into_inner());
        while preludes.len() <= nth {
            // SAFETY: plain object creation.
            let made = unsafe {
                host.device
                    .CreateCommandAllocator::<ID3D12CommandAllocator>(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .and_then(|allocator| {
                        let list: ID3D12GraphicsCommandList =
                            host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                        list.Close()?;
                        Ok((allocator, list))
                    })
            };
            match made {
                Ok(pair) => preludes.push(pair),
                Err(e) => fail(format_args!("no host state prelude ({e})")),
            }
        }
        let (allocator, list) = &preludes[nth];
        // SAFETY: this prelude is not executing (every submission is waited
        // for); the textures are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                for (depth, to_depth) in &copies {
                    let Some(Some(twin)) = depth.twin.get() else { continue };
                    // From each texture's real state to the copy's and back.
                    let mut back = Vec::new();
                    twin.copy(list, depth, *to_depth, &mut |t, wanted| {
                        let real = t.state.load(Ordering::Relaxed);
                        let wanted = legal_state(&t.texture, wanted);
                        (real != wanted).then(|| {
                            let there = transition(&t.texture, D3D12_RESOURCE_STATES(real as i32), D3D12_RESOURCE_STATES(wanted as i32));
                            back.push(transition(
                                &t.texture,
                                D3D12_RESOURCE_STATES(wanted as i32),
                                D3D12_RESOURCE_STATES(real as i32),
                            ));
                            there
                        })
                    });
                    if !back.is_empty() {
                        list.ResourceBarrier(&back);
                    }
                }
                let barriers = moves();
                if !barriers.is_empty() {
                    list.ResourceBarrier(&barriers);
                }
                list.Close()
            })
        };
        left();
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    /// With `VICTORY_GPU_TIMING`: what the host records next (setup,
    /// barriers, a command without a note of its own) is timed as `what`,
    /// not as the command before it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_host(&self, what: &'static str) {
        if timing() {
            self.note(|| what.to_owned(), None);
        }
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        // Timing alone needs the stamps too, not only device-removed data.
        if dred_on() || timing() {
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            traced(&what);
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
        }
    }

    /// Notes a texture the list renders to or writes, in a listed frame.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_target(&self, resource: usize, by: &str) {
        let mut targets = self.targets.lock().unwrap_or_else(|e| e.into_inner());
        if !targets.iter().any(|t| t.0 == resource) {
            targets.push((resource, by.to_owned()));
        }
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        // Kept in the copy's own addresses (a buffer sharing it has others).
        let address = shadow.canon(address as u64) as usize;
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// A draw or dispatch that may write through views is being recorded.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_shader_write(&self) {
        let mut deferred = self.deferred();
        deferred.clamp_clean = false;
        deferred.args_ordered = false;
    }

    /// A title barrier group ordering indirect arguments is being recorded.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_args_barrier(&self) {
        self.deferred().args_ordered = true;
    }

    /// An indirect clamp is being recorded.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_clamp(&self) {
        use crate::copystat::{C, add};
        add(C::Clamps, 1);
        let mut deferred = self.deferred();
        if std::mem::replace(&mut deferred.clamp_clean, true) {
            add(C::ClampsNoWrite, 1);
        }
        // Left set: the clamp writes no title data the next one reads.
        if deferred.args_ordered {
            add(C::ClampsArgsOrdered, 1);
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run
    /// (`marked`: the pages they mark).
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
        marked: bool,
    ) {
        let address = shadow.canon(address as u64) as usize;
        let memory = self.memory_writer.load(Ordering::Relaxed);
        self.deferred()
            .copies
            .push(Step::GpuWrite(shadow.clone(), address, len, marked, memory));
    }

    /// Shaders of the list take views from the heap itself
    /// (`crate::bindless`): what the heap's views reach is brought up to
    /// date at submission (once until a title copy intervenes).
    pub(crate) fn defer_heap(&self, viewed: &std::sync::Arc<crate::bindless::Viewed>) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(viewed) as usize, usize::MAX, 0);
        if deferred.refreshed.insert(key) {
            deferred.copies.push(Step::Heap(viewed.clone()));
        }
    }

    /// Shaders of the list read title memory through the page table of
    /// view heap `heap` (`crate::memory::prepare` at submission).
    pub(crate) fn defer_memory(&self, heap: usize) {
        self.deferred().copies.push(Step::Memory(heap));
    }

    /// After the list has run, before its writes go back: `check` says
    /// whether its shaders read title memory with no copy.
    pub(crate) fn defer_missed(&self, check: impl Fn() -> bool + Send + 'static) {
        self.deferred().missed.push(Box::new(check));
    }

    /// At the list's end (when it is closed), `bytes` of `from` (in the
    /// unordered-access state) are copied into the read-back buffer `to`.
    pub(crate) fn read_back_at_close(
        &self,
        from: &ID3D12Resource,
        to: &ID3D12Resource,
        bytes: u64,
    ) {
        self.deferred()
            .read_backs
            .push((from.clone(), to.clone(), bytes));
    }

    /// Runs `work` once the list has run (each submission), before its
    /// bottom-of-pipe writes.
    pub(crate) fn defer_after(&self, work: impl Fn() + Send + 'static) {
        self.deferred().after.push(Box::new(work));
    }

    /// Its bottom-of-pipe writes alone, for a list whose commands do not
    /// run (`crate::queue`, a list that lost its slots): a query result
    /// is not resolved, so it is the visible one, as when not waited for.
    pub(crate) fn labels(&self) -> Vec<(usize, crate::queue::Written, usize)> {
        let unresolved = |what| match what {
            crate::queue::Written::HostValue(_) => crate::queue::Written::Value(1 << 20),
            what => what,
        };
        self.deferred().writes.iter().map(|&(at, what, bytes)| (at, unresolved(what), bytes)).collect()
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// A memory value the list waits for before it starts.
    pub(crate) fn defer_wait(&self, wait: crate::hold::ValueWait) {
        self.deferred().waits.push(wait);
    }

    /// The memory values the list waits for before it starts.
    pub(crate) fn waits(&self) -> Vec<crate::hold::ValueWait> {
        self.deferred().waits.clone()
    }

    /// Whether nothing has been recorded since the list was made or reset
    /// that a wait recorded now would have to follow: no work, texture
    /// use, bottom-of-pipe write or pipeline (state settings alone carry
    /// over).
    pub(crate) fn is_empty(&self) -> bool {
        let d = self.deferred();
        let deferred_empty = d.copies.is_empty() && d.uploads.is_empty() && d.writes.is_empty() && d.after.is_empty() && d.missed.is_empty();
        drop(d);
        deferred_empty
            && self.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.twins.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.upload_buffer(size)?;
        crate::memsys::add(crate::memsys::N::ListUploads, 1);
        crate::memsys::add(crate::memsys::N::ListUploadBytes, size);
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// A mapped readback buffer of `size` bytes, alive until the list is
    /// reset.
    pub(crate) fn readback(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.mapped_buffer(size, true)?;
        crate::memsys::add(crate::memsys::N::ListReadbacks, 1);
        crate::memsys::add(crate::memsys::N::ListReadbackBytes, size);
        name(&buffer, || format!("list readback {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).checked_next_multiple_of(256)?;
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().uploads.push(buffer);
            return Some((gpu, data));
        }
        while let Some(chunk) = s.chunks.get(s.current) {
            if len <= chunk.capacity - s.used {
                break;
            }
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let next = s.chunks.last().map_or(FIRST_SCRATCH_CHUNK, |c| 2 * c.capacity);
            let capacity = len.next_multiple_of(FIRST_SCRATCH_CHUNK).max(next).min(SCRATCH_CHUNK);
            let (buffer, data) = host.upload_buffer(capacity)?;
            name(&buffer, || format!("list scratch {capacity:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            #[cfg(feature = "diagnostics")]
            tracing::debug!(target: "victory_gpu::scratch", list = ?std::ptr::from_ref(self),
                chunk = s.current, requested = len, capacity, "allocated list scratch");
            s.chunks.push(ScratchChunk { _buffer: buffer, mapped: data as usize, gpu, capacity });
            s.used = 0;
        }
        let chunk = &s.chunks[s.current];
        let at = s.used;
        let result = (chunk.gpu + at, (chunk.mapped + at as usize) as *mut u8);
        s.used += len;
        Some(result)
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped word, written before it runs.
            deferred
                .copies
                .push(Step::Host(Box::new(move || unsafe { (at as *mut u32).write_volatile(0) })));
        }
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    /// The list's argument buffer, at least `bytes` long, common between
    /// its uses (`crate::indirect`).
    pub(crate) fn arguments(&self, bytes: u64) -> Option<ID3D12Resource> {
        let mut arguments = self.arguments.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: a plain query on a live buffer.
        if let Some(b) = arguments.as_ref().filter(|b| unsafe { b.GetDesc() }.Width >= bytes) {
            return Some(b.clone());
        }
        let (made, _) = host()?.shared_buffer(bytes.next_power_of_two().max(1 << 16))?;
        name(&made, || "indirect arguments".into());
        // Commands recorded earlier may still name the smaller one.
        if let Some(old) = arguments.replace(made.clone()) {
            self.keep(&old);
        }
        Some(made)
    }

    /// The list's open window of indirect draws clamped together.
    pub(crate) fn clamp_window(&self) -> std::sync::MutexGuard<'_, Option<crate::indirect::Window>> {
        self.clamp_window.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The list's window check buffers, made on first use.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn window_check(&self) -> Option<&crate::indirect::WindowCheck> {
        self.window_check.get_or_init(crate::indirect::WindowCheck::new).as_ref()
    }

    /// Begins occlusion query `index` of `heap` unless it is running in
    /// the list already. Whether it began.
    pub(crate) fn begin_query(&self, heap: &ID3D12QueryHeap, kind: u32, index: u32) -> bool {
        let mut queries = self.queries.lock().unwrap_or_else(|e| e.into_inner());
        if queries.iter().any(|(h, _, i, _)| h == heap && *i == index) {
            return false;
        }
        let kind = D3D12_QUERY_TYPE(kind as i32);
        // SAFETY: an open list and a live heap, kept until the list is reset.
        unsafe { self.list.BeginQuery(heap, kind, index) };
        queries.push((heap.clone(), kind, index, false));
        drop(queries);
        self.keep(heap);
        true
    }

    /// Ends query `index` of `heap` if it began in this list, as the type
    /// it began as. That type, if the result is usable: a query with a draw
    /// inside that the host did not run counts too few samples, and the
    /// title would cull by it what the draw shows.
    pub(crate) fn end_query(&self, heap: &ID3D12QueryHeap, index: u32) -> Option<u32> {
        let mut queries = self.queries.lock().unwrap_or_else(|e| e.into_inner());
        let at = queries.iter().position(|(h, _, i, _)| h == heap && *i == index)?;
        let (_, kind, _, missed) = queries.swap_remove(at);
        // SAFETY: an open list and a query begun in it.
        unsafe { self.list.EndQuery(heap, kind, index) };
        (!missed).then_some(kind.0 as u32)
    }

    /// A draw the title recorded was not run on the host: the queries
    /// running around it did not see it.
    pub(crate) fn missed_draw(&self) {
        for q in self.queries.lock().unwrap_or_else(|e| e.into_inner()).iter_mut() {
            q.3 = true;
        }
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Run(Box::new(work)));
        self.deferred().refreshed.clear();
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Brings a texture's staging buffer and host texture `copy` up to date
    /// when the list is submitted (`crate::bindless::uploads`), keeping
    /// both alive until the list is reset.
    pub(crate) fn defer_texture(
        &self,
        staging: std::sync::Arc<crate::texture::Staging>,
        copy: std::sync::Arc<crate::resource::HostTexture>,
    ) {
        self.deferred().copies.push(Step::Texture(staging, copy));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges).
    fn run_deferred(&self, batch: &mut Batch) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        let deferred = self.deferred();
        let mut steps = deferred.copies.iter().peekable();
        while let Some(step) = steps.next() {
            let mut started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(
                started,
                match step {
                    Step::Refresh(..) => C::StepRefreshUs,
                    Step::Copy(..) | Step::Fill(..) => C::StepCopyUs,
                    Step::Run(_) | Step::Host(_) => C::StepWorkUs,
                    Step::GpuWrite(..) => C::StepGpuWriteUs,
                    Step::Texture(..) => C::StepTextureUs,
                    Step::Heap(_) => C::StepHeapUs,
                    Step::Memory(_) => C::StepMemoryUs,
                },
            );
            let copied = match step {
                &Step::Copy(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                        batch.touch(src, len, C::RangeCopy);
                        batch.touch(dst, len, C::RangeCopy);
                    }
                    batch.title_written();
                    // The flush counted its own time.
                    started = std::time::Instant::now();
                    // The copy follows the recorded order: a source the
                    // submission's shaders write is still old here (it
                    // lands in title memory in the copy-back), and a
                    // destination they write would lose their bytes to
                    // it. Either way it runs after the copy-backs.
                    if batch.gpu_written(src, len) || batch.gpu_written(dst, len) {
                        batch.late.push((dst, src, len));
                        0
                    } else {
                        copy_mapped(dst as *mut u8, src as *const u8, len, true);
                        len
                    }
                }
                &Step::Fill(dst, src, len) => {
                    // Always at once, even from a range the shaders write:
                    // the lists read the destination when they run.
                    // shortcut: such a fill, and one over a late copy's
                    // destination, reads the bytes from before; splitting the
                    // submission there would give the new ones.
                    let fill = |dst: usize, src: usize, len: usize| {
                        #[cfg(feature = "diagnostics")]
                        {
                            add(C::Copies, 1);
                            add(C::CopyBytes, len as u64);
                        }
                        copy_mapped(dst as *mut u8, src as *const u8, len, false);
                        len
                    };
                    #[cfg(feature = "diagnostics")]
                    batch.touch(src, len, C::RangeCopy);
                    let mut copied = fill(dst, src, len);
                    // A run of fills (a list's constant buffers, mostly a
                    // few hundred bytes each) is timed as one: reading the
                    // clock around each would cost as much as the copies.
                    while let Some(&Step::Fill(dst, src, len)) =
                        steps.next_if(|s| matches!(s, Step::Fill(..)))
                    {
                        #[cfg(feature = "diagnostics")]
                        batch.touch(src, len, C::RangeCopy);
                        copied += fill(dst, src, len);
                    }
                    copied
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(work) | Step::Host(work) => {
                    if matches!(step, Step::Run(_)) {
                        batch.title_written();
                    }
                    #[cfg(feature = "diagnostics")]
                    let run = std::time::Instant::now();
                    work();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, run.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len, marked, _) => {
                    // The shaders start from the title's data, which a
                    // snapshot must hold (marked ranges keep none).
                    if !marked {
                        batch.flush();
                    }
                    #[cfg(feature = "diagnostics")]
                    batch.touch(*address, *len, C::RangeGpuWrite);
                    #[cfg(feature = "diagnostics")]
                    batch.prep_write(shadow, *marked);
                    batch.gpu_write(*address, *len);
                    shadow.writing(*address, *len, *marked);
                    continue;
                }
                // Brought up to date before the submission's lists were made.
                Step::Texture(..) => continue,
                Step::Heap(viewed) => {
                    crate::bindless::refresh(viewed, |s, a, l| batch.add(s, a, l));
                    continue;
                }
                &Step::Memory(heap) => {
                    crate::memory::prepare(heap, |s, a, l| batch.add(s, a, l));
                    continue;
                }
            };
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range that covers them: done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
///
/// A copy's range brought up to date is not copied again in the submission
/// until a step writes title memory: every list of a world submission using
/// title memory asks for every copy whole. What the title's threads write
/// meanwhile is for later submissions (on the console it would race the
/// GPU).
///
/// A copy none of whose pages changed since it took them is not asked for
/// at all ([`crate::shadow::Shadow::clean`]): lists using title memory
/// ask for every copy there is, a few thousand in the world, of which a
/// submission's writes touch a few.
#[derive(Default)]
struct Batch {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Where each copy's refresh is in `refreshes`, by copy.
    pending: crate::fx::Map<usize, usize>,
    /// The range of each copy brought up to date since the last title
    /// write, by copy (held, so that its address names no other).
    current: crate::fx::Map<usize, (std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Title ranges the submission's shaders are declared to write so far
    /// (start, end): their results reach title memory in the copy-back
    /// after the lists run, so a deferred copy touching one waits for it.
    gpu_writes: Vec<(usize, usize)>,
    /// Deferred title copies kept past the lists' run, in record order (they
    /// run after the copy-backs): destination, source, bytes.
    late: Vec<(usize, usize, usize)>,
    /// Flight study: the last title submission's written ranges (merged),
    /// and whether this one touches them.
    #[cfg(feature = "diagnostics")]
    flight_prev: Vec<(usize, usize)>,
    #[cfg(feature = "diagnostics")]
    flight_touched: bool,
    /// Pipelining study: the host copies the last title submission wrote
    /// (sorted); whether this one takes bytes it wrote other than through a
    /// clean refresh, and whether it uses one of those copies.
    #[cfg(feature = "diagnostics")]
    flight_prev_copies: Vec<usize>,
    #[cfg(feature = "diagnostics")]
    flight_real: bool,
    #[cfg(feature = "diagnostics")]
    flight_shared: bool,
    /// Overlap hazards: the copies the last title submission used (sorted)
    /// and this one uses; whether this one marks pages, and met a hazard.
    #[cfg(feature = "diagnostics")]
    flight_prev_used: Vec<usize>,
    #[cfg(feature = "diagnostics")]
    flight_used: Vec<usize>,
    #[cfg(feature = "diagnostics")]
    flight_marked: bool,
    #[cfg(feature = "diagnostics")]
    flight_hazard: bool,
}

/// The lists and tables of a [`Batch`] that ended, emptied: the next
/// batch takes them, so that a submission's refreshes allocate nothing
/// once the lists have grown to a world submission's size (two tables of
/// a few thousand copies and three lists were made and freed 29 times a
/// frame).
struct Spare {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    pending: crate::fx::Map<usize, usize>,
    current: crate::fx::Map<usize, (std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    gpu_writes: Vec<(usize, usize)>,
    late: Vec<(usize, usize, usize)>,
}

/// At most [`SPARES`] of them: a batch ends on the releasing thread, a
/// little after the next began.
static SPARE: Mutex<Vec<Spare>> = Mutex::new(Vec::new());
const SPARES: usize = 4;

impl Drop for Batch {
    fn drop(&mut self) {
        // The copies held go here, before the lock.
        self.refreshes.clear();
        self.pending.clear();
        self.current.clear();
        self.gpu_writes.clear();
        self.late.clear();
        // Tests: the thread a batch with the mark's room ended on.
        #[cfg(test)]
        if self.gpu_writes.capacity() >= batch::MARK {
            *batch::ENDED_ON.lock().unwrap_or_else(|e| e.into_inner()) = Some(std::thread::current().id());
        }
        if self.refreshes.capacity() == 0 && self.current.capacity() == 0 && self.gpu_writes.capacity() == 0 {
            return;
        }
        let spare = Spare {
            refreshes: std::mem::take(&mut self.refreshes),
            pending: std::mem::take(&mut self.pending),
            current: std::mem::take(&mut self.current),
            gpu_writes: std::mem::take(&mut self.gpu_writes),
            late: std::mem::take(&mut self.late),
        };
        let mut spares = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        if spares.len() < SPARES {
            spares.push(spare);
        }
    }
}

impl Batch {
    /// An empty batch, with the memory of one that ended if there is one.
    fn reused() -> Self {
        let mut batch = Self::default();
        if let Some(spare) = SPARE.lock().unwrap_or_else(|e| e.into_inner()).pop() {
            batch.refreshes = spare.refreshes;
            batch.pending = spare.pending;
            batch.current = spare.current;
            batch.gpu_writes = spare.gpu_writes;
            batch.late = spare.late;
        }
        batch
    }

    /// Flight study: title bytes the submission reads or writes, counted
    /// under `c` when the last title submission wrote them.
    #[cfg(feature = "diagnostics")]
    fn touch(&mut self, at: usize, len: usize, c: crate::copystat::C) {
        if !crate::copystat::touches(&self.flight_prev, at, len) {
            return;
        }
        self.flight_touched = true;
        self.flight_real |= !matches!(c, crate::copystat::C::RangeClean);
        crate::copystat::add(c, 1);
    }

    /// Overlap hazards: a range the submission's shaders write, of a copy
    /// the last one wrote (its snapshot or marks would meet that one's).
    #[cfg(feature = "diagnostics")]
    fn prep_write(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, marked: bool) {
        self.flight_marked |= marked;
        if self.flight_prev_copies.binary_search(&(std::sync::Arc::as_ptr(shadow) as usize)).is_ok() {
            crate::copystat::add(if marked { crate::copystat::C::PrepMarkedWritten } else { crate::copystat::C::PrepSnapshotWritten }, 1);
            self.flight_hazard = true;
        }
    }

    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let address = shadow.canon(address as u64) as usize;
        #[cfg(feature = "diagnostics")]
        let same = self.flight_prev_copies.binary_search(&(std::sync::Arc::as_ptr(shadow) as usize)).is_ok();
        #[cfg(feature = "diagnostics")]
        {
            self.flight_shared |= same;
            self.flight_used.push(std::sync::Arc::as_ptr(shadow) as usize);
        }
        // Nothing to copy: a world submission asks for every copy its heaps'
        // views and page tables reach (~9,000), a few of them written.
        if shadow.clean() {
            #[cfg(feature = "diagnostics")]
            {
                self.touch(address, len, crate::copystat::C::RangeClean);
                crate::copystat::add(crate::copystat::C::Skipped, 1);
                crate::copystat::add(crate::copystat::C::SkippedBytes, len as u64);
            }
            return;
        }
        #[cfg(feature = "diagnostics")]
        {
            self.touch(address, len, if same { crate::copystat::C::RangeSame } else { crate::copystat::C::RangeAlias });
            if self.flight_prev_used.binary_search(&(std::sync::Arc::as_ptr(shadow) as usize)).is_ok() {
                crate::copystat::add(crate::copystat::C::PrepDirtyInUse, 1);
                self.flight_hazard = true;
            }
        }
        let (key, end) = (std::sync::Arc::as_ptr(shadow) as usize, address + len);
        if self.current.get(&key).is_some_and(|&(_, a, b)| a <= address && end <= b) {
            return;
        }
        match self.pending.get(&key) {
            Some(&i) => {
                let (_, from, to) = &mut self.refreshes[i];
                (*from, *to) = ((*from).min(address), (*to).max(end));
            }
            None => {
                self.pending.insert(key, self.refreshes.len());
                self.refreshes.push((shadow.clone(), address, end));
            }
        }
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.current.clear();
    }

    /// A range the submission's shaders write (`Step::GpuWrite`), in
    /// recorded order: a deferred copy touching it sees its bytes only in
    /// the copy-back.
    fn gpu_write(&mut self, at: usize, len: usize) {
        self.gpu_writes.push((at, at + len));
    }

    /// Whether `len` title bytes at `at` touch a range this submission's
    /// shaders write: the bytes are not in title memory until the
    /// copy-back after the lists run.
    // shortcut: linear over the submission's writes (thousands in the
    // world); sort or merge them if copy steps show up in the timing.
    fn gpu_written(&self, at: usize, len: usize) -> bool {
        self.gpu_writes.iter().any(|&(a, b)| a < at + len && at < b)
    }

    /// Does the refreshes; returns the bytes copied. Their copies of pages
    /// no shader writes are made together at the end, on several threads
    /// ([`crate::copy_pool`]); the time per range counts only finding them.
    fn flush(&mut self) -> usize {
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::FlushUs);
        self.pending.clear();
        let started = std::time::Instant::now();
        let (total, pieces) = crate::copy_pool::queued(|| self.update_all());
        crate::copy_pool::run(pieces);
        PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        total
    }

    /// [`Batch::flush`]'s updates, one per copy.
    fn update_all(&mut self) -> usize {
        let mut total = 0;
        for (shadow, from, to) in self.refreshes.drain(..) {
            let started = std::time::Instant::now();
            let copied = shadow.update(from, to - from);
            #[cfg(feature = "diagnostics")]
            crate::copystat::range(from, to - from, copied, started.elapsed().as_micros() as u64, 0);
            #[cfg(not(feature = "diagnostics"))]
            let _ = started;
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
            total += copied;
            // Kept: the union with what was current if they meet and it
            // holds no whole page neither part held (pages are taken whole:
            // a page two parts share stays changed), else the larger.
            let key = std::sync::Arc::as_ptr(&shadow) as usize;
            let (_, a, b) = self.current.entry(key).or_insert((shadow, from, to));
            let page = crate::watch::PAGE;
            let holds = |x: usize, y: usize, p: usize| x <= p * page && (p + 1) * page <= y;
            let (lo, hi) = (from.min(*a), to.max(*b));
            // Only a page with an end of a part can be whole in neither.
            if from <= *b
                && *a <= to
                && [from, to, *a, *b]
                    .iter()
                    .map(|x| x / page)
                    .all(|p| !holds(lo, hi, p) || holds(from, to, p) || holds(*a, *b, p))
            {
                (*a, *b) = (lo, hi);
            } else if to - from > *b - *a {
                (*a, *b) = (from, to);
            }
        }
        total
    }
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped takes nothing from its
/// first unmapped page on.
///
/// Both ways of reading the source report an unmapped page instead of
/// faulting: in user mode into host memory ([`crate::title_copy`]), through
/// the kernel (`ReadProcessMemory`) into title memory, which also reports an
/// unmapped page of the destination it writes. Asking `VirtualQuery` first
/// cost about 2 ms per copy in the title's big pools (it walks the page
/// tables to the region's end), which made copies run at 100-200 MB/s; a
/// title destination is asked only once its copy has failed (asked before
/// each, the world's 80 copies per frame cost 60 ms).
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    // The kernel writes the destination: it must not be guarded meanwhile.
    let _writing = title_dst.then(|| crate::watch::writing(dst as usize, len));
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    // A page that cannot be read fails either way and is found below.
    // `_copied`: the destination's bytes written (a prefix).
    let (whole, _copied) = if title_dst {
        let copied = read_copied(src, dst, len);
        (copied == len, copied)
    } else {
        // SAFETY: the destination is host memory of `len` bytes.
        (unsafe { crate::title_copy::copy(dst, src, len) }, 0)
    };
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        #[cfg(feature = "diagnostics")]
        if title_dst {
            crate::copystat::ours_wrote(dst as usize, len);
        }
        return;
    }
    // The copy stopped at an unmapped page. One of a title destination ends
    // it: the retries below write the destination themselves (zeros for
    // what cannot be read), and would fault there.
    if title_dst && !crate::texture::committed(dst, len) {
        #[cfg(feature = "diagnostics")]
        if _copied != 0 {
            crate::copystat::ours_wrote(dst as usize, _copied);
        }
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    #[cfg(feature = "diagnostics")]
    let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::FallbackUs);
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
    #[cfg(feature = "diagnostics")]
    if title_dst {
        crate::copystat::ours_wrote(dst as usize, len);
    }
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` in the kernel: false if a source
/// page is not readable or a destination page not writable (the bytes
/// before that page may have been copied).
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    read_copied(src, dst, len) == len
}

/// [`read`], returning the bytes copied: all `len`, or those before the
/// first page that could not be read or written.
fn read_copied(src: *const u8, dst: *mut u8, len: usize) -> usize {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the destination is `len` bytes the caller may write, host
    // memory or title memory; the kernel checks both ranges, and a page it
    // cannot read or write fails the call instead of faulting.
    let ok = unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 };
    if ok { len } else { done.min(len) }
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
///
/// Through [`read`]: the kernel refuses an unmapped destination page as it
/// does a source page. `WriteProcessMemory` makes the same copy, but asks
/// for the destination's region first, at `VirtualQuery`'s cost (see
/// [`copy_mapped`]): 0.35 ms for each piece of the world's copy-back, 39 ms
/// per frame.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    write_mapped_each(dst, src, len, |_, _| {});
}

/// [`write_mapped`], calling `wrote(at, bytes)` for the destination's bytes
/// it wrote (some more than once).
pub(crate) fn write_mapped_each(dst: *mut u8, src: *const u8, len: usize, mut wrote: impl FnMut(usize, usize)) {
    let mut write = |at: usize, n: usize| {
        let copied = read_copied(src.wrapping_add(at - dst as usize), at as *mut u8, n);
        if copied != 0 {
            wrote(at, copied);
        }
        copied == n
    };
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

#[cfg(test)]
mod title_copies {
    use super::{copy_mapped, write_mapped};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
        VirtualAlloc, VirtualFree, VirtualProtect,
    };

    const BLOCK: usize = 64 << 10;

    /// Three reserved blocks, the first and the last committed and filled
    /// with `fill`: title memory with an unmapped block in it.
    fn holed(fill: u8) -> *mut u8 {
        // SAFETY: a fresh reservation; the committed blocks are ours to fill.
        unsafe {
            let base =
                VirtualAlloc(std::ptr::null(), 3 * BLOCK, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            for block in [0, 2] {
                let at = base.add(block * BLOCK);
                assert!(!VirtualAlloc(at.cast(), BLOCK, MEM_COMMIT, PAGE_READWRITE).is_null());
                at.write_bytes(fill, BLOCK);
            }
            base
        }
    }

    fn block<'a>(base: *const u8, n: usize) -> &'a [u8] {
        // SAFETY: tests ask for committed blocks only.
        unsafe { std::slice::from_raw_parts(base.add(n * BLOCK), BLOCK) }
    }

    fn release(base: *mut u8) {
        // SAFETY: a reservation of `holed`.
        assert_ne!(unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) }, 0);
    }

    fn pattern() -> Vec<u8> {
        (0..3 * BLOCK).map(|i| (i % 251) as u8 + 1).collect()
    }

    #[test]
    fn title_copies_reach_mapped_destinations_and_stop_at_unmapped_pages() {
        let from = pattern();
        // All mapped: copied, through a destination that was never asked.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), BLOCK, true);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        release(to);
        // An unmapped block in the destination: nothing faults, and nothing
        // arrives from that block on.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), 3 * BLOCK, true);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
        // An unmapped block in the source reads as zeros.
        let holes = holed(7);
        let mut to = vec![0xeeu8; 3 * BLOCK];
        copy_mapped(to.as_mut_ptr(), holes, 3 * BLOCK, true);
        assert!(to[..BLOCK].iter().all(|&b| b == 7));
        assert!(to[BLOCK..2 * BLOCK].iter().all(|&b| b == 0));
        assert!(to[2 * BLOCK..].iter().all(|&b| b == 7));
        release(holes);
    }

    #[test]
    fn copy_back_writes_mapped_pages_only() {
        let from = pattern();
        let to = holed(0xee);
        write_mapped(to, from.as_ptr(), 3 * BLOCK);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        assert_eq!(block(to, 2), &from[2 * BLOCK..]);
        release(to);
        // Neither aligned nor whole: the bytes around the range stay.
        let to = holed(0xee);
        // SAFETY: inside the first block.
        write_mapped(unsafe { to.add(100) }, from.as_ptr(), BLOCK - 300);
        let first = block(to, 0);
        assert_eq!(first[100..BLOCK - 200], from[..BLOCK - 300]);
        let around = first[..100].iter().chain(&first[BLOCK - 200..]);
        assert!(around.into_iter().all(|&b| b == 0xee));
        // A page that is mapped but not writable is not written either
        // (guarded title blocks are opened before they are copied back).
        let mut old = 0;
        // SAFETY: a committed block of ours.
        let protected =
            unsafe { VirtualProtect(to.add(2 * BLOCK).cast(), BLOCK, PAGE_READONLY, &mut old) };
        assert_ne!(protected, 0);
        // SAFETY: the third block.
        write_mapped(unsafe { to.add(2 * BLOCK) }, from.as_ptr(), BLOCK);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
    }
}

impl Host {
    /// A buffer in the upload heap, mapped for writing.
    /// A GPU-only buffer of `size` bytes in the copy-destination state.
    pub(crate) fn default_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
        };
        let buffer = made.ok().and(buffer)?;
        crate::memsys::made(crate::memsys::Class::Vram, size);
        Some(buffer)
    }

    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        self.mapped_buffer(size, false)
    }

    /// A mapped buffer of `size` bytes: upload memory, or (`readback`)
    /// readback memory in the copy-destination state.
    fn mapped_buffer(&self, size: u64, readback: bool) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: if readback { D3D12_HEAP_TYPE_READBACK } else { D3D12_HEAP_TYPE_UPLOAD },
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    if readback { D3D12_RESOURCE_STATE_COPY_DEST } else { D3D12_RESOURCE_STATE_GENERIC_READ },
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => {
                use crate::memsys::Class;
                crate::memsys::made(if readback { Class::Readback } else { Class::Upload }, size);
                Some((buffer?, data.cast()))
            }
            Err(e) => {
                tracing::warn!(error = %e, size, readback, "no mapped buffer");
                None
            }
        }
    }

    /// The last host submission the GPU has finished (every one, once the
    /// device is removed).
    pub(crate) fn completed(&self) -> u64 {
        // SAFETY: a plain query.
        unsafe { self.fence.GetCompletedValue() }
    }

    /// A buffer in video memory shaders may write (unordered access), in
    /// the common state.
    pub(crate) fn uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        self.buffer_in(
            D3D12_HEAP_TYPE_DEFAULT,
            size,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )
    }

    /// A buffer in the read-back heap, mapped for reading (what the GPU
    /// copied there, once the submission that copies it has run).
    pub(crate) fn readback_buffer(&self, size: u64) -> Option<(ID3D12Resource, *const u8)> {
        let buffer = self.buffer_in(
            D3D12_HEAP_TYPE_READBACK,
            size,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer of ours, mapped whole.
        match unsafe { buffer.Map(0, None, Some(&mut data)) } {
            Ok(()) => Some((buffer, data.cast_const().cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no read-back buffer");
                None
            }
        }
    }

    fn buffer_in(
        &self,
        heap: D3D12_HEAP_TYPE,
        size: u64,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: heap,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => {
                use crate::memsys::Class;
                let class = if heap.Type == D3D12_HEAP_TYPE_READBACK { Class::Readback } else { Class::Vram };
                crate::memsys::made(class, size);
                buffer
            }
            Err(e) => {
                tracing::warn!(error = %e, size, heap = heap.Type.0, "no buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;
const MAX_PLANES: u32 = 8;

/// Built-in fullscreen-triangle compositor for multi-plane `PresentX` calls
/// (e.g. Series S presenting a 1080p scene plane with a 4K UI overlay plane).
struct Compositor {
    format: DXGI_FORMAT,
    root: ID3D12RootSignature,
    opaque: ID3D12PipelineState,
    overlay: ID3D12PipelineState,
    rtv_heap: ID3D12DescriptorHeap,
    srv_heap: ID3D12DescriptorHeap,
    srv_stride: u32,
}

impl Compositor {
    fn new(device: &ID3D12Device, format: DXGI_FORMAT) -> windows::core::Result<Self> {
        const SOURCE: &str = r#"
            Texture2D tex : register(t0);
            SamplerState smp : register(s0);
            void vs(uint id : SV_VertexID, out float4 pos : SV_Position, out float2 uv : TEXCOORD) {
                uv = float2((id << 1) & 2, id & 2);
                pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
            }
            float4 ps(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target {
                return tex.Sample(smp, uv);
            }
        "#;
        let compile = |entry: PCSTR, target: PCSTR| -> windows::core::Result<ID3DBlob> {
            let mut code: Option<ID3DBlob> = None;
            let mut errors: Option<ID3DBlob> = None;
            // SAFETY: static source and entry/target strings; out pointers ours.
            let res = unsafe {
                D3DCompile(
                    SOURCE.as_ptr().cast(),
                    SOURCE.len(),
                    s!("present_composite"),
                    None,
                    None,
                    entry,
                    target,
                    0,
                    0,
                    &mut code,
                    Some(&mut errors),
                )
            };
            if let Err(e) = res {
                if let Some(errors) = errors {
                    // SAFETY: the blob's own bytes.
                    let text = unsafe {
                        std::slice::from_raw_parts(
                            errors.GetBufferPointer().cast::<u8>(),
                            errors.GetBufferSize(),
                        )
                    };
                    tracing::warn!("present compositor: {}", String::from_utf8_lossy(text));
                }
                return Err(e);
            }
            code.ok_or_else(windows::core::Error::empty)
        };
        let vs = compile(s!("vs"), s!("vs_5_1"))?;
        let ps = compile(s!("ps"), s!("ps_5_1"))?;
        // SAFETY: compiled shader blobs live through PSO creation.
        unsafe {
            let vs_bytes =
                std::slice::from_raw_parts(vs.GetBufferPointer().cast::<u8>(), vs.GetBufferSize());
            let ps_bytes =
                std::slice::from_raw_parts(ps.GetBufferPointer().cast::<u8>(), ps.GetBufferSize());
            // Wine's D3DCompile does not embed a root signature in the shader blobs.
            let range = D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: 1,
                OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
                ..Default::default()
            };
            let parameter = D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: &range,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            };
            let sampler = D3D12_STATIC_SAMPLER_DESC {
                Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                MaxAnisotropy: 16,
                ComparisonFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
                MaxLOD: f32::MAX,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
                ..Default::default()
            };
            let desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: 1,
                pParameters: &parameter,
                NumStaticSamplers: 1,
                pStaticSamplers: &sampler,
                ..Default::default()
            };
            let mut signature = None;
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut signature, None)?;
            let signature = signature.ok_or_else(windows::core::Error::empty)?;
            let signature = std::slice::from_raw_parts(
                signature.GetBufferPointer().cast::<u8>(),
                signature.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, signature)?;
            let make_pso = |blend: bool| -> windows::core::Result<ID3D12PipelineState> {
                let mut rt_blend = D3D12_RENDER_TARGET_BLEND_DESC {
                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    ..Default::default()
                };
                if blend {
                    rt_blend.BlendEnable = true.into();
                    rt_blend.SrcBlend = D3D12_BLEND_ONE;
                    rt_blend.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
                    rt_blend.BlendOp = D3D12_BLEND_OP_ADD;
                    rt_blend.SrcBlendAlpha = D3D12_BLEND_ONE;
                    rt_blend.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
                    rt_blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
                }
                let mut blend_desc = D3D12_BLEND_DESC::default();
                blend_desc.RenderTarget[0] = rt_blend;
                let mut rtv_formats = [DXGI_FORMAT_UNKNOWN; 8];
                rtv_formats[0] = format;
                let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
                    VS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: vs_bytes.as_ptr().cast(),
                        BytecodeLength: vs_bytes.len(),
                    },
                    PS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: ps_bytes.as_ptr().cast(),
                        BytecodeLength: ps_bytes.len(),
                    },
                    BlendState: blend_desc,
                    SampleMask: u32::MAX,
                    RasterizerState: D3D12_RASTERIZER_DESC {
                        FillMode: D3D12_FILL_MODE_SOLID,
                        CullMode: D3D12_CULL_MODE_NONE,
                        DepthClipEnable: false.into(),
                        ..Default::default()
                    },
                    PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                    NumRenderTargets: 1,
                    RTVFormats: rtv_formats,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    ..Default::default()
                };
                let pso = device.CreateGraphicsPipelineState(&desc);
                let _ = std::mem::ManuallyDrop::into_inner(desc.pRootSignature);
                pso
            };
            let opaque = make_pso(false)?;
            let overlay = make_pso(true)?;
            let rtv_heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })?;
            let srv_heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: MAX_PLANES,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
            let srv_stride =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            Ok(Compositor {
                format,
                root,
                opaque,
                overlay,
                rtv_heap,
                srv_heap,
                srv_stride,
            })
        }
    }
}

#[derive(Default)]
struct Presentation {
    presenter: Option<Presenter>,
    retry: SwapChainRetry,
}

#[derive(Default)]
struct SwapChainRetry {
    next_attempt: Option<Instant>,
}

impl SwapChainRetry {
    fn ready(&self, now: Instant) -> bool {
        self.next_attempt.is_none_or(|next| now >= next)
    }

    /// A desktop can become presentable again. Keep retrying while limiting
    /// the DXGI calls and warnings during the outage.
    fn attempt<T, E>(
        &mut self,
        now: Instant,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Option<Result<T, E>> {
        if !self.ready(now) {
            return None;
        }
        let result = operation();
        self.next_attempt = result.is_err().then(|| now + Duration::from_secs(1));
        Some(result)
    }
}

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    hwnd: HWND,
    meter: Meter,
    compositor: Option<Compositor>,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    report_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            report_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        self.frames += 1;
        crate::memory::log_frame();
        #[cfg(feature = "diagnostics")]
        crate::watch::frame_done();
        self.report_frames += 1;
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            let report_frames = std::mem::take(&mut self.report_frames) as f64;
            let dropped = crate::draw::take_dropped() as f64 / report_frames;
            let indirect = crate::list::take_indirect() as f64 / report_frames;
            let root_views = crate::draw::take_root_view_slots() as f64 / report_frames;
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect; {root_views:.0} root-view slots/frame",
                making = crate::pipeline::making()
            );
            let (textures, bytes, used, shared, saved, replaced) = crate::resource::texture_counts();
            let (heaps, heap_used, heap_free) = crate::placement::stats();
            tracing::info!(
                "host textures: {textures} ({:.0} MB, {:.0} MB of them used by lists), title textures sharing one {shared} (saving {:.0} MB), replacing one {replaced}; \
                 heaps of textures {heaps} ({:.0} MB used, {:.0} MB free); \
                 copies between textures of one memory {:.1}/frame, depth twin copies {:.1}/frame",
                bytes as f64 / 1e6,
                used as f64 / 1e6,
                saved as f64 / 1e6,
                heap_used as f64 / 1e6,
                heap_free as f64 / 1e6,
                crate::texture::ALIAS_COPIES.swap(0, Ordering::Relaxed) as f64 / report_frames,
                crate::twin::COPIES.swap(0, Ordering::Relaxed) as f64 / report_frames,
            );
            #[cfg(feature = "diagnostics")]
            {
                tracing::info!(
                    "pipeline syncs per frame, flags/control calls (barriers): {}",
                    crate::list::take_syncs(report_frames)
                );
                tracing::info!(
                    "host copies kept over changed title memory {:.1}/frame, host-written marks cleared {} in all",
                    crate::texture::HOST_KEPT_STALE.swap(0, Ordering::Relaxed) as f64 / report_frames,
                    crate::texture::HOST_MARKS_CLEARED.swap(0, Ordering::Relaxed)
                );
                if let Some([(local, budget), (system, system_budget)]) = video_memory() {
                    let (commit, working_set) = process_memory();
                    tracing::info!(
                        "video memory: {local:.0} MB of a {budget:.0} MB budget on the card, \
                         {system:.0} MB of {system_budget:.0} MB in system memory; process commit {:.0} MB, \
                         working set {:.0} MB; Rust heap {:.0} MB; placed textures made {:.0} MB; list upload memory {} blocks {:.0} MB",
                        commit as f64 / 1e6,
                        working_set as f64 / 1e6,
                        crate::copystat::HEAP_BYTES.load(Ordering::Relaxed) as f64 / 1e6,
                        crate::resource::PLACED_BYTES.load(Ordering::Relaxed) as f64 / 1e6,
                        crate::list::UPLOADS[0].load(Ordering::Relaxed),
                        crate::list::UPLOADS[1].load(Ordering::Relaxed) as f64 / 1e6,
                    );
                }
                let [gpu, title] = crate::resource::texture_census().map(|r| r.map(|b| b as f64 / 1e6));
                tracing::info!(
                    "host texture MB (used by lists, viewed by heap slots only, neither): GPU-written {:.0} {:.0} {:.0}, \
                     from title memory {:.0} {:.0} {:.0}",
                    gpu[0], gpu[1], gpu[2], title[0], title[1], title[2]
                );
                // Window caption updates leave the report counters intact.
                let drops = crate::pipeline::take_drops();
                let [waits, waited_us, late] = crate::pipeline::take_compute_waits();
                if waits > 0 {
                    tracing::info!(
                        "compute pipelines waited for: {waits}, {} ms in all, {late} not made in time all the same",
                        waited_us / 1000
                    );
                }
                crate::frame::log();
                log_gpu_times(report_frames);
                crate::gpumeasure::log(report_frames);
                crate::copystat::log(report_frames);
                crate::bindless::log(report_frames);
                crate::shadow::log_views(report_frames);
                crate::query::counts::log(report_frames);
                let [n, bytes] = [0, 1].map(|i| crate::list::READ_BACKS[i].swap(0, Ordering::Relaxed));
                if n > 0 {
                    tracing::info!(
                        "texture read backs: {:.1}/frame, {:.1} KB/frame",
                        n as f64 / report_frames,
                        bytes as f64 / report_frames / 1e3
                    );
                }
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / report_frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows one or more `planes` (each a live host texture and its current
    /// host state, kept in that state), compositing planes 1.. on top of
    /// plane 0 when multiple planes are presented, as frame `frame` (saved if
    /// captures are on). `None` if nothing was shown; else, in diagnostics
    /// builds, whether the frame was black.
    pub(crate) fn present(&self, planes: &[(&ID3D12Resource, u32)], frame: u64) -> Option<bool> {
        self.present_with_compositor(
            planes,
            frame,
            Compositor::new,
            #[cfg(all(test, feature = "diagnostics"))]
            |_, _| {},
        )
    }

    fn present_with_compositor(
        &self,
        planes: &[(&ID3D12Resource, u32)],
        frame: u64,
        create_compositor: impl FnOnce(&ID3D12Device, DXGI_FORMAT) -> windows::core::Result<Compositor>,
        #[cfg(all(test, feature = "diagnostics"))] inspect: impl FnOnce(
            &ID3D12Resource,
            &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        ),
    ) -> Option<bool> {
        let &(primary, _) = planes.first()?;
        // SAFETY: live resources.
        let primary_desc = unsafe { primary.GetDesc() };
        let Some(format) = swap_format(primary_desc.Format) else {
            tracing::warn!(format = primary_desc.Format.0, "cannot present this format");
            return None;
        };
        let mut presentation = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        let Presentation { presenter, retry } = &mut *presentation;
        if presenter.is_none() && !retry.ready(Instant::now()) {
            return None;
        }
        let cached_format = presenter
            .as_ref()
            .and_then(|p| p.compositor.as_ref())
            .map(|c| c.format);
        let mut new_compositor = None;
        let planes = if planes.len() > 1 && cached_format != Some(format) {
            match create_compositor(&self.device, format) {
                Ok(c) => {
                    new_compositor = Some(c);
                    planes
                }
                Err(e) => {
                    tracing::warn!(error = %e, "presenting the primary plane after compositor initialization failed");
                    &planes[..1]
                }
            }
        } else {
            planes
        };
        let (mut width, mut height) = (primary_desc.Width as u32, primary_desc.Height);
        for &(tex, _) in &planes[1..] {
            // SAFETY: live resource.
            let d = unsafe { tex.GetDesc() };
            width = width.max(d.Width as u32);
            height = height.max(d.Height);
        }
        if presenter.is_none() {
            *presenter = match retry
                .attempt(Instant::now(), || self.presenter(width, height, format))?
            {
                Ok(p) => Some(p),
                Err((step, e)) => {
                    tracing::warn!(error = %e, step, width, height, format = format.0, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if let Some(c) = new_compositor {
            p.compositor = Some(c);
        }
        if (p.width, p.height, p.format) != (width, height, format) {
            let resized = retry.attempt(Instant::now(), || {
                // SAFETY: no back buffer is held; every submission was waited for.
                unsafe {
                    p.swap_chain.ResizeBuffers(
                        BUFFERS,
                        width,
                        height,
                        format,
                        DXGI_SWAP_CHAIN_FLAG(0),
                    )
                }
            })?;
            match resized {
                Ok(()) => (p.width, p.height, p.format) = (width, height, format),
                Err(e) => {
                    tracing::warn!(error = %e, width, height, format = format.0, "swap chain not resized");
                    return None;
                }
            }
        }
        // SAFETY: our swap chain, list and live source textures.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            p.copy.reset();
            let list = &p.copy.list;
            let (capture, probe) = if planes.len() == 1 {
                let (texture, state) = planes[0];
                let state = D3D12_RESOURCE_STATES(state as i32);
                let mut before = Vec::with_capacity(2);
                if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    before.push(transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE));
                }
                before.push(transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ));
                list.ResourceBarrier(&before);
                list.CopyResource(&back, texture);
                #[cfg(all(test, feature = "diagnostics"))]
                let capture = {
                    // Inspect what reached the swap chain, before Present.
                    list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    let readback = crate::capture::read_back(&self.device, list, &back);
                    list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    )]);
                    readback
                };
                #[cfg(all(not(test), feature = "diagnostics"))]
                let capture = crate::capture::wanted(frame)
                    .then(|| crate::capture::read_back(&self.device, list, texture))
                    .flatten();
                #[cfg(feature = "diagnostics")]
                let probe = crate::capture::probe(&self.device, list, texture);
                #[cfg(not(feature = "diagnostics"))]
                let (capture, probe) = {
                    let _ = frame;
                    (None::<()>, None::<()>)
                };
                let mut after = Vec::with_capacity(2);
                if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    after.push(transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state));
                }
                after.push(transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ));
                list.ResourceBarrier(&after);
                (capture, probe)
            } else {
                let comp = p.compositor.as_ref().expect("compositor was prepared for multiple planes");
                let rtv = comp.rtv_heap.GetCPUDescriptorHandleForHeapStart();
                self.device.CreateRenderTargetView(
                    &back,
                    Some(&D3D12_RENDER_TARGET_VIEW_DESC {
                        Format: format,
                        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                        Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_RTV {
                                MipSlice: 0,
                                PlaneSlice: 0,
                            },
                        },
                    }),
                    rtv,
                );
                let srv_cpu_base = comp.srv_heap.GetCPUDescriptorHandleForHeapStart();
                let srv_gpu_base = comp.srv_heap.GetGPUDescriptorHandleForHeapStart();
                let count = planes.len().min(MAX_PLANES as usize);
                let mut before = Vec::with_capacity(count + 1);
                let mut after = Vec::with_capacity(count);
                for (i, &(tex, state)) in planes[..count].iter().enumerate() {
                    let d = tex.GetDesc();
                    let srv_fmt = swap_format(d.Format).unwrap_or(d.Format);
                    let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                        ptr: srv_cpu_base.ptr + i * comp.srv_stride as usize,
                    };
                    self.device.CreateShaderResourceView(
                        tex,
                        Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                            Format: srv_fmt,
                            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                                Texture2D: D3D12_TEX2D_SRV {
                                    MostDetailedMip: 0,
                                    MipLevels: 1,
                                    PlaneSlice: 0,
                                    ResourceMinLODClamp: 0.0,
                                },
                            },
                        }),
                        handle,
                    );
                    let state = D3D12_RESOURCE_STATES(state as i32);
                    if state != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                        && planes[..i]
                            .iter()
                            .all(|(earlier, _)| earlier.as_raw() != tex.as_raw())
                    {
                        before.push(transition(
                            tex,
                            state,
                            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                        ));
                        after.push(transition(
                            tex,
                            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                            state,
                        ));
                    }
                }
                before.push(transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                ));
                list.ResourceBarrier(&before);
                list.SetDescriptorHeaps(&[Some(comp.srv_heap.clone())]);
                list.SetGraphicsRootSignature(&comp.root);
                list.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: width as f32,
                    Height: height as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                list.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                }]);
                list.OMSetRenderTargets(1, Some(&rtv), false, None);
                list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                for i in 0..count {
                    list.SetPipelineState(if i == 0 { &comp.opaque } else { &comp.overlay });
                    list.SetGraphicsRootDescriptorTable(
                        0,
                        D3D12_GPU_DESCRIPTOR_HANDLE {
                            ptr: srv_gpu_base.ptr + (i as u64) * u64::from(comp.srv_stride),
                        },
                    );
                    list.DrawInstanced(3, 1, 0, 0);
                }
                #[cfg(feature = "diagnostics")]
                let (capture, probe) = {
                    list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    let capture = (cfg!(test) || crate::capture::wanted(frame))
                        .then(|| crate::capture::read_back(&self.device, list, &back))
                        .flatten();
                    let probe = crate::capture::probe(&self.device, list, &back);
                    list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_PRESENT,
                    )]);
                    (capture, probe)
                };
                #[cfg(not(feature = "diagnostics"))]
                let (capture, probe) = {
                    let _ = frame;
                    list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        D3D12_RESOURCE_STATE_PRESENT,
                    )]);
                    (None::<()>, None::<()>)
                };
                if !after.is_empty() {
                    list.ResourceBarrier(&after);
                }
                (capture, probe)
            };
            p.copy.close();
            self.execute(&[&p.copy]);
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                // Tests inspect the completed copy before flip-discard Present
                // can invalidate the back buffer, with the presenter still locked.
                #[cfg(test)]
                inspect(&buffer, &footprint);
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            {
                let _ = (capture, probe);
                Some(false)
            }
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> Result<Presenter, (&'static str, windows::core::Error)> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height).map_err(|e| ("window", e))?,
        };
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))
                .map_err(|e| ("DXGI factory", e))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)
                .and_then(|chain| chain.cast::<IDXGISwapChain3>())
                .map_err(|e| ("swap chain", e))?;
            let copy = self.new_list().map_err(|e| ("copy list", e))?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                hwnd,
                meter: Meter::new(),
                compositor: None,
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    let legal = legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state);
    #[cfg(feature = "diagnostics")]
    if legal & 0x141c != 0 && state & !legal & 0xc0 != 0 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(state, legal, "shader read states dropped beside a write state"));
    }
    legal
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    // Only texture states the host takes, and a write state only alone (the
    // lowest one): D3D12 rejects it combined with any other.
    let s = FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state & 0x100_3cfc, |s, &(_, _, bits)| s & !bits);
    let write = s & 0x141c;
    if write != 0 { write & write.wrapping_neg() } else { s }
}

/// Bytes of host copies in GPU upload heaps now ([`Host::gpu_upload_buffer`]):
/// charged when one is made, given back when D3D12 destroys it
/// ([`give_back_upload`]).
static UPLOAD_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Called by D3D12 when a host copy in a GPU upload heap is destroyed (its
/// last reference gone: the copy's, lists', and descriptor tables' clones
/// of the resource): its `size` (the callback's data) leaves
/// [`UPLOAD_BYTES`]. Counted for the whole run, the budget ran out after
/// enough buffers came and went and every later copy stayed in system
/// memory (64 copies of 32 MiB on an RTX 3050 Laptop, #2120).
unsafe extern "system" fn give_back_upload(size: *mut std::ffi::c_void) {
    UPLOAD_BYTES.fetch_sub(size as u64, Ordering::Relaxed);
}

/// Live [`Host::table_buffer`]s in GPU upload heaps, and their bytes.
static TABLE_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TABLE_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Called by D3D12 when a table in a GPU upload heap is destroyed.
unsafe extern "system" fn give_back_table(size: *mut std::ffi::c_void) {
    TABLE_COUNT.fetch_sub(1, Ordering::Relaxed);
    TABLE_BYTES.fetch_sub(size as u64, Ordering::Relaxed);
}

/// Takes `bytes` of the budget host copies in GPU upload heaps share
/// ([`UPLOAD_BUDGET`]), for a heap buffers are placed in
/// (`crate::memsys::heaps`); false, with nothing taken, past it.
pub(crate) fn charge_upload(bytes: u64) -> bool {
    if UPLOAD_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes > UPLOAD_BUDGET {
        UPLOAD_BYTES.fetch_sub(bytes, Ordering::Relaxed);
        return false;
    }
    true
}

/// Gives back what [`charge_upload`] took (the heap was not made).
pub(crate) fn uncharge_upload(bytes: u64) {
    UPLOAD_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

/// [`UPLOAD_BYTES`], for tests.
#[cfg(test)]
pub(crate) fn upload_bytes() -> u64 {
    UPLOAD_BYTES.load(Ordering::Relaxed)
}

/// Video memory host copies may take in GPU upload heaps; the rest stay in
/// system memory.
/// shortcut: a fixed amount; ask DXGI for the adapter's budget if 8 GB
/// cards run short.
const UPLOAD_BUDGET: u64 = 2 << 30;

impl Host {
    /// Diagnostics, `VICTORY_COPIES_IN_VIDEO_MEMORY`: [`Host::shared_buffer`]s
    /// (host copies of title memory among them) in video memory the CPU maps
    /// (a GPU upload heap, where the host has one), to measure how much of
    /// the GPU's time is shaders reading the copies over the bus. CPU reads
    /// of them (snapshots, copy-backs) become slow: for that measurement only.
    #[cfg(feature = "diagnostics")]
    fn copies_in_video_memory(&self) -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            if !crate::switch::on("VICTORY_COPIES_IN_VIDEO_MEMORY") {
                return false;
            }
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
            // SAFETY: the query writes its own structure of the given size.
            let ok = unsafe {
                self.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS16,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
            }
            .is_ok()
                && options.GPUUploadHeapSupported.as_bool();
            tracing::warn!(on = ok, "host copies of title memory in video memory (diagnostics)");
            ok
        })
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        self.shared_mapped_buffer(size, false)
    }

    /// A buffer in video memory that the CPU maps through the PCI BAR (a
    /// GPU upload heap, `D3D12_OPTIONS16`): shaders read it as fast as any
    /// video memory (on an RTX 5070 Ti ~720 GB/s against ~23 GB/s from
    /// system memory), the CPU writes it about as fast as system memory,
    /// and reading it from the CPU is very slow (~50 MB/s). For host copies
    /// of title memory that shaders only read. None when the host has no
    /// such heaps (no resizable BAR, older runtimes), `VICTORY_GPU_UPLOAD`
    /// is off, or the copies there would pass [`UPLOAD_BUDGET`].
    pub(crate) fn gpu_upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        if !self.gpu_upload() {
            return None;
        }
        let size = size.max(1);
        // A small one goes into a heap made ahead (`VICTORY_MEMSYS`), which
        // the budget counts whole: not charged again here.
        if let Some(made) = crate::memsys::heaps::buffer(&self.device, true, size) {
            return Some(made);
        }
        if UPLOAD_BYTES.fetch_add(size, Ordering::Relaxed) + size > UPLOAD_BUDGET {
            UPLOAD_BYTES.fetch_sub(size, Ordering::Relaxed);
            static FULL: std::sync::Once = std::sync::Once::new();
            FULL.call_once(|| {
                tracing::info!("host copies past the GPU upload budget go to system memory")
            });
            return None;
        }
        let Some((buffer, data)) = self.shared_mapped_buffer(size, true) else {
            UPLOAD_BYTES.fetch_sub(size, Ordering::Relaxed);
            return None;
        };
        // The charge goes back when the buffer is destroyed, not when the
        // copy drops: lists and descriptor tables may hold it longer.
        // SAFETY: a live buffer; the callback's data is the size alone.
        let watched = unsafe {
            buffer
                .cast::<windows::Win32::Graphics::Direct3D::ID3DDestructionNotifier>()
                .and_then(|n| {
                    n.RegisterDestructionCallback(Some(give_back_upload), size as *const _)
                })
        };
        if watched.is_err() {
            // Not given back when destroyed, it would count for good.
            drop(buffer);
            UPLOAD_BYTES.fetch_sub(size, Ordering::Relaxed);
            static UNWATCHED: std::sync::Once = std::sync::Once::new();
            UNWATCHED.call_once(|| {
                tracing::info!(
                    "no destruction callback on this host: host copies stay in system memory"
                )
            });
            return None;
        }
        Some((buffer, data))
    }

    /// Whether host copies that shaders only read go to GPU upload heaps:
    /// `VICTORY_GPU_UPLOAD=1` (a prototype, off by default) and a host that
    /// has them. Asked once.
    pub(crate) fn gpu_upload(&self) -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            if !crate::switch::on("VICTORY_GPU_UPLOAD") {
                return false;
            }
            let on = self.gpu_upload_heaps();
            tracing::info!(
                on,
                "host copies that shaders only read in GPU upload heaps (VICTORY_GPU_UPLOAD)"
            );
            on
        })
    }

    /// Whether the host has GPU upload heaps (`D3D12_OPTIONS16`: resizable
    /// BAR and a recent runtime). Asked once.
    fn gpu_upload_heaps(&self) -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
            // SAFETY: a feature query writing the structure of its size.
            let known = unsafe {
                self.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS16,
                    (&raw mut o).cast(),
                    size_of_val(&o) as u32,
                )
            }
            .is_ok();
            known && o.GPUUploadHeapSupported.as_bool()
        })
    }

    /// A mapped buffer for a table every shader reads and only the CPU
    /// writes (the title memory page table, the view strides, marks and
    /// addresses): in a GPU upload heap where the host has one, so shaders
    /// read it from video memory instead of over the bus; upload memory
    /// otherwise, or with `VICTORY_TABLES_GPU_UPLOAD=0`. Both are
    /// write-combined: callers only write it. Outside [`UPLOAD_BUDGET`]:
    /// a few tables per view heap, made once.
    pub(crate) fn table_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        static ON: OnceLock<bool> = OnceLock::new();
        let on = *ON.get_or_init(|| {
            let on = crate::switch::on_by_default("VICTORY_TABLES_GPU_UPLOAD") && self.gpu_upload_heaps();
            tracing::info!(on, "tables shaders read in GPU upload heaps (VICTORY_TABLES_GPU_UPLOAD)");
            on
        });
        static CAP: OnceLock<u64> = OnceLock::new();
        let cap = *CAP.get_or_init(|| {
            let mb = std::env::var("VICTORY_TABLES_GPU_UPLOAD_MAX_MB").ok().and_then(|v| v.parse::<u64>().ok());
            mb.unwrap_or(512).saturating_mul(1 << 20)
        });
        if on {
            if TABLE_BYTES.fetch_add(size, Ordering::Relaxed) + size > cap {
                TABLE_BYTES.fetch_sub(size, Ordering::Relaxed);
                static FULL: std::sync::Once = std::sync::Once::new();
                FULL.call_once(|| {
                    tracing::info!(cap, "tables past VICTORY_TABLES_GPU_UPLOAD_MAX_MB go to upload memory")
                });
            } else if let Some((buffer, data)) = self.shared_mapped_buffer(size, true) {
                let count = TABLE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                // SAFETY: a live buffer; the callback's data is the size alone.
                // Without the callback the table stays counted for good.
                let _ = unsafe {
                    buffer
                        .cast::<windows::Win32::Graphics::Direct3D::ID3DDestructionNotifier>()
                        .and_then(|n| {
                            n.RegisterDestructionCallback(Some(give_back_table), size as *const _)
                        })
                };
                let mb = TABLE_BYTES.load(Ordering::Relaxed) as f64 / (1 << 20) as f64;
                tracing::info!(count, mb, "tables in GPU upload heaps");
                return Some((buffer, data));
            } else {
                TABLE_BYTES.fetch_sub(size, Ordering::Relaxed);
            }
        }
        self.upload_buffer(size)
    }

    /// Whether host copies that shaders write go to GPU upload heaps too,
    /// with a system memory mirror for the CPU (`crate::shadow`):
    /// `VICTORY_GPU_UPLOAD_WRITES=1` with [`Host::gpu_upload`] on. Asked once.
    pub(crate) fn gpu_upload_writes(&self) -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            let on = self.gpu_upload() && crate::switch::on("VICTORY_GPU_UPLOAD_WRITES");
            tracing::info!(
                on,
                "host copies that shaders write in GPU upload heaps (VICTORY_GPU_UPLOAD_WRITES)"
            );
            on
        })
    }

    /// Copies `regions` (destination, its offset, source, its offset,
    /// bytes) between buffers on the GPU, now, and waits: for the GPU's
    /// writes into copies in video memory to reach their system memory
    /// mirrors before the CPU reads them (`crate::shadow::pull`). Runs in a
    /// list of its own after whatever the queue holds; callable from inside
    /// a submission's copy-back. Buffers start and end in the common state
    /// (buffers enter and leave the copy states implicitly).
    pub(crate) fn copy_now(&self, regions: &[(&ID3D12Resource, u64, &ID3D12Resource, u64, u64)]) {
        if regions.is_empty() {
            return;
        }
        // A pull riding a submission ([`Host::pull_riding`]): its copies run
        // after the submission's lists, under their fence wait.
        let recorded = RIDING.with(|r| {
            r.borrow_mut().as_mut().map(|v| {
                v.extend(regions.iter().map(|&(d, da, s, sa, n)| (d.clone(), da, s.clone(), sa, n)));
            })
        });
        if recorded.is_some() {
            return;
        }
        struct Copier {
            allocator: ID3D12CommandAllocator,
            list: ID3D12GraphicsCommandList,
            fence: ID3D12Fence,
            event: HANDLE,
            value: u64,
        }
        // SAFETY: D3D12 objects are free-threaded; the event is waited on
        // under the mutex only.
        unsafe impl Send for Copier {}
        static COPIER: OnceLock<Option<Mutex<Copier>>> = OnceLock::new();
        let copier = COPIER.get_or_init(|| {
            // SAFETY: plain creation; the list is closed until used.
            let made = unsafe {
                (|| -> windows::core::Result<Copier> {
                    let allocator: ID3D12CommandAllocator =
                        self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                    let list: ID3D12GraphicsCommandList =
                        self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                    list.Close()?;
                    let fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
                    let event = CreateEventW(None, false, false, None)?;
                    Ok(Copier { allocator, list, fence, event, value: 0 })
                })()
            };
            match made {
                Ok(c) => Some(Mutex::new(c)),
                Err(e) => {
                    tracing::warn!(error = %e, "no host copier: copies in video memory are not read back");
                    None
                }
            }
        });
        let Some(copier) = copier else {
            return;
        };
        let mut c = copier.lock().unwrap_or_else(|e| e.into_inner());
        c.value += 1;
        let value = c.value;
        // SAFETY: the allocator's last list finished (each call waits); the
        // buffers are live and their ranges inside them (the caller's).
        let ran = unsafe {
            c.allocator.Reset().and_then(|()| {
                c.list.Reset(&c.allocator, None)?;
                for &(dst, dst_at, src, src_at, bytes) in regions {
                    c.list.CopyBufferRegion(dst, dst_at, src, src_at, bytes);
                }
                c.list.Close()?;
                self.queue.ExecuteCommandLists(&[Some(c.list.cast()?)]);
                self.queue.Signal(&c.fence, value)?;
                c.fence.SetEventOnCompletion(value, c.event)?;
                Ok(wait_for_submission(
                    || {
                        // A removed device signals the event and completes
                        // every fence at u64::MAX: not a copy that ran.
                        WaitForSingleObject(c.event, WAIT_SLICE_MS) != WAIT_TIMEOUT
                            && (value..u64::MAX).contains(&c.fence.GetCompletedValue())
                    },
                    || self.device.GetDeviceRemovedReason().is_err(),
                    |secs| tracing::warn!(secs, "host copy still on the GPU"),
                ))
            })
        };
        // The copier stays locked through a failure report, which takes none
        // of it: its allocator may still be in use on the GPU.
        match ran {
            Ok(Waited::Done) => {}
            Ok(Waited::Removed) => fail(format_args!("host GPU device removed during a host copy")),
            Ok(Waited::Stalled) => fail(format_args!(
                "host GPU has not finished a host copy in {STALL_SECS} s and reports no removal: the \
                 GPU has most likely faulted or hung"
            )),
            Err(e) => tracing::warn!(error = %e, "a host copy did not run"),
        }
    }

    /// Before a submission runs: `crate::shadow::pull(snapshots)` with its
    /// copies recorded into a list for the submission to run right after its
    /// own lists, under the same fence wait, instead of a submission and a
    /// wait of their own after it. `Some(None)`: nothing to copy. `None`:
    /// the pull is still due after the wait (no list, or the last one's run
    /// was not waited for); the stale bits it cleared stay cleared, and it
    /// copies its ranges whole again.
    fn pull_riding(&self, snapshots: &[(std::sync::Arc<crate::shadow::Shadow>, usize, usize)]) -> Option<Option<Riding>> {
        static LIST: OnceLock<Option<Mutex<PullList>>> = OnceLock::new();
        let list = LIST.get_or_init(|| {
            // SAFETY: plain creation; the list is closed until used.
            let made = unsafe {
                (|| -> windows::core::Result<PullList> {
                    let allocator: ID3D12CommandAllocator =
                        self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                    let list: ID3D12GraphicsCommandList =
                        self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                    list.Close()?;
                    Ok(PullList { allocator, list, busy: false, value: 0 })
                })()
            };
            made.inspect_err(|e| tracing::warn!(error = %e, "no pull list: pulls wait on their own")).ok().map(Mutex::new)
        });
        let pull = list.as_ref()?;
        let mut guard = pull.lock().unwrap_or_else(|e| e.into_inner());
        // Its last run was not waited for: the allocator may be in use,
        // until the fence passes it.
        // SAFETY: a plain query.
        if guard.busy && unsafe { self.fence.GetCompletedValue() } >= guard.value {
            guard.busy = false;
        }
        if guard.busy {
            return None;
        }
        // Unset on every way out, an unwind included.
        struct Recording;
        impl Drop for Recording {
            fn drop(&mut self) {
                RIDING.with(|r| r.borrow_mut().take());
            }
        }
        RIDING.with(|r| *r.borrow_mut() = Some(Vec::new()));
        let recording = Recording;
        let mut stale = crate::shadow::pull(snapshots);
        let regions = RIDING.with(|r| r.borrow_mut().take()).unwrap_or_default();
        drop(recording);
        if regions.is_empty() {
            return Some(None);
        }
        // SAFETY: the allocator's last list was waited for (`busy`); the
        // buffers are live (held in `regions`) and the ranges inside them.
        let recorded = unsafe {
            guard.allocator.Reset().and_then(|()| {
                guard.list.Reset(&guard.allocator, None)?;
                for (dst, dst_at, src, src_at, bytes) in &regions {
                    guard.list.CopyBufferRegion(dst, *dst_at, src, *src_at, *bytes);
                }
                guard.list.Close()?;
                guard.list.cast::<ID3D12CommandList>()
            })
        };
        match recorded {
            Ok(list) => {
                // Busy from now: its value is set once the submission signals.
                guard.busy = true;
                guard.value = u64::MAX;
                Some(Some(Riding { pull, list, _regions: regions, stale }))
            }
            Err(e) => {
                tracing::warn!(error = %e, "pull list not recorded: the pull waits on its own");
                stale.restale();
                None
            }
        }
    }

    /// A [`Host::shared_buffer`] in system memory, or (`upload`) in a GPU
    /// upload heap ([`Host::gpu_upload_buffer`]).
    fn shared_mapped_buffer(&self, size: u64, upload: bool) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = if upload {
            D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_GPU_UPLOAD,
                ..Default::default()
            }
        } else {
            D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            }
        };
        #[cfg(feature = "diagnostics")]
        let heap = if self.copies_in_video_memory() {
            D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() }
        } else {
            heap
        };
        // Small ones in system memory are placed in heaps made ahead
        // (`VICTORY_MEMSYS`), unless the diagnostics above move them; those
        // in video memory were placed by `gpu_upload_buffer`, or are not.
        #[cfg(feature = "diagnostics")]
        let placed = !upload && !self.copies_in_video_memory();
        #[cfg(not(feature = "diagnostics"))]
        let placed = !upload;
        if placed && let Some(made) = crate::memsys::heaps::buffer(&self.device, false, size) {
            return Some(made);
        }
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => {
                use crate::memsys::Class;
                crate::memsys::made(if upload { Class::Bar } else { Class::Shared }, size);
                Some((buffer?, data.cast()))
            }
            Err(e) => {
                tracing::warn!(error = %e, size, upload, "no shared buffer");
                None
            }
        }
    }
}

/// Host resources that view heap slots stopped viewing while a submission
/// may have been running ([`retire`]).
#[allow(clippy::type_complexity)]
static RETIRED: Mutex<Vec<(ID3D12Resource, Option<std::sync::Arc<crate::shadow::Shadow>>)>> = Mutex::new(Vec::new());

/// Keeps `resource` alive until the GPU work running now (or the next
/// submission's) has run: title threads rewrite view heap slots while the
/// GPU runs, and a shader that took the old descriptor still reads what it
/// views.
pub(crate) fn retire(resource: ID3D12Resource, copy: Option<std::sync::Arc<crate::shadow::Shadow>>) {
    RETIRED.lock().unwrap_or_else(|e| e.into_inner()).push((resource, copy));
}

/// The host fence value of the last group sent to the GPU (stored before its
/// lists go, so work that reads memory written before this call is covered
/// by the value returned): with [`completed`], what view heaps wait for
/// before they rewrite a slot work in flight may read (`crate::descriptor`).
// `Host::submit` stores it where it sets `t.value`, for each group; a
// pipelined retire changes nothing here: [`completed`] reads the fence.
static SENT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn sent() -> u64 {
    #[cfg(test)]
    if let Some((sent, _)) = FENCE.get() {
        return sent;
    }
    SENT.load(Ordering::SeqCst)
}

#[cfg(test)]
thread_local! {
    /// Tests: this thread's [`sent`] and [`completed`] values, not the host's
    /// (work in flight without a submission).
    pub(crate) static FENCE: std::cell::Cell<Option<(u64, u64)>> = const { std::cell::Cell::new(None) };
}

/// The host fence's completed value: every group up to it has run (all of
/// them without a device: nothing runs).
pub(crate) fn completed() -> u64 {
    #[cfg(test)]
    if let Some((_, completed)) = FENCE.get() {
        return completed;
    }
    // SAFETY: a live fence.
    host().map_or(u64::MAX, |h| unsafe { h.fence.GetCompletedValue() })
}

/// After a submission has run, with none running: the retired resources go.
pub(crate) fn release_retired() {
    let retired = std::mem::take(&mut *RETIRED.lock().unwrap_or_else(|e| e.into_inner()));
    if !retired.is_empty() {
        release_later(Box::new(retired));
    }
}

/// Drops `what` on a thread of its own: releasing a D3D12 resource is a
/// kernel call (~5 % of a world frame on the title's submitting thread,
/// run perf1). Only for objects no GPU work in flight uses.
pub(crate) fn release_later(what: Box<dyn std::any::Any + Send>) {
    type Release = std::sync::mpsc::Sender<Box<dyn std::any::Any + Send>>;
    static RELEASE: OnceLock<Release> = OnceLock::new();
    let release = RELEASE.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Box<dyn std::any::Any + Send>>();
        // Without the thread the receiver is gone: sends fail and drop here.
        let _ = std::thread::Builder::new()
            .name("victory-release".into())
            .spawn(move || rx.into_iter().for_each(drop));
        tx
    });
    let _ = release.send(what);
}

/// Diagnostics: a noted command running a shader `VICTORY_TRACE_SHADERS`
/// names (hashes, comma-separated) is logged with what it binds: all of
/// them in listed capture frames, the first 500 elsewhere.
#[cfg(feature = "diagnostics")]
fn traced(what: &str) {
    static LOGGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if is_traced(what)
        && (crate::capture::listed(crate::queue::presented() + 1)
            || LOGGED.fetch_add(1, Ordering::Relaxed) < 500)
    {
        tracing::info!("traced command: {what}");
    }
}

/// The shader hashes `VICTORY_TRACE_SHADERS` lists (comma-separated;
/// diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn traced_shaders() -> &'static [String] {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    HASHES.get_or_init(|| {
        let v = std::env::var("VICTORY_TRACE_SHADERS").unwrap_or_default();
        v.split(',').map(str::trim).filter(|h| !h.is_empty()).map(str::to_owned).collect()
    })
}

/// Whether `what` names a shader [`traced_shaders`] lists.
#[cfg(feature = "diagnostics")]
pub(crate) fn is_traced(what: &str) -> bool {
    traced_shaders().iter().any(|h| what.contains(h.as_str()))
}

/// Diagnostics, `VICTORY_GPU_TIMING`: every command of the list during
/// whose run (each list alone) the device was found removed.
fn removed_during(runs: &[&HostList], list: Option<usize>) -> String {
    #[cfg(feature = "diagnostics")]
    if let Some(i) = list {
        let notes = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner());
        let mut text = format!("; removed while list {i} ran alone ({} commands):", notes.len());
        for (k, (n, _)) in notes.iter().enumerate() {
            text.push_str(&format!(" [{k}] {n}"));
        }
        return text;
    }
    let _ = (runs, list);
    String::new()
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    for i in from..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    #[cfg(test)]
    tests::ISSUED.with(|l| l.borrow_mut().push((resource.as_raw() as usize, before.0 as u32, after.0 as u32)));
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

impl Host {
    /// Records a list of our own with `record`, runs it now, after
    /// whatever the queue holds, and waits for it: for work between
    /// submissions that is more than copies ([`Host::copy_now`]), such as
    /// `crate::memsys::diff`'s compares. Buffers start and end in the
    /// common state unless `record` moves them back itself. False when
    /// nothing ran (no list on this host, or the run failed; logged).
    pub(crate) fn run_now(&self, record: &mut dyn FnMut(&ID3D12GraphicsCommandList)) -> bool {
        struct Runner {
            allocator: ID3D12CommandAllocator,
            list: ID3D12GraphicsCommandList,
            fence: ID3D12Fence,
            event: HANDLE,
            value: u64,
        }
        // SAFETY: D3D12 objects are free-threaded; the event is waited on
        // under the mutex only.
        unsafe impl Send for Runner {}
        static RUNNER: OnceLock<Option<Mutex<Runner>>> = OnceLock::new();
        let runner = RUNNER.get_or_init(|| {
            // SAFETY: plain creation; the list is closed until used.
            let made = unsafe {
                (|| -> windows::core::Result<Runner> {
                    let allocator: ID3D12CommandAllocator =
                        self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                    let list: ID3D12GraphicsCommandList =
                        self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                    list.Close()?;
                    let fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
                    let event = CreateEventW(None, false, false, None)?;
                    Ok(Runner { allocator, list, fence, event, value: 0 })
                })()
            };
            match made {
                Ok(r) => Some(Mutex::new(r)),
                Err(e) => {
                    tracing::warn!(error = %e, "no list of our own to run between submissions");
                    None
                }
            }
        });
        let Some(runner) = runner else {
            return false;
        };
        let mut r = runner.lock().unwrap_or_else(|e| e.into_inner());
        r.value += 1;
        let value = r.value;
        // SAFETY: the allocator's last list finished (each call waits);
        // what `record` names is live (the caller's).
        let ran = unsafe {
            r.allocator.Reset().and_then(|()| {
                r.list.Reset(&r.allocator, None)?;
                record(&r.list);
                r.list.Close()?;
                self.queue.ExecuteCommandLists(&[Some(r.list.cast()?)]);
                self.queue.Signal(&r.fence, value)?;
                r.fence.SetEventOnCompletion(value, r.event)?;
                Ok(wait_for_submission(
                    || {
                        // A removed device signals the event and completes
                        // every fence at u64::MAX: not a list that ran.
                        WaitForSingleObject(r.event, WAIT_SLICE_MS) != WAIT_TIMEOUT
                            && (value..u64::MAX).contains(&r.fence.GetCompletedValue())
                    },
                    || self.device.GetDeviceRemovedReason().is_err(),
                    |secs| tracing::warn!(secs, "a list of our own still on the GPU"),
                ))
            })
        };
        // The runner stays locked through a failure report, which takes
        // none of it: its allocator may still be in use on the GPU.
        match ran {
            Ok(Waited::Done) => true,
            Ok(Waited::Removed) => fail(format_args!("host GPU device removed during a list of our own")),
            Ok(Waited::Stalled) => fail(format_args!(
                "host GPU has not finished a list of our own in {STALL_SECS} s and reports no removal: the \
                 GPU has most likely faulted or hung"
            )),
            Err(e) => {
                tracing::warn!(error = %e, "a list of our own did not run");
                false
            }
        }
    }
}

#[cfg(test)]
mod batch {   // `mod tests` is already taken at the top of this file
    /// Room in `gpu_writes` no real batch has: names the batch of
    /// `a_batch_released_later_is_recycled_on_the_releasing_thread`.
    pub(super) const MARK: usize = 54_321;
    /// The thread that batch ended on.
    pub(super) static ENDED_ON: std::sync::Mutex<Option<std::thread::ThreadId>> = std::sync::Mutex::new(None);

    /// A submission's batch ends on the releasing thread
    /// (`release_later`), a little after the next submission began: its
    /// lists are emptied and kept from there too.
    #[test]
    fn a_batch_released_later_is_recycled_on_the_releasing_thread() {
        // Submissions take and leave spares: none meanwhile.
        let _quiet = super::tests::no_submissions();
        let mut batch = super::Batch::default();
        batch.gpu_writes.reserve(MARK);
        batch.gpu_write(0x1000, 0x100);
        batch.late.push((1, 2, 3));
        super::release_later(Box::new(batch));
        let kept = || {
            let spares = super::SPARE.lock().unwrap_or_else(|e| e.into_inner());
            // Batches of other tests may have filled the spares first.
            let ours = spares.iter().any(|s| s.gpu_writes.capacity() >= MARK && s.gpu_writes.is_empty() && s.late.is_empty());
            ours || spares.len() == super::SPARES
        };
        let mut ended_on = None;
        for _ in 0..500 {
            ended_on = *ENDED_ON.lock().unwrap_or_else(|e| e.into_inner());
            if ended_on.is_some() && kept() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(ended_on.is_some(), "the batch ended");
        assert_ne!(ended_on, Some(std::thread::current().id()), "on the releasing thread, not here");
        assert!(kept(), "its lists were kept for the next batch");
    }

    /// A batch that ended leaves its lists and tables, emptied, to the
    /// next: nothing of the old one's copies or ranges is in it.
    #[test]
    fn a_batch_reuses_what_the_last_left_and_starts_empty() {
        let mut batch = super::Batch::reused();
        batch.gpu_write(0x1000, 0x100);
        batch.late.push((1, 2, 3));
        batch.pending.insert(7, 0);
        assert!(batch.gpu_written(0x1000, 1));
        drop(batch);
        // Other tests' batches end meanwhile too: any of them is empty.
        for _ in 0..8 {
            let next = super::Batch::reused();
            assert!(next.refreshes.is_empty() && next.pending.is_empty() && next.current.is_empty());
            assert!(next.late.is_empty() && !next.gpu_written(0x1000, 1));
        }
    }

    use super::Batch;

    /// A host copy of tracked title memory: a submission whose lists all ask
    /// for it whole copies the one page the title wrote, once, and again
    /// only after a step that writes title memory.
    #[test]
    fn a_copy_is_refreshed_once_per_submission_by_the_pages_written() {
        use crate::watch::PAGE;
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        crate::shadow::register(p, p as u64, len as u64);
        let Some(copy) = crate::shadow::find(p as u64, len as u64) else {
            crate::shadow::unregister(p, p as u64);
            return; // no host GPU here
        };
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), len, "all of it at first");
        // SAFETY: our pool, guarded since the copy took it.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(7) };
        // The next submission: two lists ask for the copy.
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        batch.add(&copy, p + 100, 200);
        assert_eq!(batch.flush(), PAGE, "the page written");
        assert_eq!(copy.read_host((p + PAGE + 8) as u64, 1)[0], 7);
        // SAFETY: as above.
        unsafe { (p as *mut u8).write_volatile(1) };
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), 0, "nothing again in the submission");
        batch.title_written();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), PAGE, "after a title write, what changed");
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }

    /// A copy whose memory nothing wrote since it was taken is left out of
    /// the batch (no refresh queued, none remembered); a write the guard
    /// takes or one that is reported brings it back, for the pages written.
    #[test]
    fn a_copy_nothing_wrote_is_not_asked_for() {
        use crate::watch::PAGE;
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        crate::shadow::register(p, p as u64, len as u64);
        let Some(copy) = crate::shadow::find(p as u64, len as u64) else {
            crate::shadow::unregister(p, p as u64);
            return; // no host GPU here
        };
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        assert_eq!((batch.refreshes.len(), batch.flush()), (1, len), "all of it at first");
        // Later lists and submissions: nothing queued, nothing kept.
        batch.add(&copy, p, len);
        batch.title_written();
        batch.add(&copy, p + PAGE, 100);
        let mut next = Batch::default();
        next.add(&copy, p, len);
        assert!(batch.refreshes.is_empty() && next.refreshes.is_empty());
        assert_eq!(next.flush(), 0);
        // SAFETY: our pool, guarded since the copy took it.
        unsafe { ((p + 2 * PAGE + 8) as *mut u8).write_volatile(7) };
        next.add(&copy, p, len);
        assert_eq!((next.refreshes.len(), next.flush()), (1, PAGE), "the page written");
        assert_eq!(copy.read_host((p + 2 * PAGE + 8) as u64, 1)[0], 7);
        // A write of our own or the kernel's, reported.
        drop(crate::watch::writing(p + 3 * PAGE, 16));
        next.title_written();
        next.add(&copy, p, len);
        assert_eq!(next.flush(), PAGE, "the page reported");
        next.add(&copy, p, len);
        assert!(next.refreshes.is_empty());
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }


    /// Two partial refreshes of a page do not make it current: neither took
    /// it, so a whole refresh still copies it (upstream #2473).
    #[test]
    fn parts_of_a_page_do_not_make_it_current() {
        use crate::watch::PAGE;
        let p = crate::watch::tests::pool(1);
        crate::shadow::register(p, p as u64, PAGE as u64);
        let Some(copy) = crate::shadow::find(p as u64, PAGE as u64) else {
            crate::shadow::unregister(p, p as u64);
            eprintln!("skipped parts_of_a_page_do_not_make_it_current: no host GPU");
            return;
        };
        let mut batch = Batch::default();
        batch.add(&copy, p, 100);
        batch.flush();
        batch.add(&copy, p + 100, PAGE - 100);
        batch.flush();
        batch.add(&copy, p, PAGE);
        assert_eq!(batch.flush(), PAGE);
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }

    /// A world frame's refreshes, on a made-up heap shaped like the world's
    /// (#400): 36 submissions, each after the title wrote 256 scattered
    /// pages of a 174 MB tracked heap, each declaring 24 ranges of it (of
    /// 64 views) as written by marking shaders, refreshed through `Batch`
    /// as `Host::execute` does (ranges armed, then one flush). Times the
    /// flushes with the copies on one thread and on the workers
    /// (`crate::copy_pool`), and checks every title write reached the copy.
    /// Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture world_frame_refreshes`.
    #[test]
    #[ignore = "benchmark"]
    fn world_frame_refreshes() {
        use crate::watch::PAGE;
        use std::sync::atomic::Ordering;
        use std::time::{Duration, Instant};
        const HEAP: usize = 174 << 20;
        const SUBMISSIONS: usize = 36;
        const DECLARED: usize = 24;
        const VIEWS: usize = 64;
        const TITLE_PAGES: usize = 256;
        const FRAMES: usize = 6;
        let p = crate::watch::tests::pool(HEAP / PAGE);
        crate::shadow::register(p, p as u64, HEAP as u64);
        let Some(copy) = crate::shadow::find(p as u64, HEAP as u64) else {
            crate::shadow::unregister(p, p as u64);
            crate::watch::tests::free(p);
            return; // no host GPU here
        };
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        // Views of the heap: 256-byte aligned, 64 KiB to 32 MiB.
        let views: Vec<(usize, usize)> = (0..VIEWS)
            .map(|_| {
                let len = (64 << 10) << next(10);
                let at = next((HEAP - len) / 256) * 256;
                (p + at, len)
            })
            .collect();
        let mut batch = Batch::default();
        batch.add(&copy, p, HEAP);
        batch.flush();
        let mut stamp = 1u64;
        // Not with another test switching the refreshes' threads meanwhile.
        let _serial = crate::copy_pool::test_lock();
        let mut frame = |serial: bool| -> (Duration, usize) {
            crate::copy_pool::SERIAL.store(serial, Ordering::Relaxed);
            let (mut took, mut copied) = (Duration::ZERO, 0);
            for _ in 0..SUBMISSIONS {
                let writes: Vec<usize> = (0..TITLE_PAGES)
                    .map(|_| p + next(HEAP / PAGE) * PAGE + next(PAGE / 8) * 8)
                    .collect();
                for &at in &writes {
                    // SAFETY: 8 aligned bytes of our pool.
                    unsafe { (at as *mut u64).write_volatile(stamp) };
                    stamp += 1;
                }
                let declared: Vec<(usize, usize)> =
                    (0..DECLARED).map(|_| views[next(VIEWS)]).collect();
                let mut batch = Batch::default();
                for &(at, len) in &declared {
                    batch.add(&copy, at, len);
                    copy.writing(at, len, true);
                }
                let started = Instant::now();
                copied += batch.flush();
                took += started.elapsed();
                // Every title write inside a declared range is in the copy.
                for &at in &writes {
                    if declared.iter().any(|&(a, n)| a <= at && at + 8 <= a + n) {
                        // SAFETY: 8 aligned bytes of our pool.
                        let title = unsafe { (at as *const u64).read_volatile() };
                        let host = copy.read_host(at as u64, 2);
                        assert_eq!(
                            u64::from(host[0]) | u64::from(host[1]) << 32,
                            title,
                            "{at:#x}"
                        );
                    }
                }
                for &(at, len) in &declared {
                    copy.copy_back(at, len);
                }
            }
            (took, copied)
        };
        let mut report = |serial: bool| {
            frame(serial);
            let mut ms: Vec<f64> = (0..FRAMES)
                .map(|_| {
                    let (took, copied) = frame(serial);
                    println!(
                        "  {}: {:.1} ms, {} MB copied",
                        if serial { "one thread" } else { "workers" },
                        took.as_secs_f64() * 1e3,
                        copied >> 20
                    );
                    took.as_secs_f64() * 1e3
                })
                .collect();
            ms.sort_by(f64::total_cmp);
            ms[FRAMES / 2]
        };
        let serial = report(true);
        let workers = report(false);
        println!("refresh per frame, median: one thread {serial:.1} ms, workers {workers:.1} ms");
        crate::copy_pool::SERIAL.store(false, Ordering::Relaxed);
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }
}

#[cfg(test)]
mod withholding {
    /// A list in which a program using title memory missed memory: what
    /// that program wrote is withheld from title memory and leaves the
    /// copies at once; what an ordinary program wrote in the same list goes
    /// back and stays in its copy (run light13: the imposter transforms in
    /// the BVH builder's list); an ordinary range overlapping a withheld one
    /// is withheld with it, since its copy-back would take the other's bytes
    /// along.
    #[test]
    fn only_programs_using_title_memory_are_withheld() {
        use crate::watch::PAGE;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let _alone = crate::memory::without_a_miss();
        let pools = [crate::watch::tests::pool(1), crate::watch::tests::pool(1)];
        for &p in &pools {
            // SAFETY: our pool, not guarded yet.
            unsafe { ((p + 16) as *mut u8).write_volatile(0x11) };
            crate::shadow::register(p, p as u64, PAGE as u64);
        }
        let copies: Vec<_> = pools.iter().map(|&p| crate::shadow::find(p as u64, PAGE as u64).unwrap()).collect();
        let list = host.new_list().unwrap();
        // (copy, start, bytes, the writer uses title memory, byte written at +0 and +16)
        let writes = [(0, 0, 32, false, 0x22), (1, 16, 16, true, 0x33), (1, 0, 32, false, 0x44)];
        for &(k, at, len, memory, value) in &writes {
            let _writer = list.writer_uses_memory(memory);
            let (copy, address) = (copies[k].clone(), (pools[k] + at) as u64);
            copy.refresh(&list, address, len);
            copy.gpu_writes(&list, address, len, false);
            // The shader's write, after the step that keeps the range's bytes.
            list.defer_host(move || copy.write_host(address, &[value]));
        }
        list.defer_missed(|| true);
        list.close();
        host.execute(&[&list]);
        // SAFETY: bytes of our pools.
        let title = |k: usize, at: usize| unsafe { ((pools[k] + at) as *const u8).read_volatile() };
        let copy = |k: usize, at: usize| {
            copies[k].update(pools[k], PAGE);
            copies[k].read_host((pools[k] + at) as u64, 1)[0] & 0xff
        };
        assert_eq!((title(0, 0), copy(0, 0)), (0x22, 0x22), "an ordinary program's write goes back and stays");
        assert_eq!((title(1, 16), copy(1, 16)), (0x11, 0x11), "withheld, then dropped from the copy");
        assert_eq!((title(1, 0), copy(1, 0)), (0, 0), "overlapping a withheld range: withheld with it");
        for (&p, copy) in pools.iter().zip(copies) {
            crate::shadow::unregister(p, p as u64);
            drop(copy);
            crate::watch::tests::free(p);
        }
    }

    /// An ordinary marked range withheld because it overlaps a title-memory
    /// withheld range drops the pages marked in the ordinary bitmap via
    /// `copy_back_marks`, while leaving unmarked pages in both ranges clean.
    #[test]
    fn withheld_overlapping_ordinary_marked_write_drops_only_marked_pages() {
        use crate::watch::PAGE;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let _alone = crate::memory::without_a_miss();
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        for (page, byte) in [(0, 0x11u8), (1, 0x22), (2, 0x33), (3, 0x44)] {
            // SAFETY: our pool, not guarded yet.
            unsafe { ((p + page * PAGE + 16) as *mut u8).write_volatile(byte) };
        }
        crate::shadow::register(p, p as u64, len as u64);
        let copy = crate::shadow::find(p as u64, len as u64).unwrap();
        let list = host.new_list().unwrap();
        {
            let _writer = list.writer_uses_memory(true);
            copy.refresh(&list, p as u64, (2 * PAGE) as u64);
            copy.gpu_writes(&list, p as u64, (2 * PAGE) as u64, true);
        }
        {
            let _writer = list.writer_uses_memory(false);
            copy.refresh(&list, (p + PAGE) as u64, (3 * PAGE) as u64);
            copy.gpu_writes(&list, (p + PAGE) as u64, (3 * PAGE) as u64, true);
        }
        let c = copy.clone();
        list.defer(move || {
            c.write_host((p + 16) as u64, &[0xaa]);
            c.mark_page_for_test(p, true);
            c.write_host((p + PAGE + 16) as u64, &[0xbb]);
            c.mark_page_for_test(p + PAGE, false);
            // A page the ordinary range alone covers: out only by the overlap.
            c.write_host((p + 2 * PAGE + 16) as u64, &[0xcc]);
            c.mark_page_for_test(p + 2 * PAGE, false);
        });
        list.defer_missed(|| true);
        list.close();
        host.execute(&[&list]);
        // SAFETY: bytes of our pool.
        let title = |page: usize| unsafe { ((p + page * PAGE + 16) as *const u8).read_volatile() };
        assert_eq!(
            (title(0), title(1), title(2)),
            (0x11, 0x22, 0x33),
            "both withheld writes stay out of title memory"
        );
        assert_eq!(
            copy.update(p + 3 * PAGE, PAGE),
            0,
            "unmarked page 3 stays clean"
        );
        assert_eq!(
            copy.update(p + 2 * PAGE, PAGE),
            PAGE,
            "page 2, the ordinary range's own, dropped"
        );
        assert_eq!(
            copy.update(p, PAGE),
            PAGE,
            "title-memory marked page 0 dropped"
        );
        assert_eq!(
            copy.update(p + PAGE, PAGE),
            PAGE,
            "ordinary-bitmap withheld page 1 dropped"
        );
        assert_eq!(copy.read_host((p + 16) as u64, 1)[0] & 0xff, 0x11);
        assert_eq!(copy.read_host((p + PAGE + 16) as u64, 1)[0] & 0xff, 0x22);
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }

    /// A withheld range declared by both a marked step and an unmarked step
    /// merges `marked` with AND and still drops the whole range, so unmarked
    /// writes cannot remain stale in the host copy.
    #[test]
    fn withheld_range_mixing_marked_and_unmarked_steps_drops_whole_range() {
        use crate::watch::PAGE;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let _alone = crate::memory::without_a_miss();
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        for (page, byte) in [(0, 0x11u8), (1, 0x22)] {
            // SAFETY: our pool, not guarded yet.
            unsafe { ((p + page * PAGE + 16) as *mut u8).write_volatile(byte) };
        }
        crate::shadow::register(p, p as u64, len as u64);
        let copy = crate::shadow::find(p as u64, len as u64).unwrap();
        let list = host.new_list().unwrap();
        {
            let _writer = list.writer_uses_memory(true);
            copy.refresh(&list, p as u64, len as u64);
            copy.gpu_writes(&list, p as u64, len as u64, true);
        }
        {
            let _writer = list.writer_uses_memory(false);
            copy.gpu_writes(&list, p as u64, len as u64, false);
        }
        let c = copy.clone();
        list.defer(move || {
            // Unmarked write to page 1 (no mark bit set).
            c.write_host((p + PAGE + 16) as u64, &[0xcc]);
        });
        list.defer_missed(|| true);
        list.close();
        host.execute(&[&list]);
        // SAFETY: bytes of our pool.
        let title = |page: usize| unsafe { ((p + page * PAGE + 16) as *const u8).read_volatile() };
        assert_eq!(
            (title(0), title(1)),
            (0x11, 0x22),
            "withheld range stays out of title memory"
        );
        assert_eq!(
            copy.update(p, len),
            len,
            "mixed marked/unmarked range drops whole"
        );
        assert_eq!(
            copy.read_host((p + PAGE + 16) as u64, 1)[0] & 0xff,
            0x22,
            "unmarked write on page 1 is cleared from the copy"
        );
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }

    /// Dropping withheld marked pages in `copy_back_marks` runs after all
    /// non-withheld marked copy-backs, so an immediate retake on a shared page
    /// does not mark another copy's page changed before its own copy-back.
    #[test]
    fn withheld_marked_drop_runs_after_other_copy_marked_copy_back() {
        drop_after_other_copy(false);
    }

    /// The same where the withheld copy's mark is in the bitmap of shaders
    /// using title memory.
    #[test]
    fn held_title_memory_mark_drops_after_other_copy_marked_copy_back() {
        drop_after_other_copy(true);
    }

    fn drop_after_other_copy(memory: bool) {
        use crate::watch::PAGE;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let _alone = crate::memory::without_a_miss();
        let p = crate::watch::tests::pool(2);
        // SAFETY: our pool, not guarded yet.
        unsafe {
            ((p + 16) as *mut u8).write_volatile(0x11);
            ((p + 32) as *mut u8).write_volatile(0x22);
        }
        crate::shadow::register(p, p as u64, (2 * PAGE) as u64);
        crate::shadow::register(p + 1, p as u64, PAGE as u64);
        let outer = crate::shadow::find(p as u64, (2 * PAGE) as u64).unwrap();
        let inner = crate::shadow::find(p as u64, PAGE as u64).unwrap();
        assert!(!std::sync::Arc::ptr_eq(&outer, &inner));
        let list = host.new_list().unwrap();
        {
            // Outer has a withheld marked range, plus an overlapping ordinary
            // marked write whose mark lands in the ordinary bitmap.
            let _writer = list.writer_uses_memory(true);
            outer.refresh(&list, p as u64, (2 * PAGE) as u64);
            outer.gpu_writes(&list, p as u64, (2 * PAGE) as u64, true);
        }
        {
            let _writer = list.writer_uses_memory(false);
            outer.gpu_writes(&list, p as u64, PAGE as u64, true);
            inner.refresh(&list, p as u64, PAGE as u64);
            inner.gpu_writes(&list, p as u64, PAGE as u64, true);
        }
        let (o, i) = (outer.clone(), inner.clone());
        list.defer(move || {
            o.write_host((p + 16) as u64, &[0xaa]);
            o.mark_page_for_test(p, memory);
            i.write_host((p + 32) as u64, &[0xbb]);
            i.mark_page_for_test(p, false);
        });
        list.defer_missed(|| true);
        list.close();
        host.execute(&[&list]);
        // SAFETY: bytes of our pool.
        let title = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!(
            title(16),
            0x11,
            "withheld outer write does not reach title memory"
        );
        assert_eq!(
            title(32),
            0xbb,
            "non-withheld inner write copies back before outer's page is retaken"
        );
        assert_eq!(
            outer.update(p + PAGE, PAGE),
            0,
            "unmarked outer page 1 stays clean"
        );
        assert_eq!(
            outer.update(p, PAGE),
            PAGE,
            "withheld marked outer page 0 was dropped"
        );
        assert_eq!(outer.read_host((p + 16) as u64, 1)[0] & 0xff, 0x11);
        assert_eq!(outer.read_host((p + 32) as u64, 1)[0] & 0xff, 0xbb);
        crate::shadow::unregister(p + 1, p as u64);
        crate::shadow::unregister(p, p as u64);
        drop((outer, inner));
        crate::watch::tests::free(p);
    }
}

#[cfg(all(test, feature = "diagnostics"))]
mod present_plane_tests {
    use super::*;

    #[test]
    fn compositor_failure_presents_primary_in_the_same_frame() {
        let Some(host) = host() else {
            eprintln!("skipping PresentX recovery: no D3D12 device");
            return;
        };
        let list = host.new_list().unwrap();
        let make_plane = |width, height, color: &[f32; 4]| {
            let mut texture = None;
            // SAFETY: live device, valid render target description, and an open list.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_DEFAULT,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                            Width: width,
                            Height: height,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                            SampleDesc: DXGI_SAMPLE_DESC {
                                Count: 1,
                                Quality: 0,
                            },
                            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                            ..Default::default()
                        },
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        None,
                        &mut texture,
                    )
                    .unwrap();
                let texture = texture.unwrap();
                list.list
                    .ClearRenderTargetView(host.rtv(&texture, None), color, None);
                texture
            }
        };
        let scene = make_plane(32, 32, &[1.0, 0.0, 0.0, 1.0]);
        let overlay = make_plane(64, 64, &[0.0, 0.0, 128.0 / 255.0, 128.0 / 255.0]);
        list.close();
        host.execute(&[&list]);
        let state = D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32;
        assert!(
            host.present(&[(&scene, state)], 1).is_some(),
            "single-plane control"
        );
        let first_pixel =
            |buffer: &ID3D12Resource, footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| {
                let mut mapped = std::ptr::null_mut();
                // SAFETY: the present copy has completed. The footprint names
                // a live RGBA pixel inside the readback buffer.
                let rgba = unsafe {
                    buffer.Map(0, None, Some(&mut mapped)).unwrap();
                    let rgba = mapped
                        .cast::<u8>()
                        .add(footprint.Offset as usize)
                        .cast::<[u8; 4]>()
                        .read();
                    buffer.Unmap(0, None);
                    rgba
                };
                (footprint.Footprint.Width, footprint.Footprint.Height, rgba)
            };

        let mut attempted = false;
        let mut fallback = None;
        let shown = host.present_with_compositor(
            &[(&scene, state), (&overlay, state)],
            2,
            |_, _| {
                attempted = true;
                Err(windows::core::Error::from_hresult(
                    windows::Win32::Foundation::E_FAIL,
                ))
            },
            |buffer, footprint| fallback = Some(first_pixel(buffer, footprint)),
        );
        assert!(attempted);
        assert!(
            shown.is_some(),
            "the primary plane must still be shown in this frame"
        );
        assert_eq!(fallback, Some((32, 32, [255, 0, 0, 255])));
        let presentation = host.presenter.lock().unwrap_or_else(|e| e.into_inner());
        let swap_chain = presentation.presenter.as_ref().unwrap();
        assert_eq!((swap_chain.width, swap_chain.height), (32, 32));
        drop(presentation);

        let mut composite = None;
        assert!(
            host.present_with_compositor(
                &[(&scene, state), (&overlay, state)],
                3,
                Compositor::new,
                |buffer, footprint| composite = Some(first_pixel(buffer, footprint)),
            )
            .is_some(),
            "compositor initialization can be retried after failure"
        );
        // Premultiplied half-blue over opaque red.
        assert_eq!(composite, Some((64, 64, [127, 0, 128, 255])));
    }
}

#[cfg(test)]
mod copy_order {
    /// A deferred title copy recorded behind work its shaders do reads what
    /// they wrote: run after the copy-backs, it sees the GPU's bytes; one
    /// touching nothing the GPU writes runs at once. Both the copy's source
    /// and its destination order against the declared writes.
    #[test]
    fn a_copy_past_a_gpu_write_sees_what_it_wrote() {
        use crate::watch::PAGE;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        // src of a GPU write, a destination behind it, a plain source, a
        // plain destination.
        let pools = [
            crate::watch::tests::pool(1),
            crate::watch::tests::pool(1),
            crate::watch::tests::pool(1),
            crate::watch::tests::pool(1),
        ];
        // SAFETY: our pools, not guarded yet.
        unsafe {
            (pools[0] as *mut u8).write_volatile(0xAA);
            (pools[2] as *mut u8).write_volatile(0xCC);
        }
        crate::shadow::register(pools[0], pools[0] as u64, PAGE as u64);
        let copy = crate::shadow::find(pools[0] as u64, PAGE as u64).unwrap();
        let list = host.new_list().unwrap();
        // The program writes the first page of pool 0: the declaration
        // first (its snapshot keeps 0xAA), the shader's write after.
        copy.refresh(&list, pools[0] as u64, PAGE as u64);
        copy.gpu_writes(&list, pools[0] as u64, PAGE as u64, false);
        let src_copy = copy.clone();
        list.defer_host(move || src_copy.write_host(pools[0] as u64, &[0xBB]));
        // Pools 0 -> 1 reads the declared write; pools 2 -> 0 writes over
        // it; pools 2 -> 3 touches nothing the GPU writes.
        list.defer_copy(pools[1] as *mut u8, pools[0] as *const u8, PAGE);
        list.defer_copy(pools[0] as *mut u8, pools[2] as *const u8, PAGE);
        list.defer_copy(pools[3] as *mut u8, pools[2] as *const u8, PAGE);
        list.close();
        host.execute(&[&list]);
        // SAFETY: bytes of our pools.
        let at = |k: usize| unsafe { (pools[k] as *const u8).read_volatile() };
        assert_eq!(at(1), 0xBB, "the copy sees what the shaders wrote");
        assert_eq!(at(0), 0xCC, "a copy over the write lands after it");
        assert_eq!(at(3), 0xCC, "an unrelated copy is copied at once");
        crate::shadow::unregister(pools[0], pools[0] as u64);
        drop(copy);
        for &p in &pools {
            crate::watch::tests::free(p);
        }
    }

    /// With `VICTORY_MEMSYS_ALIAS`, a buffer showing physical pages that
    /// another buffer's copy holds takes that copy, at its pages: one
    /// refresh serves both addresses, a title write through either is
    /// seen, and a remap gives the buffer a copy of its own again.
    #[test]
    fn buffers_showing_the_same_pages_share_one_copy() {
        use crate::memsys::alias::ON_HERE;
        use crate::watch::PAGE;
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE};
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let len = 4 * PAGE;
        // SAFETY: a pagefile section mapped read-write three times (two
        // title addresses and a view of our own); the test keeps them.
        let (a, b, own) = unsafe {
            let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, len as u32, std::ptr::null());
            assert!(!section.is_null());
            let map = || MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize;
            (map(), map(), map())
        };
        assert!(a != 0 && b != 0 && own != 0);
        struct Here;
        impl Drop for Here {
            fn drop(&mut self) {
                ON_HERE.set(false);
            }
        }
        ON_HERE.set(true);
        let _here = Here;
        // As the runtime reports a mapping: what it shows, then where.
        crate::watch::title_memory_remapped(a, len);
        crate::watch::title_memory_own(a, len, own);
        crate::watch::title_memory_remapped(b, len);
        crate::watch::title_memory_own(b, len, own);
        for k in 0..4 {
            crate::watch::title_memory_aliased(b + k * PAGE, &[a + k * PAGE]);
        }
        let (au, bu) = (a as u64, b as u64);
        let page = PAGE as u64;
        // One buffer over all four pages at `a`, one over the middle two
        // at `b`.
        crate::shadow::register(a, au, 4 * page);
        crate::shadow::register(b, bu + page, 2 * page);
        let whole = crate::shadow::find(au, 4 * page).unwrap();
        let shared = crate::memsys::count(crate::memsys::N::AliasShared);
        let middle = crate::shadow::find(bu + page + 16, 64).unwrap();
        assert!(std::sync::Arc::ptr_eq(&whole, &middle), "one copy of the pages");
        assert!(crate::memsys::count(crate::memsys::N::AliasShared) > shared);
        assert_eq!(middle.gpu(bu + page + 16), whole.gpu(au + page + 16));
        assert_eq!(middle.offset(bu + 2 * page), whole.offset(au + 2 * page));
        assert_eq!(middle.left(bu + page + 16), 2 * page - 16, "to its own buffer's end");
        assert_eq!(middle.page_range(bu + page + 16, 64), (au + page, page), "in the copy's own addresses");
        assert_eq!(whole.shared_at(), [(bu + page, 2 * page, au + page)]);
        assert_eq!(crate::shadow::all().iter().filter(|c| std::sync::Arc::ptr_eq(c, &whole)).count(), 1);
        // The title writes through one address; a list reading through
        // the other sees it.
        let read = |address: u64| whole.read_host(address, 1)[0] & 0xff;
        let run = |through: &std::sync::Arc<crate::shadow::Shadow>, address: u64| {
            let list = host.new_list().unwrap();
            through.refresh(&list, address, page);
            list.close();
            host.execute(&[&list]);
        };
        // SAFETY: bytes of our views.
        unsafe { ((b + PAGE + 16) as *mut u8).write_volatile(0x5a) };
        run(&whole, au + page);
        assert_eq!(read(au + page + 16), 0x5a, "written at b, refreshed at a");
        // SAFETY: as above.
        unsafe { ((a + PAGE + 16) as *mut u8).write_volatile(0x77) };
        run(&middle, bu + page);
        assert_eq!(read(au + page + 16), 0x77, "written at a, refreshed at b");
        // The second page at `b` shows something else: the buffers part.
        crate::watch::title_memory_remapped(b + PAGE, PAGE);
        assert!(whole.shared_at().is_empty());
        let apart = crate::shadow::find(bu + page + 16, 64).unwrap();
        assert!(!std::sync::Arc::ptr_eq(&whole, &apart), "a copy of its own again");
        assert_eq!(apart.range(), (bu + page, 2 * page));
        // A list recorded before the remap still finds its bytes.
        assert_eq!(whole.gpu(bu + page + 16), whole.gpu(au + page + 16));
        crate::shadow::unregister(b, bu + page);
        crate::shadow::unregister(a, au);
        crate::watch::title_memory_remapped(a, len);
        crate::watch::title_memory_remapped(b, len);
        drop((whole, middle, apart));
    }

    /// A buffer released while it shares another's copy leaves that copy
    /// to its own buffer, and one freed takes no copy of other memory
    /// with it.
    #[test]
    fn a_sharing_buffer_released_or_freed_leaves_the_copy() {
        use crate::memsys::alias::ON_HERE;
        use crate::watch::PAGE;
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE};
        if super::host().is_none() {
            return; // no host GPU here
        }
        let len = 2 * PAGE;
        // SAFETY: a pagefile section mapped read-write three times; kept.
        let (a, b, own) = unsafe {
            let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, len as u32, std::ptr::null());
            assert!(!section.is_null());
            let map = || MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize;
            (map(), map(), map())
        };
        assert!(a != 0 && b != 0 && own != 0);
        struct Here;
        impl Drop for Here {
            fn drop(&mut self) {
                ON_HERE.set(false);
            }
        }
        ON_HERE.set(true);
        let _here = Here;
        crate::watch::title_memory_own(a, len, own);
        crate::watch::title_memory_own(b, len, own);
        let (au, bu, n) = (a as u64, b as u64, len as u64);
        crate::shadow::register(a, au, n);
        crate::shadow::register(b, bu, n);
        let copy = crate::shadow::find(au, n).unwrap();
        assert!(std::sync::Arc::ptr_eq(&copy, &crate::shadow::find(bu, n).unwrap()));
        // The memory at `b` is freed: the copy, which is `a`'s, lives on.
        drop(crate::shadow::freed(bu, n));
        assert!(copy.shared_at().is_empty(), "parted");
        assert!(std::sync::Arc::ptr_eq(&copy, &crate::shadow::find(au, n).unwrap()));
        let list = super::host().unwrap().new_list().unwrap();
        // SAFETY: a byte of our view.
        unsafe { (a as *mut u8).write_volatile(0x42) };
        copy.refresh(&list, au, n);
        list.close();
        super::host().unwrap().execute(&[&list]);
        assert_eq!(copy.read_host(au, 1)[0] & 0xff, 0x42, "still refreshed: not gone with b's memory");
        // Shared again, then released: the copy stays its own buffer's.
        crate::shadow::unregister(b, bu);
        crate::shadow::register(b, bu, n);
        assert!(std::sync::Arc::ptr_eq(&copy, &crate::shadow::find(bu, n).unwrap()));
        crate::shadow::unregister(b, bu);
        assert!(copy.shared_at().is_empty());
        assert!(std::sync::Arc::ptr_eq(&copy, &crate::shadow::find(au, n).unwrap()));
        crate::shadow::unregister(a, au);
        crate::watch::title_memory_remapped(a, len);
        crate::watch::title_memory_remapped(b, len);
    }

    /// The buffer whose copy others share is released and made again at
    /// its addresses: those sharing part, and the pages end up in one
    /// copy again, not in the released one and a new one beside it.
    #[test]
    fn a_shared_copy_released_by_its_buffer_is_not_kept_beside_a_new_one() {
        use crate::memsys::alias::ON_HERE;
        use crate::watch::PAGE;
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE};
        if super::host().is_none() {
            return; // no host GPU here
        }
        let len = 2 * PAGE;
        // SAFETY: a pagefile section mapped read-write three times; kept.
        let (a, b, own) = unsafe {
            let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, len as u32, std::ptr::null());
            assert!(!section.is_null());
            let map = || MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize;
            (map(), map(), map())
        };
        assert!(a != 0 && b != 0 && own != 0);
        struct Here;
        impl Drop for Here {
            fn drop(&mut self) {
                ON_HERE.set(false);
            }
        }
        ON_HERE.set(true);
        let _here = Here;
        crate::watch::title_memory_own(a, len, own);
        crate::watch::title_memory_own(b, len, own);
        let (au, bu, n) = (a as u64, b as u64, len as u64);
        crate::shadow::register(a, au, n);
        crate::shadow::register(b, bu, n);
        let old = crate::shadow::find(au, n).unwrap();
        assert!(std::sync::Arc::ptr_eq(&old, &crate::shadow::find(bu, n).unwrap()));
        // A list still holds the copy of `a`'s released buffer.
        crate::shadow::unregister(a, au);
        assert!(old.shared_at().is_empty(), "the buffer sharing it parted");
        crate::shadow::register(a, au, n);
        let at_b = crate::shadow::find(bu, n).unwrap();
        let at_a = crate::shadow::find(au, n).unwrap();
        assert!(!std::sync::Arc::ptr_eq(&old, &at_b), "the released copy is not taken again");
        assert!(std::sync::Arc::ptr_eq(&at_a, &at_b), "one copy of the pages");
        crate::shadow::unregister(a, au);
        crate::shadow::unregister(b, bu);
        crate::watch::title_memory_remapped(a, len);
        crate::watch::title_memory_remapped(b, len);
        drop((old, at_a, at_b));
    }

    /// With the gate on, a submission's write marks are taken on the GPU
    /// as it ends, its marked page goes back from them, and the bitmap is
    /// clear for the next submission.
    #[test]
    fn marks_taken_by_submission_bring_its_page_back() {
        use crate::memsys::{GATE_IN_TESTS, N};
        use crate::watch::PAGE;
        use std::sync::atomic::Ordering::Relaxed;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let _alone = crate::memory::without_a_miss();
        struct Gating;
        impl Drop for Gating {
            fn drop(&mut self) {
                GATE_IN_TESTS.store(false, Relaxed);
            }
        }
        GATE_IN_TESTS.store(true, Relaxed);
        let _gating = Gating;
        let p = crate::watch::tests::pool(2);
        // SAFETY: our pool, not guarded yet.
        unsafe { ((p + PAGE + 16) as *mut u8).write_volatile(0x11) };
        crate::shadow::register(p, p as u64, (2 * PAGE) as u64);
        let copy = crate::shadow::find(p as u64, (2 * PAGE) as u64).unwrap();
        let list = host.new_list().unwrap();
        copy.refresh(&list, p as u64, (2 * PAGE) as u64);
        copy.gpu_writes(&list, p as u64, (2 * PAGE) as u64, true);
        let c = copy.clone();
        list.defer(move || {
            c.write_host((p + PAGE + 16) as u64, &[0xbb]);
            c.mark_page_for_test(p + PAGE, false);
        });
        list.close();
        let taken = crate::memsys::count(N::MarksSnaps);
        host.execute(&[&list]);
        // SAFETY: a byte of our pool.
        let title = unsafe { ((p + PAGE + 16) as *const u8).read_volatile() };
        assert_eq!(title, 0xbb, "the marked page went back");
        if crate::memsys::marks::ready(host) {
            assert!(crate::memsys::count(N::MarksSnaps) > taken, "from marks taken on the GPU");
        }
        assert!(!copy.mark_set_for_test(p + PAGE, false), "and the bitmap is clear");
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }

    /// The lists read a fill's destination when they run, so it is filled
    /// before them even from a range their shaders write.
    #[test]
    fn a_fill_from_a_gpu_written_range_is_copied_before_the_lists_run() {
        use crate::watch::PAGE;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let pool = crate::watch::tests::pool(1);
        // SAFETY: our pool, not guarded yet.
        unsafe { (pool as *mut u8).write_volatile(0xAA) };
        crate::shadow::register(pool, pool as u64, PAGE as u64);
        let copy = crate::shadow::find(pool as u64, PAGE as u64).unwrap();
        let list = host.new_list().unwrap();
        copy.refresh(&list, pool as u64, PAGE as u64);
        copy.gpu_writes(&list, pool as u64, PAGE as u64, false);
        let (_, to) = list.scratch(PAGE as u64).expect("scratch");
        // SAFETY: the list's scratch memory, `PAGE` bytes.
        unsafe { to.write_bytes(0, PAGE) };
        list.defer_fill(to, pool as *const u8, PAGE);
        list.run_deferred(&mut super::Batch::default());
        // SAFETY: as above.
        assert_eq!(unsafe { to.read_volatile() }, 0xAA);
        crate::shadow::unregister(pool, pool as u64);
        drop(copy);
        crate::watch::tests::free(pool);
    }
}

#[cfg(test)]
mod fills {
    use super::{HostList, host};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    const PAGE: usize = 4096;

    fn list() -> Option<HostList> {
        host()?.new_list().ok()
    }

    /// Fills and copies run in recorded order, and a fill's unmapped pages
    /// read as zeros while its mapped ones are copied.
    #[test]
    fn steps_run_in_order_and_unmapped_pages_read_as_zeros() {
        let Some(list) = list() else {
            return; // no host GPU
        };
        // Three reserved pages, the middle one not committed.
        // SAFETY: a fresh reservation; the committed pages are ours to fill.
        let holed = unsafe {
            let at =
                VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_READWRITE).cast::<u8>();
            assert!(!at.is_null());
            for page in [0, 2] {
                let page = at.add(page * PAGE);
                assert!(!VirtualAlloc(page.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
                page.write_bytes(7, PAGE);
            }
            at
        };
        let old = [1u8; 256];
        let new = [2u8; 256];
        let mut title = old;
        let (mut first, mut second) = ([0u8; 256], [0u8; 100]);
        let mut whole = vec![9u8; 3 * PAGE];
        let memory = title.as_mut_ptr();
        list.defer_fill(first.as_mut_ptr(), memory, 256);
        list.defer_fill(whole.as_mut_ptr(), holed, 3 * PAGE);
        // The title's own copy: fills after it see its bytes.
        list.defer_copy(memory, new.as_ptr(), 256);
        list.defer_fill(second.as_mut_ptr(), memory, 100);
        list.run_deferred(&mut super::Batch::default());
        assert_eq!((first, second), (old, [2u8; 100]));
        assert_eq!(title, new);
        assert!(whole[..PAGE].iter().all(|&b| b == 7));
        assert!(whole[PAGE..2 * PAGE].iter().all(|&b| b == 0));
        assert!(whole[2 * PAGE..].iter().all(|&b| b == 7));
        // SAFETY: the reservation above.
        assert_ne!(unsafe { VirtualFree(holed.cast(), 0, MEM_RELEASE) }, 0);
    }

    /// A list's constant buffer fills, as a submission runs them. Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture constant_fills`.
    #[test]
    #[ignore = "benchmark"]
    fn constant_fills_per_list() {
        let Some(list) = list() else {
            return; // no host GPU
        };
        const FILLS: usize = 4096;
        let title: Vec<u8> = (0..FILLS * 512).map(|i| (i * 31 + i / 512) as u8).collect();
        for len in [64usize, 256, 512] {
            list.close();
            list.reset();
            let mut filled = Vec::new();
            for i in 0..FILLS {
                let (_, to) = list.scratch(len as u64).expect("scratch");
                list.defer_fill(to, title[i * 512..].as_ptr(), len);
                filled.push((to, i * 512));
            }
            let mut best = std::time::Duration::MAX;
            for _ in 0..50 {
                let started = std::time::Instant::now();
                list.run_deferred(&mut super::Batch::default());
                best = best.min(started.elapsed());
            }
            for (to, at) in filled {
                // SAFETY: the list's scratch memory.
                let got = unsafe { std::slice::from_raw_parts(to, len) };
                assert_eq!(got, &title[at..at + len]);
            }
            let each = best.as_nanos() as f64 / FILLS as f64;
            println!("{FILLS} fills of {len} B: {best:?} ({each:.1} ns each)");
        }
    }
}
