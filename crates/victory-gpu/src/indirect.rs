//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups in y and z, 2^20 in x and per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

/// The record bounds and the budget's charge, shared by the single clamp and
/// the window's pass so the two cannot drift apart.
macro_rules! clamp_hlsl {
    () => {
        r#"
// The title's records, read only; the clamped ones go to the list's own.
ByteAddressBuffer args : register(t0);
RWByteAddressBuffer dest : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// Groups in x that run as the title asks: its tile lists are a group per
// tile, 67,440 at 2560x1440. D3D12 names 65,535 for a dispatch and does not
// look at indirect counts; an RTX 3080 Ti runs every group (the test
// `a_dispatch_of_more_than_65535_groups_runs_whole`).
static const uint WIDEST = 1u << 20;
// Clamps the record of kind `what` at `from` into `dest` at `to`; its cost.
uint clamped(uint what, uint from, uint to) {
    uint cost;
    if (what <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(from), 1u << 24);
        uint instances = min(args.Load(from + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        dest.Store2(to, uint2(count, instances));
        // The rest of the record as it is: first vertex and instance (4 words
        // in all), or first index, base vertex and first instance (5).
        if (what == 1) {
            dest.Store3(to + 8, args.Load3(from + 8));
        } else {
            dest.Store2(to + 8, args.Load2(from + 8));
        }
    } else if (what == 2) {
        uint3 g = args.Load3(from);
        // Up to WIDEST groups in x run as asked; a record asking for more
        // runs 65,535.
        g = min(g, uint3(g.x <= WIDEST ? WIDEST : 65535u, 65535u, 65535u));
        // x * y fits 32 bits where it is looked at; the whole count may not.
        if ((g.y != 0 && g.x > GROUPS / g.y) || (g.z != 0 && g.x * g.y > GROUPS / g.z)) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        dest.Store3(to, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(from);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        dest.Store3(to, g);
    }
    return cost;
}
// A record past the budget (no cost reaches it).
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
// Charges a tile's costs, `cost` this lane's, against `left` (the first
// lane's): after it `costs[lane]` is PAST where the record does not fit.
void charge(uint lane, uint cost, inout uint left) {
    costs[lane] = cost;
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0) {
        for (uint i = 0; i < 64; i++) {
            if (costs[i] > left) {
                // The first record that does not fit takes what is
                // left, so those after it run empty too.
                costs[i] = PAST;
                left = 0;
            } else {
                left -= costs[i];
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
}
"#
    };
}

const SOURCE: &str = concat!(
    r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), UAV(u1), UAV(u0)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint unused; };
"#,
    clamp_hlsl!(),
    r#"
// One group walks the records in tiles of 64: every lane clamps its own
// record, then the first lane charges the tile's records in order. No
// other group shares the budget: a shared add wraps once enough records
// are rejected, and a compare-exchange retries more the more records fit.
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget.Load(0), BUDGET);
    }
    for (uint tile = 0; tile < records / 64 + min(records % 64, 1u); tile++) {
        uint record = tile * 64 + lane;
        uint at = record * stride;
        uint cost = 0;
        if (record < records) {
            cost = clamped(kind, at, at);
        }
        charge(lane, cost, left);
        if (costs[lane] == PAST) {
            // Draws keep their layout with no vertices; groups become zero.
            dest.Store(at, 0);
        }
    }
    if (lane == 0) {
        budget.Store(0, BUDGET - left);
    }
}
"#
);

/// The window's pass: the single clamp's bounds and budget over the records
/// the table names, one tile of them.
const WINDOW_SOURCE: &str = concat!(
    r#"
#define RS "SRV(t0), SRV(t1), UAV(u1), UAV(u0)"
// The count of records (the first of 4 words), then 4 words per record in
// draw order: offset in `args`, offset in `dest`, kind, unused.
ByteAddressBuffer table : register(t1);
"#,
    clamp_hlsl!(),
    r#"
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget.Load(0), BUDGET);
    }
    uint cost = 0;
    uint to = 0;
    if (lane < table.Load(0)) {
        uint3 record = table.Load3(16 + 16 * lane);
        to = record.y;
        cost = clamped(record.z, record.x, to);
    }
    charge(lane, cost, left);
    if (costs[lane] == PAST) {
        dest.Store(to, 0);
    }
    if (lane == 0) {
        budget.Store(0, BUDGET - left);
    }
}
"#
);

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    make_from(SOURCE)
}

/// A compute pipeline of ours compiled from `source`.
fn make_from(source: &str) -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` (a title
/// buffer's copy, only read) into the list's own argument buffer, left
/// ready to be read as arguments: that buffer, for [`done`] after the
/// execution. Replaces the list's compute root signature and pipeline
/// (callers bind theirs after). `None` if the clamp is not available.
///
/// Title buffers' copies are never transitioned: they are common between
/// submissions and every use promotes them (shader and root views, the
/// input assembler, these reads), with unordered-access barriers for order.
/// Only the list's argument buffer moves, common, unordered access,
/// indirect argument, common: its states are known whatever order lists
/// run in.
// shortcut: a buffer a shader writes through an unordered view and the
// input assembler then reads in one submission is, by the letter of D3D12,
// promoted to a write state the input assembler may not use (the debug
// layer cannot see view accesses; GPU-based validation might); upgrade to
// per-list tracked buffer states if a driver or that validation minds.
pub(crate) fn bound(list: &HostList, gpu: u64, (kind, stride, records): (u32, u32, u32)) -> Option<ID3D12Resource> {
    // An open window holds the argument buffer, in another state.
    end_window(list);
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return None;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let out = list.arguments(u64::from(stride) * u64::from(records.max(1)))?;
    #[cfg(feature = "diagnostics")]
    list.note_clamp();
    let l = &list.list;
    let constants = [kind, stride, records, 0];
    #[cfg(feature = "diagnostics")]
    list.note_host("host-clamp");
    // SAFETY: an open list; our root signature, pipeline and buffers, and
    // a title buffer's copy the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, out.GetGPUVirtualAddress());
        // One group walks every record.
        l.Dispatch(records.min(1), 1, 1);
        // No unordered-access barrier here: the transition orders `out`
        // before its read as arguments, and only the next clamp touches the
        // budget (and `out`) unordered again, behind its own barrier above.
        l.ResourceBarrier(&[transition(
            &out,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        )]);
    }
    Some(out)
}

/// After the execution: the list's argument buffer back to common.
pub(crate) fn done(list: &HostList, arguments: &ID3D12Resource) {
    #[cfg(feature = "diagnostics")]
    list.note_host("host-done");
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            arguments,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

// Tests: the switch on for this thread.
#[cfg(test)]
thread_local!(pub(crate) static FORCE_WINDOWS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) });

/// Whether consecutive indirect draws are clamped a window at a time
/// (`VICTORY_CLAMP_WINDOW=1`; off by default).
///
/// An indirect draw opens a window: one pass is recorded where its single
/// clamp would be, and the indirect draws after it that may join add their
/// records to the pass's table (upload memory the CPU fills while
/// recording; the GPU reads it when the list runs). They have no pass or
/// barrier of their own. The pass charges the list's budget in draw order,
/// as the single clamps do one after another.
///
/// Clamping a record ahead of the window's earlier draws changes nothing
/// while nothing in between writes buffers. So [`end_window`] runs before
/// any pipeline that may store (`crate::draw`), unordered-access clears,
/// host library recordings, single clamps (indirect dispatches among them)
/// and the list's close; a reset forgets the window. Draws and dispatches
/// that store nothing write no buffer, and title barriers only order or
/// move what was already written, so the window stays open across both.
/// Title buffer copies run on the CPU at submission, before the list.
pub(crate) fn windows_on() -> bool {
    #[cfg(test)]
    if FORCE_WINDOWS.get() {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_CLAMP_WINDOW"))
}

/// Records a window clamps at most: one tile of its pass.
const WINDOW_RECORDS: u32 = 64;
/// Bytes of clamped records a window holds at most.
const WINDOW_BYTES: u64 = 1 << 16;
/// A table: 4 words, then 4 per record.
const TABLE_BYTES: u64 = 16 + 16 * WINDOW_RECORDS as u64;

/// A list's open window.
pub(crate) struct Window {
    /// The list's argument buffer, indirect arguments until the window ends.
    out: ID3D12Resource,
    /// The table's CPU address (upload memory of the list).
    table: usize,
    /// The host GPU address of the title buffer's copy the pass reads.
    source: u64,
    records: u32,
    bytes: u64,
}

fn window_clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make_from(WINDOW_SOURCE) {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no windowed argument clamp: each indirect draw clamped on its own");
                None
            }
        })
        .as_ref()
}

/// Ends `list`'s window, if one is open: its argument buffer back to
/// common. Before any command that may write buffers.
pub(crate) fn end_window(list: &HostList) {
    let open = list.clamp_window().take();
    if let Some(open) = open {
        done(list, &open.out);
    }
}

/// The clamped records of an indirect draw of `records` records of `kind`
/// `stride` bytes apart, `offset` bytes into the title buffer's copy at host
/// GPU address `source` (only read), in `list`'s open window or in one
/// opened for it: the argument buffer (not for [`done`]: [`end_window`]
/// returns it), the records' offset in it, and whether a window was opened
/// (its pass replaced the list's compute root signature and pipeline).
/// `None` if the draw cannot be clamped in a window; the caller clamps it on
/// its own ([`bound`]).
pub(crate) fn windowed(
    list: &HostList,
    source: u64,
    offset: u64,
    (kind, stride, records): (u32, u32, u32),
) -> Option<(ID3D12Resource, u64, bool)> {
    let bytes = u64::from(stride) * u64::from(records);
    if !(1..=WINDOW_RECORDS).contains(&records)
        || !source.is_multiple_of(4)
        || !offset.is_multiple_of(4)
        || !stride.is_multiple_of(4)
        || bytes > WINDOW_BYTES
        || offset.checked_add(bytes)? > u64::from(u32::MAX)
    {
        return None;
    }
    let fits = list.clamp_window().as_ref().is_some_and(|w| {
        w.source == source && w.records + records <= WINDOW_RECORDS && w.bytes + bytes <= WINDOW_BYTES
    });
    if !fits {
        end_window(list);
        let new = open(list, source)?;
        *list.clamp_window() = Some(new);
    }
    let mut window = list.clamp_window();
    let w = window.as_mut()?;
    let at = w.bytes;
    let table = w.table as *mut u32;
    for i in 0..records {
        let record = [offset as u32 + i * stride, at as u32 + i * stride, kind, 0];
        // SAFETY: at most `WINDOW_RECORDS` records, inside the table in the
        // list's upload memory, alive until the list is reset; the GPU reads
        // it only once the list runs.
        unsafe { table.add(4 + 4 * (w.records + i) as usize).cast::<[u32; 4]>().write_volatile(record) };
    }
    w.records += records;
    w.bytes += bytes;
    // SAFETY: the table's first word, as above.
    unsafe { table.write_volatile(w.records) };
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::ClampWindowDraws, 1);
    Some((w.out.clone(), at, !fits))
}

/// Records a window's pass in `list`, reading records from `source`.
fn open(list: &HostList, source: u64) -> Option<Window> {
    let (Some(c), Some(budget)) = (window_clamp(), list.budget()) else {
        return None;
    };
    let out = list.arguments(WINDOW_BYTES)?;
    let (table_gpu, table) = list.scratch(TABLE_BYTES)?;
    #[cfg(feature = "diagnostics")]
    {
        list.note_clamp();
        crate::copystat::add(crate::copystat::C::ClampWindows, 1);
    }
    let l = &list.list;
    #[cfg(feature = "diagnostics")]
    list.note_host("host-clamp");
    // SAFETY: an open list; our root signature, pipeline and buffers, and
    // a title buffer's copy the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRootShaderResourceView(0, source);
        l.SetComputeRootShaderResourceView(1, table_gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, out.GetGPUVirtualAddress());
        l.Dispatch(1, 1, 1);
        l.ResourceBarrier(&[transition(
            &out,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        )]);
    }
    Some(Window { out, table: table as usize, source, records: 0, bytes: 0 })
}

/// Opens a window in `list` over a buffer of ours (tests of what ends one).
#[cfg(test)]
pub(crate) fn open_for_test(list: &HostList) {
    let (buffer, _) = host().unwrap().shared_buffer(256).unwrap();
    list.keep(&buffer);
    // SAFETY: a live buffer the list holds.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    windowed(list, gpu, 0, (0, 16, 1)).expect("a window");
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    #[cfg(test)]
    tests::UAV_BARRIERS.with(|n| n.set(n.get() + 1));
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    thread_local! {
        /// The unordered-access barriers built on this thread.
        pub(super) static UAV_BARRIERS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// Clamps `records` records of `kind`, filled by repeating `words`,
    /// `calls` times in one list: each record's first word in the list's
    /// argument buffer after (zero when it runs empty). The title's copy,
    /// one record more included, stays as written. `None` without a host
    /// GPU.
    fn clamped(kind: u32, words: &[u32], records: u32, calls: u32) -> Option<Vec<u32>> {
        let host = super::host()?;
        // Words in a record of `kind`.
        let per = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let spec = (kind, per as u32 * 4, records);
        let all = per * (records as usize + 1);
        let (buffer, data) = host.shared_buffer(all as u64 * 4).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                data.add(at).write_volatile(words[at % words.len()]);
            }
            let gpu = buffer.GetGPUVirtualAddress();
            let mut out = None;
            for _ in 0..calls {
                let clamped = super::bound(&list, gpu, spec).unwrap();
                super::done(&list, &clamped);
                out = Some(clamped);
            }
            list.close();
            host.execute(&[&list]);
            for at in 0..all {
                assert_eq!(data.add(at).read_volatile(), words[at % words.len()], "the title's copy as written");
            }
            let mut out_data = std::ptr::null_mut();
            out?.Map(0, None, Some(&mut out_data)).unwrap();
            let first = |i: usize| out_data.cast::<u32>().add(i * per).read_volatile();
            Some((0..records as usize).map(first).collect())
        }
    }

    /// The title's tile lists are dispatches of a group per tile, more than
    /// 65,535 of them at 2560x1440: every group runs once, under its number.
    #[test]
    fn a_dispatch_of_more_than_65535_groups_runs_whole() {
        use windows::Win32::Graphics::Direct3D12::*;
        const WORKER: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer seen : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main(uint3 group : SV_GroupID) {
    uint was;
    seen.InterlockedAdd(0, 1, was);
    seen.InterlockedMax(4, group.x, was);
}
"#;
        const GROUPS: u32 = 67_440;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let worker = super::make_from(WORKER).unwrap();
        let (title, data) = host.shared_buffer(24).unwrap();
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 12,
            NumArgumentDescs: 1,
            pArgumentDescs: &argument,
            NodeMask: 0,
        };
        let mut signature: Option<ID3D12CommandSignature> = None;
        // SAFETY: a valid single-argument signature.
        unsafe { host.device.CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut signature) }.unwrap();
        let (holder, list) = (host.new_list().unwrap(), host.new_list().unwrap());
        let seen = holder.arguments(16).unwrap();
        let mut at = std::ptr::null_mut();
        // SAFETY: our mapped buffers; an open list and live objects.
        unsafe {
            data.cast::<[u32; 3]>().write_volatile([GROUPS, 1, 1]);
            seen.Map(0, None, Some(&mut at)).unwrap();
            at.cast::<[u32; 2]>().write_volatile([0, 0]);
            let arguments = super::bound(&list, title.GetGPUVirtualAddress(), (2, 12, 1)).unwrap();
            list.list.SetComputeRootSignature(&worker.root);
            list.list.SetPipelineState(&worker.pso);
            list.list.SetComputeRootUnorderedAccessView(0, seen.GetGPUVirtualAddress());
            list.list.ExecuteIndirect(signature.as_ref().unwrap(), 1, &arguments, 0, None, 0);
            super::done(&list, &arguments);
            list.close();
            host.execute(&[&list]);
            assert_eq!(at.cast::<[u32; 2]>().read_volatile(), [GROUPS, GROUPS - 1]);
        }
    }

    /// Rejected records do not wrap the budget: of 128 garbage draws (2^26
    /// vertex instances each once clamped) two run, where a wrapping add
    /// let four; of 600 garbage mesh dispatches 16, not 31.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        let Some(draws) = clamped(0, &[u32::MAX], 128, 1) else {
            return; // no host GPU here
        };
        assert_eq!(draws.iter().filter(|&&count| count != 0).count(), 2);
        let meshes = clamped(10, &[u32::MAX], 600, 1).unwrap();
        assert_eq!(meshes.iter().filter(|&&groups| groups != 0).count(), 16);
    }

    /// The budget goes to the first records, in order: of 200 draws of
    /// 3 * 2^18 vertices the first 170 run (it ends inside the third tile);
    /// small records that all fit all run.
    #[test]
    fn the_budget_goes_to_the_first_records() {
        let Some(draws) = clamped(0, &[3 << 18, 1, 0, 0], 200, 1) else {
            return; // no host GPU here
        };
        assert!(draws[..170].iter().all(|&count| count != 0));
        assert!(draws[170..].iter().all(|&count| count == 0));
        let small = clamped(0, &[1], 1 << 16, 1).unwrap();
        assert!(small.iter().all(|&count| count == 1));
    }

    /// A record that does not fit ends the budget: after two draws of
    /// 3 * 2^24 vertex instances a third takes what is left, and the small
    /// ones after it run empty.
    #[test]
    fn a_record_that_does_not_fit_ends_the_budget() {
        let mut words = [1 << 24, 3, 0, 0].repeat(3);
        words.extend([3, 1, 0, 0].repeat(6));
        let Some(draws) = clamped(0, &words, 8, 1) else {
            return; // no host GPU here
        };
        let ran: Vec<bool> = draws.iter().map(|&count| count != 0).collect();
        assert_eq!(ran, [true, true, false, false, false, false, false, false]);
    }

    /// Two lists clamping into their own argument buffers, run in either
    /// order: every barrier starts from the buffer's current state (the
    /// debug layer's rule; runs light19 and light21 had the title's GPU
    /// heap moved to unordered access under the input assembler), the
    /// title's copy is never transitioned, and all end common.
    #[test]
    fn argument_buffers_move_from_their_current_states() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: our mapped buffer; a live buffer.
        let gpu = unsafe {
            data.cast::<[u32; 3]>().write_volatile([(1 << 20) + 1, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        for a_first in [true, false] {
            let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
            issued();
            let recorded: Vec<_> = lists
                .iter()
                .map(|l| {
                    let out = super::bound(l, gpu, (2, 12, 1)).unwrap();
                    super::done(l, &out);
                    l.close();
                    (out, issued())
                })
                .collect();
            let order = if a_first { [0, 1] } else { [1, 0] };
            host.execute(&[&lists[order[0]], &lists[order[1]]]);
            let mut run = issued();
            run.extend(order.iter().flat_map(|&i| recorded[i].1.clone()));
            let mut states: std::collections::HashMap<usize, u32> =
                [key(&title), key(&recorded[0].0), key(&recorded[1].0)].into_iter().map(|k| (k, 0)).collect();
            replay(&run, &mut states).unwrap();
            assert!(run.iter().all(|t| t.0 != key(&title)), "the title's copy is never transitioned");
            assert!(states.values().all(|&s| s == 0), "all common again");
            for (out, _) in &recorded {
                let mut p = std::ptr::null_mut();
                // SAFETY: a mapped buffer of ours, the lists have run.
                let clamped = unsafe {
                    out.Map(0, None, Some(&mut p)).unwrap();
                    p.cast::<[u32; 3]>().read_volatile()
                };
                assert_eq!(clamped, [65_535, 1, 1]);
            }
            // SAFETY: as above.
            assert_eq!(unsafe { data.cast::<[u32; 3]>().read_volatile() }, [(1 << 20) + 1, 1, 1], "the title's copy as written");
        }
    }

    /// A clamp records one unordered-access barrier, ahead of its dispatch:
    /// it orders the title's writes and the previous clamp's budget.
    #[test]
    fn a_clamp_records_one_uav_barrier() {
        let before = UAV_BARRIERS.get();
        let Some(twice) = clamped(0, &[1], 1, 2) else {
            return; // no host GPU here
        };
        assert_eq!(UAV_BARRIERS.get() - before, 2);
        assert_eq!(twice, [1]);
    }

    /// What a test list records, in order.
    #[derive(Clone, Copy)]
    enum Step {
        /// An indirect draw: title buffer, byte offset, kind, records.
        Draw(usize, u64, u32, u32),
        /// A shader of the title writes a value at a byte offset of a title
        /// buffer, after the window ended as before any storing pipeline.
        Write(usize, u64, u32),
    }

    const WRITE: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), UAV(u0)"
cbuffer Word : register(b0) { uint at; uint value; };
RWByteAddressBuffer title : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() { title.Store(at, value); }
"#;

    fn stride(kind: u32) -> u32 {
        match kind {
            0 => 16,
            1 => 20,
            _ => 12,
        }
    }

    /// Records `steps` into one list over title buffers holding `init`, each
    /// draw clamped on its own or (`window`) as `execute_indirect` does with
    /// the switch on, and runs it: each draw's clamped words, and how many
    /// windows were opened. `None` without a host GPU.
    fn run(init: &[Vec<u32>], steps: &[Step], window: bool) -> Option<(Vec<Vec<u32>>, u32)> {
        use super::{D3D12_RESOURCE_STATE_COPY_SOURCE as COPY_SOURCE, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT as ARGUMENTS, transition};
        let host = super::host()?;
        let titles: Vec<_> = init
            .iter()
            .map(|words| {
                let (buffer, data) = host.shared_buffer(words.len() as u64 * 4).unwrap();
                // SAFETY: our mapped buffer, written before the list runs; a
                // live buffer.
                let gpu = unsafe {
                    std::ptr::copy_nonoverlapping(words.as_ptr(), data.cast::<u32>(), words.len());
                    buffer.GetGPUVirtualAddress()
                };
                (buffer, gpu)
            })
            .collect();
        let (results, data) = host.shared_buffer(1 << 20).unwrap();
        let writer = super::make_from(WRITE).unwrap();
        let list = host.new_list().unwrap();
        // Each draw's records: offset in `results`, words.
        let mut placed = Vec::new();
        let (mut at_results, mut windows) = (0u64, 0);
        for &step in steps {
            match step {
                Step::Draw(buffer, offset, kind, records) => {
                    let spec = (kind, stride(kind), records);
                    let len = u64::from(stride(kind)) * u64::from(records);
                    let source = titles[buffer].1;
                    let joined = if window { super::windowed(&list, source, offset, spec) } else { None };
                    let single = joined.is_none();
                    let (out, at) = match joined {
                        Some((out, at, opened)) => {
                            windows += u32::from(opened);
                            (out, at)
                        }
                        None => (super::bound(&list, source + offset, spec).unwrap(), 0),
                    };
                    // Copied at once, in a window too: its pass runs before
                    // this, with the whole table.
                    // SAFETY: an open list, live buffers.
                    unsafe {
                        list.list.ResourceBarrier(&[transition(&out, ARGUMENTS, COPY_SOURCE)]);
                        list.list.CopyBufferRegion(&results, at_results, &out, at, len);
                        list.list.ResourceBarrier(&[transition(&out, COPY_SOURCE, ARGUMENTS)]);
                    }
                    if single {
                        super::done(&list, &out);
                    }
                    placed.push((at_results, len as usize / 4));
                    at_results += len.next_multiple_of(256);
                }
                Step::Write(buffer, at, value) => {
                    super::end_window(&list);
                    let constants = [at as u32, value];
                    // SAFETY: an open list; our pipeline and a title buffer.
                    unsafe {
                        list.list.SetComputeRootSignature(&writer.root);
                        list.list.SetPipelineState(&writer.pso);
                        list.list.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                        list.list.SetComputeRootUnorderedAccessView(1, titles[buffer].1);
                        list.list.Dispatch(1, 1, 1);
                    }
                }
            }
        }
        list.close();
        host.execute(&[&list]);
        let words = |&(at, n): &(u64, usize)| {
            // SAFETY: inside our mapped buffer; the list has run.
            unsafe { std::slice::from_raw_parts(data.add(at as usize).cast::<u32>(), n).to_vec() }
        };
        Some((placed.iter().map(words).collect(), windows))
    }

    /// The windows' records, after checking they are the single clamps' word
    /// for word, and the windows opened.
    fn same(init: &[Vec<u32>], steps: &[Step]) -> Option<(Vec<Vec<u32>>, u32)> {
        let (single, _) = run(init, steps, false)?;
        let (windowed, windows) = run(init, steps, true)?;
        assert_eq!(single.len(), windowed.len());
        for (k, (a, b)) in single.iter().zip(&windowed).enumerate() {
            assert_eq!(a, b, "draw {k}");
        }
        Some((windowed, windows))
    }

    /// All kinds from two buffers and draws of several records, one of them
    /// too many to join: the windows clamp what the single clamps do.
    #[test]
    fn windows_clamp_as_single_clamps_do() {
        let mut a: Vec<u32> = (0..140u32).flat_map(|i| [100 + i, 1, 1]).collect();
        let draws = a.len() as u64 * 4;
        a.extend((0..8u32).flat_map(|i| [3 * i + 1, 2, i, 7]));
        let indexed = a.len() as u64 * 4;
        a.extend((0..8u32).flat_map(|i| [6 * i, 1 + i, 7, 1, i]));
        let dispatch = a.len() as u64 * 4;
        a.extend([3, 2, 4]);
        let b: Vec<u32> = (0..64u32).flat_map(|i| [70 + i, 2, 3]).collect();
        let mut steps: Vec<Step> = (0..70).map(|i| Step::Draw(0, 12 * i, 10, 1)).collect();
        steps.extend([
            Step::Draw(0, draws, 0, 8),
            Step::Draw(0, indexed, 1, 8),
            Step::Draw(0, dispatch, 2, 1),
            Step::Draw(1, 0, 10, 40),
            Step::Draw(1, 480, 10, 24),
            Step::Draw(0, 0, 10, 65),
            Step::Draw(0, 780, 10, 60),
            Step::Draw(0, 1500, 10, 10),
        ]);
        let Some((out, windows)) = same(&[a, b], &steps) else {
            return; // no host GPU here
        };
        assert!(windows > 3);
        assert_eq!(out[0], [100, 1, 1], "a record as it is");
    }

    /// Every fifth record garbage: the budget runs out in the second window
    /// and the third finds it spent, as with single clamps.
    #[test]
    fn the_budget_carries_across_windows() {
        let a: Vec<u32> = (0..192u32).flat_map(|i| if i % 5 == 3 { [u32::MAX; 3] } else { [1, 1, 1] }).collect();
        let steps: Vec<Step> = (0..192).map(|i| Step::Draw(0, 12 * i, 10, 1)).collect();
        let Some((out, windows)) = same(&[a], &steps) else {
            return; // no host GPU here
        };
        assert_eq!(windows, 3);
        assert!(out[..64].iter().all(|r| r[0] != 0), "the first window fits");
        assert!(out[64..128].iter().any(|r| r[0] != 0) && out[64..128].iter().any(|r| r[0] == 0));
        assert!(out[128..].iter().all(|r| r[0] == 0));
    }

    /// A shader writing a record between two draws, the window ended: the
    /// draw after it reads the record written, as clamped on its own.
    #[test]
    fn a_draw_after_an_ending_store_reads_it() {
        let a = [5, 1, 1].repeat(4);
        let steps = [Step::Draw(0, 0, 10, 1), Step::Write(0, 12, 9), Step::Draw(0, 12, 10, 1)];
        let Some((out, windows)) = same(&[a], &steps) else {
            return; // no host GPU here
        };
        assert_eq!((out[1].as_slice(), windows), ([9, 1, 1].as_slice(), 2));
    }

    /// 64 records share a window; the 65th opens the next, as does another
    /// source buffer. Larger draws stay single.
    #[test]
    fn the_65th_record_opens_a_new_window() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, _) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: a live buffer.
        let gpu = unsafe { title.GetGPUVirtualAddress() };
        let list = host.new_list().unwrap();
        let draw = |source, records| super::windowed(&list, source, 0, (10, 12, records)).map(|(_, at, opened)| (at, opened));
        assert_eq!(draw(gpu, 1), Some((0, true)));
        for i in 1..64 {
            assert_eq!(draw(gpu, 1), Some((12 * i, false)));
        }
        assert_eq!(draw(gpu, 1), Some((0, true)));
        assert_eq!(draw(gpu + 256, 1), Some((0, true)));
        assert_eq!(draw(gpu + 256, 64), Some((0, true)));
        assert_eq!(draw(gpu, 65), None);
        assert_eq!(draw(gpu, 0), None);
        list.close();
    }

    /// A single clamp (an indirect dispatch's too) ends the window, and so
    /// does the list's close; after a reset the next draw opens one.
    #[test]
    fn single_clamps_close_and_reset_end_the_window() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, _) = host.shared_buffer(256).unwrap();
        // SAFETY: a live buffer.
        let gpu = unsafe { title.GetGPUVirtualAddress() };
        let list = host.new_list().unwrap();
        super::open_for_test(&list);
        let out = super::bound(&list, gpu, (2, 12, 1)).unwrap();
        assert!(list.clamp_window().is_none());
        super::done(&list, &out);
        super::open_for_test(&list);
        list.close();
        assert!(list.clamp_window().is_none());
        host.execute(&[&list]);
        list.reset();
        assert!(super::windowed(&list, gpu, 0, (10, 12, 1)).unwrap().2);
        list.close();
    }

    /// Windows keep the argument buffer's states right: indirect arguments
    /// while open, common when the list closes, whatever order lists run in.
    #[test]
    fn windows_leave_the_argument_buffer_common() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: our mapped buffer; a live buffer.
        let gpu = unsafe {
            data.cast::<[u32; 6]>().write_volatile([3, 1, 1, 4, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
        issued();
        let recorded: Vec<_> = lists
            .iter()
            .map(|l| {
                let (out, _, _) = super::windowed(l, gpu, 0, (10, 12, 1)).unwrap();
                let (_, at, opened) = super::windowed(l, gpu, 12, (10, 12, 1)).unwrap();
                assert!(!opened && at == 12, "the second draw joins");
                l.close();
                (out, issued())
            })
            .collect();
        host.execute(&[&lists[1], &lists[0]]);
        let mut run = issued();
        run.extend(recorded[1].1.clone());
        run.extend(recorded[0].1.clone());
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        let mut states: std::collections::HashMap<usize, u32> =
            [key(&recorded[0].0), key(&recorded[1].0)].into_iter().map(|k| (k, 0)).collect();
        replay(&run, &mut states).unwrap();
        assert!(states.values().all(|&s| s == 0), "all common again");
        for (out, _) in &recorded {
            let mut p = std::ptr::null_mut();
            // SAFETY: a mapped buffer of ours, the lists have run.
            let clamped = unsafe {
                out.Map(0, None, Some(&mut p)).unwrap();
                p.cast::<[u32; 6]>().read_volatile()
            };
            assert_eq!(clamped, [3, 1, 1, 4, 1, 1]);
        }
    }

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let Some(twice) = clamped(1, &[u32::MAX], 64, 2) else {
            return; // no host GPU here
        };
        assert!(twice.iter().all(|&count| count == 0));
    }
}
