//! Diagnostics: where the GPU's time goes.
//!
//! A normal run (no `VICTORY_GPU_TIMING`) puts a timestamp before and after
//! each submission of title lists, from two command lists recorded once, and
//! logs the GPU's busy time per frame against the time spent waiting for it.
//! `VICTORY_PIPELINE_STATS` counts each title list's pipeline statistics.

use crate::host::HostList;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;

/// A query heap of `count` queries and a read-back buffer of `bytes` bytes.
pub(crate) fn queries(
    device: &ID3D12Device,
    kind: D3D12_QUERY_HEAP_TYPE,
    count: u32,
    bytes: u64,
) -> Option<(ID3D12QueryHeap, ID3D12Resource)> {
    let mut heap: Option<ID3D12QueryHeap> = None;
    let mut back: Option<ID3D12Resource> = None;
    let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    // SAFETY: plain object creation.
    unsafe {
        device
            .CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: kind, Count: count, NodeMask: 0 }, &mut heap)
            .ok()?;
        device
            .CreateCommittedResource(
                &readback,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut back,
            )
            .ok()?;
    }
    Some((heap?, back?))
}

/// The first `count` values (at most `N`, the rest zero) of a read-back
/// buffer a finished list resolved into.
fn read<const N: usize>(back: &ID3D12Resource, count: usize) -> Option<[u64; N]> {
    let count = count.min(N);
    let mut values = [0u64; N];
    let mut data = std::ptr::null_mut();
    let range = D3D12_RANGE { Begin: 0, End: count * 8 };
    // SAFETY: the buffer holds at least `count` values and is mapped while
    // they are copied out.
    unsafe {
        back.Map(0, Some(&range), Some(&mut data)).ok()?;
        std::ptr::copy_nonoverlapping(data.cast::<u64>(), values.as_mut_ptr(), count);
        back.Unmap(0, None);
    }
    Some(values)
}

/// The two lists around a submission. Every submission is waited for before
/// the next, so one pair serves them all.
pub(crate) struct Clock {
    _allocators: [ID3D12CommandAllocator; 2],
    pub(crate) begin: ID3D12CommandList,
    pub(crate) end: ID3D12CommandList,
    back: ID3D12Resource,
    _heap: ID3D12QueryHeap,
    frequency: u64,
}

/// Since the last log: GPU busy microseconds, wall microseconds waiting for
/// the same submissions, submissions measured.
static BUSY: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

impl Clock {
    pub(crate) fn new(device: &ID3D12Device, queue: &ID3D12CommandQueue) -> Option<Self> {
        let (heap, back) = queries(device, D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2, 16)?;
        let record = |last: bool| -> windows::core::Result<(ID3D12CommandAllocator, ID3D12CommandList)> {
            // SAFETY: plain object creation; a new list is open, and the
            // heap and buffer live as long as it.
            unsafe {
                let allocator: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList =
                    device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                list.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, u32::from(last));
                if last {
                    list.ResolveQueryData(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &back, 0);
                }
                list.Close()?;
                Ok((allocator, list.cast()?))
            }
        };
        let (a, begin) = record(false).ok()?;
        let (b, end) = record(true).ok()?;
        // SAFETY: a plain query.
        let frequency = unsafe { queue.GetTimestampFrequency() }.ok()?;
        Some(Clock { _allocators: [a, b], begin, end, back, _heap: heap, frequency })
    }

    /// Adds the finished submission's GPU time, waited for during `wall`.
    pub(crate) fn add(&self, wall: Duration) {
        let Some([begin, end]) = read::<2>(&self.back, 2) else { return };
        BUSY[0].fetch_add(micros(end.saturating_sub(begin), self.frequency), Ordering::Relaxed);
        BUSY[1].fetch_add(wall.as_micros() as u64, Ordering::Relaxed);
        BUSY[2].fetch_add(1, Ordering::Relaxed);
    }
}

/// `ticks` of a clock of `frequency` ticks per second, in microseconds.
fn micros(ticks: u64, frequency: u64) -> u64 {
    (u128::from(ticks) * 1_000_000 / u128::from(frequency.max(1))) as u64
}

/// The text a timed span is shown with: its command's note. With more notes
/// than `timed` spans, the last span runs to the list's end and says so.
pub(crate) fn span_note(span: usize, timed: usize, notes: usize, note: &str) -> String {
    let note: String = note.chars().take(300).collect();
    if span + 1 == timed && notes > timed {
        format!("untimed-tail ({} commands past the stamp cap) from: {note}", notes - timed)
    } else {
        note
    }
}

/// Whether `VICTORY_PIPELINE_STATS` is on.
pub(crate) fn stats_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_PIPELINE_STATS"))
}

/// The values of `D3D12_QUERY_DATA_PIPELINE_STATISTICS1`; the older form is
/// its first [`OLD_VALUES`].
const VALUES: usize = 14;
const OLD_VALUES: usize = 11;
const VS: usize = 2;
const PRIMITIVES: usize = 6;
const PS: usize = 7;
const CS: usize = 10;
const MS: usize = 12;
const MS_PRIMITIVES: usize = 13;

/// Since the last log: the statistics summed, and the lists they came from.
static TOTALS: Mutex<([u64; VALUES], u64)> = Mutex::new(([0; VALUES], 0));

/// One list's pipeline statistics query.
pub(crate) struct ListStats {
    heap: ID3D12QueryHeap,
    back: ID3D12Resource,
    kind: D3D12_QUERY_TYPE,
    values: usize,
}

impl ListStats {
    pub(crate) fn new(device: &ID3D12Device) -> Option<Self> {
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
        // SAFETY: the query writes its own structure of the given size.
        let asked = unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS9,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
        };
        let mesh = asked.is_ok() && options.MeshShaderPipelineStatsSupported.as_bool();
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::info!(mesh, "pipeline statistics per list (mesh counts if supported)"));
        let (heap_kind, kind, values) = if mesh {
            (D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS1, D3D12_QUERY_TYPE_PIPELINE_STATISTICS1, VALUES)
        } else {
            (D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, OLD_VALUES)
        };
        let (heap, back) = queries(device, heap_kind, 1, values as u64 * 8)?;
        Some(ListStats { heap, back, kind, values })
    }

    /// At the start of `list`'s recording.
    pub(crate) fn begin(&self, list: &ID3D12GraphicsCommandList) {
        // SAFETY: an open list and our heap.
        unsafe { list.BeginQuery(&self.heap, self.kind, 0) };
    }

    /// Before `list` closes.
    pub(crate) fn end(&self, list: &ID3D12GraphicsCommandList) {
        // SAFETY: an open list in which the query began; our buffer.
        unsafe {
            list.EndQuery(&self.heap, self.kind, 0);
            list.ResolveQueryData(&self.heap, self.kind, 0, 1, &self.back, 0);
        }
    }

    /// The counts of the list's last finished run.
    fn counts(&self) -> Option<[u64; VALUES]> {
        read(&self.back, self.values)
    }
}

/// Adds the counts of the lists of a finished submission.
pub(crate) fn add_statistics(runs: &[&HostList]) {
    if !stats_on() {
        return;
    }
    let mut totals = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    for counts in runs.iter().filter_map(|l| l.statistics()?.counts()) {
        for (total, n) in totals.0.iter_mut().zip(counts) {
            *total = total.saturating_add(n);
        }
        totals.1 += 1;
    }
}

/// Logs and clears what the last `frames` frames added up.
pub(crate) fn log(frames: f64) {
    let [busy, wall, submissions] = [0, 1, 2].map(|i| BUSY[i].swap(0, Ordering::Relaxed) as f64);
    if submissions > 0.0 {
        tracing::info!(
            "GPU busy {:.1} ms/frame (timestamps around title submissions), {:.1} ms/frame of wall time \
             on the GPU for the same {:.1} submissions/frame",
            busy / 1e3 / frames,
            wall / 1e3 / frames,
            submissions / frames
        );
    }
    let (totals, lists) = std::mem::replace(
        &mut *TOTALS.lock().unwrap_or_else(|e| e.into_inner()),
        ([0; VALUES], 0),
    );
    if lists > 0 {
        let [ps, cs, vs, ms, mesh_primitives, primitives] =
            [PS, CS, VS, MS, MS_PRIMITIVES, PRIMITIVES].map(|i| totals[i] as f64 / frames);
        tracing::info!(
            "pipeline statistics/frame over {:.1} lists: PS invocations {ps:.0}, CS invocations {cs:.0}, \
             VS invocations {vs:.0}, MS invocations {ms:.0}, mesh primitives {mesh_primitives:.0}, \
             rasterizer primitives {primitives:.0}",
            lists as f64 / frames
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_become_microseconds() {
        assert_eq!(micros(25_000_000, 10_000_000), 2_500_000);
        assert_eq!(micros(u64::MAX, u64::MAX), 1_000_000);
        assert_eq!(micros(5, 0), 5_000_000);
    }

    #[test]
    fn only_the_last_span_of_a_capped_list_says_so() {
        assert_eq!(span_note(0, 2, 2, "a"), "a");
        assert_eq!(span_note(1, 2, 2, "b"), "b");
        assert_eq!(span_note(0, 2, 5, "a"), "a");
        assert_eq!(span_note(1, 2, 5, "b"), "untimed-tail (3 commands past the stamp cap) from: b");
    }

    #[test]
    fn a_dispatch_is_timed_and_counted() {
        let Some(host) = crate::host::host() else { return };
        let stats = ListStats::new(&host.device).expect("a statistics query");
        let (buffer, _) = host.shared_buffer(64).unwrap();
        let list = host.new_list().unwrap();
        // With the switch on, the list has a query of this kind running.
        if stats_on() {
            return;
        }
        stats.begin(&list.list);
        // The argument clamp: a dispatch of ours.
        // SAFETY: a live buffer.
        let records = unsafe { buffer.GetGPUVirtualAddress() };
        let out = crate::indirect::bound(&list, records, (2, 12, 1)).expect("the clamp");
        crate::indirect::done(&list, &out);
        stats.end(&list.list);
        list.close();
        let before = BUSY[2].load(Ordering::Relaxed);
        host.execute(&[&list]);
        assert!(stats.counts().unwrap()[CS] >= 1);
        // With timing mode on, lists are timed one by one instead.
        if !crate::switch::on("VICTORY_GPU_TIMING") {
            assert!(BUSY[2].load(Ordering::Relaxed) > before);
        }
    }
}
