//! Which marked pages shaders really changed, found on the GPU.
//!
//! After a submission every page its shaders marked as written comes
//! back whole from video memory into its copy's mirror, over the bus, and
//! is then read from title memory, compared and written back by the CPU.
//! Most marked pages are marked every frame (about 2,000 of 2,400 in the
//! world), and a third of them hold no changed byte: the shader stored
//! what was there.
//!
//! Each page that went back keeps a twin in video memory: the page as it
//! was when it last went back. At the next submission that marks it, a
//! compute pass compares the page with its twin, both in video memory, a
//! bit per [`BLOCK`] bytes that differ. A page with no bit set holds what
//! it held when it last went back: there is nothing to bring back or to
//! write, so it is neither pulled nor compared on the CPU. The others go
//! the old way, whole. The twins are brought up to date in the same list,
//! after the compares.
//!
//! The compare errs to "changed": bytes the title wrote and a refresh
//! copied into video memory differ from the twin too, and a page with no
//! twin yet, not wholly inside its copy, or held back from title memory,
//! is changed.
//!
//! `VICTORY_MEMSYS_DIFF=verify` (with `VICTORY_MEMSYS=1`) runs the pass
//! and counts, but skips nothing: pages it would have skipped still go
//! the old way, and one of them that then writes bytes to title memory is
//! counted as lost. `VICTORY_MEMSYS_DIFF=1` skips. Off when unset: the
//! twins take [`TWIN_MB`] MiB of video memory.
//! shortcut: one dispatch per page; batch them per copy if recording
//! thousands of dispatches shows in the timing.
//! shortcut: the bits per block only say whether a page changed; pulling
//! and comparing by block needs the mirror to equal the twin where no bit
//! is set, which nothing keeps yet.

use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{Host, transition};
use crate::shadow::Shadow;

/// Bytes of a page.
const PAGE: u64 = 64 << 10;
/// Bytes a bit of a page's mask stands for.
pub(crate) const BLOCK: u64 = 256;
/// Bytes of a page's mask (a bit per block).
const MASK: u64 = PAGE / BLOCK / 8;
/// Video memory for twins unless `VICTORY_MEMSYS_DIFF_MB` says otherwise:
/// room for a world frame's marked pages.
const TWIN_MB: u64 = 192;

/// What the pass does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Off,
    /// Compares and counts; every page still goes the old way.
    Verify,
    /// Pages equal to their twins are not pulled or copied back.
    On,
}

fn mode_of(value: Option<&str>) -> Mode {
    match value {
        None | Some("0") => Mode::Off,
        Some("verify") => Mode::Verify,
        Some(_) => Mode::On,
    }
}

/// `VICTORY_MEMSYS_DIFF` under `VICTORY_MEMSYS=1`. Asked once.
pub(crate) fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| {
        if !super::on() {
            return Mode::Off;
        }
        let mode = mode_of(std::env::var("VICTORY_MEMSYS_DIFF").ok().as_deref());
        tracing::info!(?mode, "memory system: marked pages compared with twins on the GPU (VICTORY_MEMSYS_DIFF)");
        mode
    })
}

/// Which slot holds which key's twin: slots are given out in turn, the
/// oldest twin making room once all are taken.
struct Twins<K> {
    slots: Vec<Option<K>>,
    index: crate::fx::Map<K, u32>,
    next: usize,
}

impl<K: Copy + Eq + std::hash::Hash> Twins<K> {
    fn new(slots: usize) -> Self {
        Self { slots: vec![None; slots.max(1)], index: crate::fx::Map::default(), next: 0 }
    }

    fn find(&self, key: &K) -> Option<u32> {
        self.index.get(key).copied()
    }

    /// The slot for `key`'s twin: its own if it has one, else the next in
    /// turn, whose twin is forgotten.
    fn assign(&mut self, key: K) -> u32 {
        if let Some(slot) = self.find(&key) {
            return slot;
        }
        let slot = self.next;
        self.next = (self.next + 1) % self.slots.len();
        if let Some(old) = self.slots[slot].replace(key) {
            self.index.remove(&old);
        }
        self.index.insert(key, slot as u32);
        slot as u32
    }

    fn forget(&mut self, key: &K) {
        if let Some(slot) = self.index.remove(key) {
            self.slots[slot as usize] = None;
        }
    }

    fn clear(&mut self) {
        self.slots.iter_mut().for_each(|s| *s = None);
        self.index.clear();
    }
}

const SOURCE: &str = r#"
#define RS "SRV(t0), SRV(t1), UAV(u0)"
// A page in video memory, its twin, and the page's mask: each bound at
// its own first byte.
ByteAddressBuffer page : register(t0);
ByteAddressBuffer twin : register(t1);
RWByteAddressBuffer mask : register(u0);
// A lane per word of the mask: 32 blocks of 256 bytes.
[RootSignature(RS)]
[numthreads(8, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint bits = 0;
    for (uint block = 0; block < 32; block++) {
        uint from = (lane * 32 + block) * 256;
        bool differs = false;
        for (uint k = 0; k < 256; k += 16) {
            differs = differs || any(page.Load4(from + k) != twin.Load4(from + k));
        }
        if (differs) {
            bits |= 1u << block;
        }
    }
    mask.Store(4 * lane, bits);
}
"#;

/// The pass's pipeline and buffers for `slots` twins.
struct Gpu {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// The twins, a page each, in video memory.
    twin: ID3D12Resource,
    /// The masks, [`MASK`] bytes per slot, in video memory.
    mask: ID3D12Resource,
    /// The masks read back.
    back: ID3D12Resource,
    back_data: *const u8,
    slots: usize,
}

// SAFETY: D3D12 objects are free-threaded; the mapping is read under the
// state's lock only.
unsafe impl Send for Gpu {}

/// A page to compare or to give a twin: its copy's buffer in video
/// memory, the page's offset there, and its slot.
struct Item<'a> {
    buffer: &'a ID3D12Resource,
    at: u64,
    slot: u32,
    /// Its twin is the page as it last went back: compared. Otherwise the
    /// slot is new and only filled.
    compare: bool,
}

impl Gpu {
    fn new(host: &Host, slots: usize) -> windows::core::Result<Self> {
        let none = windows::core::Error::empty;
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; out pointers ours.
        let compiled = unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("diff"),
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
                tracing::warn!("page compare: {}", String::from_utf8_lossy(text));
            }
            return Err(e);
        }
        let code = code.ok_or_else(none)?;
        // SAFETY: the blob's own bytes; the root signature is inside the shader.
        let (root, pso) = unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes)?;
            let pso: ID3D12PipelineState = host.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                ..Default::default()
            })?;
            (root, pso)
        };
        let twin = host.uav_buffer(slots as u64 * PAGE).ok_or_else(none)?;
        let mask = host.uav_buffer(slots as u64 * MASK).ok_or_else(none)?;
        let (back, back_data) = host.readback_buffer(slots as u64 * MASK).ok_or_else(none)?;
        crate::host::name(&twin, || "page twins".into());
        crate::host::name(&mask, || "page masks".into());
        Ok(Self { root, pso, twin, mask, back, back_data, slots })
    }

    /// Compares each item to compare with its twin, then makes every
    /// item's twin the page as it is now, in one list, waited for. For
    /// each item, in order: the blocks that differ (0 for one not
    /// compared), or `None` if the list did not run (the twins are then
    /// not to be trusted).
    fn run(&self, host: &Host, items: &[Item]) -> Option<Vec<u32>> {
        if items.is_empty() {
            return Some(Vec::new());
        }
        debug_assert!(items.iter().all(|i| (i.slot as usize) < self.slots && i.at.is_multiple_of(4)));
        let compared = items.iter().any(|i| i.compare);
        let (common, read, write, source, dest) = (
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        // SAFETY: (all) an open list of ours; our pipeline and buffers,
        // and the caller's live buffers, each page inside its buffer and
        // each slot inside ours. The copies' buffers are common and
        // promoted by their reads here, as everywhere.
        let ran = host.run_now(&mut |list: &ID3D12GraphicsCommandList| unsafe {
            if compared {
                let (twin_gpu, mask_gpu) = (self.twin.GetGPUVirtualAddress(), self.mask.GetGPUVirtualAddress());
                list.ResourceBarrier(&[transition(&self.twin, common, read), transition(&self.mask, common, write)]);
                list.SetComputeRootSignature(&self.root);
                list.SetPipelineState(&self.pso);
                for item in items.iter().filter(|i| i.compare) {
                    let slot = u64::from(item.slot);
                    list.SetComputeRootShaderResourceView(0, item.buffer.GetGPUVirtualAddress() + item.at);
                    list.SetComputeRootShaderResourceView(1, twin_gpu + slot * PAGE);
                    list.SetComputeRootUnorderedAccessView(2, mask_gpu + slot * MASK);
                    list.Dispatch(1, 1, 1);
                }
                list.ResourceBarrier(&[transition(&self.twin, read, dest), transition(&self.mask, write, source)]);
                list.CopyBufferRegion(&self.back, 0, &self.mask, 0, self.slots as u64 * MASK);
                list.ResourceBarrier(&[transition(&self.mask, source, common)]);
            } else {
                list.ResourceBarrier(&[transition(&self.twin, common, dest)]);
            }
            for item in items {
                list.CopyBufferRegion(&self.twin, u64::from(item.slot) * PAGE, item.buffer, item.at, PAGE);
            }
            list.ResourceBarrier(&[transition(&self.twin, dest, common)]);
        });
        if !ran {
            return None;
        }
        let blocks = |item: &Item| -> u32 {
            if !item.compare {
                return 0;
            }
            // SAFETY: the slot's mask in the mapped read-back buffer, which
            // the list just filled; byte reads, whatever its alignment.
            let bytes = unsafe {
                std::slice::from_raw_parts(self.back_data.add(item.slot as usize * MASK as usize), MASK as usize)
            };
            bytes.iter().map(|b| b.count_ones()).sum()
        };
        Some(items.iter().map(blocks).collect())
    }
}

/// The twins of copies' pages: the key is the copy's address and the
/// title page's.
struct State {
    gpu: Gpu,
    twins: Twins<(usize, usize)>,
    /// The copy each slot's twin belongs to. A copy that went away has no
    /// twins: while this names it, no other copy has its address.
    owners: Vec<Weak<Shadow>>,
}

/// None once making it failed: nothing is compared from then on.
fn state() -> Option<&'static Mutex<State>> {
    static STATE: OnceLock<Option<Mutex<State>>> = OnceLock::new();
    STATE
        .get_or_init(|| {
            let host = crate::host::host()?;
            let mb = std::env::var("VICTORY_MEMSYS_DIFF_MB").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(TWIN_MB);
            let slots = ((mb.clamp(1, 2048) << 20) / PAGE) as usize;
            match Gpu::new(host, slots) {
                Ok(gpu) => {
                    tracing::info!(slots, mb, "twins of marked pages in video memory");
                    Some(Mutex::new(State { gpu, twins: Twins::new(slots), owners: (0..slots).map(|_| Weak::new()).collect() }))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "no page compare on this host: every marked page comes back");
                    None
                }
            }
        })
        .as_ref()
}

/// For each of a submission's marked `pages` (a copy and a title page),
/// whether it holds what it held when it last went back to title memory:
/// nothing of it needs to come back. `goes_back`: whether the page goes
/// back now if changed; one that does not has no twin after. Empty when
/// the pass is off or cannot run: every page is changed then.
///
/// Call it after the submission has run and before the pages' pull, once
/// per marked page set: it makes each page's twin the page as it is now.
pub(crate) fn unchanged(pages: &[(Arc<Shadow>, usize)], goes_back: impl Fn(&Arc<Shadow>, usize) -> bool) -> Vec<bool> {
    if mode() == Mode::Off || pages.is_empty() || crate::host::riding_active() {
        return Vec::new();
    }
    let (Some(host), Some(state)) = (crate::host::host(), state()) else {
        return Vec::new();
    };
    let started = std::time::Instant::now();
    let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
    let State { gpu, twins, owners } = &mut *state;
    // The items, and which page each is of.
    let mut items = Vec::with_capacity(pages.len());
    let mut of = Vec::with_capacity(pages.len());
    for (i, (copy, page)) in pages.iter().enumerate() {
        let key = (Arc::as_ptr(copy) as usize, *page);
        let Some(at) = copy.video_page(*page).filter(|_| goes_back(copy, *page)) else {
            twins.forget(&key);
            continue;
        };
        // A twin is this copy's while the slot still names the copy.
        let known = twins.find(&key).filter(|&slot| std::ptr::eq(owners[slot as usize].as_ptr(), Arc::as_ptr(copy)));
        let slot = known.unwrap_or_else(|| {
            twins.forget(&key);
            twins.assign(key)
        });
        owners[slot as usize] = Arc::downgrade(copy);
        items.push(Item { buffer: &copy.buffer, at, slot, compare: known.is_some() });
        of.push(i);
    }
    // More pages than slots: a slot given twice holds the later page's
    // twin, so the earlier page's compare stands but it has no twin after
    // (`assign` forgot it).
    let Some(blocks) = gpu.run(host, &items) else {
        twins.clear();
        return Vec::new();
    };
    let mut same = vec![false; pages.len()];
    let (mut compared, mut equal, mut differing) = (0, 0, 0);
    for ((item, &i), &blocks) in items.iter().zip(&of).zip(&blocks) {
        if item.compare {
            compared += 1;
            differing += u64::from(blocks);
            if blocks == 0 {
                equal += 1;
                same[i] = true;
            }
        }
    }
    use super::{N, add};
    add(N::DiffPages, pages.len() as u64);
    add(N::DiffCompared, compared);
    add(N::DiffSame, equal);
    add(N::DiffBlocks, differing);
    add(N::DiffUs, started.elapsed().as_micros() as u64);
    same
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_reads_off_verify_and_on() {
        assert_eq!(mode_of(None), Mode::Off);
        assert_eq!(mode_of(Some("0")), Mode::Off);
        assert_eq!(mode_of(Some("verify")), Mode::Verify);
        assert_eq!(mode_of(Some("1")), Mode::On);
    }

    #[test]
    fn twins_keep_their_slots_and_the_oldest_makes_room() {
        let mut twins = Twins::new(3);
        assert_eq!(twins.find(&7), None);
        assert_eq!([twins.assign(7), twins.assign(8), twins.assign(9)], [0, 1, 2]);
        assert_eq!(twins.assign(8), 1, "its own slot again");
        // Full: the oldest slot is given to the next key.
        assert_eq!(twins.assign(10), 0);
        assert_eq!((twins.find(&7), twins.find(&10)), (None, Some(0)));
        twins.forget(&8);
        assert_eq!(twins.find(&8), None);
        // Slots go in turn, whether or not one was freed.
        assert_eq!(twins.assign(11), 1);
        assert_eq!(twins.assign(8), 2);
        assert_eq!(twins.find(&9), None, "made room");
        twins.clear();
        assert_eq!((twins.find(&8), twins.find(&10), twins.find(&11)), (None, None, None));
        assert_eq!(twins.assign(5), 0, "after a clear the turn goes on");
    }

    /// A page is compared with what it held when its twin was made, by
    /// blocks of 256 bytes, and the twin follows the page.
    #[test]
    fn a_page_differs_from_its_twin_where_it_was_written() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let Ok(gpu) = Gpu::new(host, 4) else {
            return; // no compiler or no such buffers here
        };
        let len = 2 * PAGE as usize;
        let (buffer, data) = host.shared_buffer(len as u64).unwrap();
        // SAFETY: (all below) inside the buffer's `len` mapped bytes.
        unsafe {
            for i in 0..len {
                data.add(i).write((i * 7 + i / 300) as u8);
            }
        }
        let item = |page: u64, slot: u32, compare: bool| Item { buffer: &buffer, at: page * PAGE, slot, compare };
        // New twins: nothing compared.
        assert_eq!(gpu.run(host, &[item(0, 2, false), item(1, 0, false)]), Some(vec![0, 0]));
        assert_eq!(gpu.run(host, &[item(0, 2, true), item(1, 0, true)]), Some(vec![0, 0]), "as the twins");
        // A byte in the second page's block 0x12, and two blocks apart in
        // the first page, one of them its last.
        // SAFETY: inside the buffer, as above.
        unsafe {
            *data.add(PAGE as usize + 0x1234) ^= 1;
            *data.add(0x00ff) ^= 0x80;
            *data.add(PAGE as usize - 1) ^= 0x80;
        }
        assert_eq!(gpu.run(host, &[item(0, 2, true), item(1, 0, true)]), Some(vec![2, 1]));
        // SAFETY: the read-back masks of slots 0 and 2, just filled.
        let mask = |slot: usize| unsafe { std::slice::from_raw_parts(gpu.back_data.add(slot * MASK as usize), MASK as usize) };
        let bit = |mask: &[u8], block: usize| (mask[block / 8] >> (block % 8)) & 1 == 1;
        assert!(bit(mask(0), 0x12) && bit(mask(2), 0) && bit(mask(2), 255));
        // The twins followed: equal again, and one page's change is not
        // the other's.
        assert_eq!(gpu.run(host, &[item(0, 2, true), item(1, 0, true)]), Some(vec![0, 0]));
        // SAFETY: inside the buffer, as above.
        unsafe { *data.add(5) ^= 1 };
        assert_eq!(gpu.run(host, &[item(1, 0, true), item(0, 2, true)]), Some(vec![0, 1]));
        // A page only given a twin is not compared, whatever the slot held.
        assert_eq!(gpu.run(host, &[item(0, 0, false)]), Some(vec![0]));
        assert_eq!(gpu.run(host, &[item(0, 0, true), item(1, 2, true)]).map(|b| (b[0], b[1] > 0)), Some((0, true)));
    }
}
