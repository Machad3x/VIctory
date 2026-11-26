//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory changed since the last detiling
//! ([`crate::watch`], which follows writes through every mapping of the
//! same physical pages) in the runtime's memory; elsewhere, a content hash
//! at every submission that samples it (of the subresources on untracked
//! pages, or of the whole memory if none is tracked).
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Textures read through descriptor tables are brought up to date, and for
//! shaders that index the heap themselves every texture with a resource
//! object the heap views (`crate::bindless`); textures the title placed in
//! memory itself are not made for the heap (see `descriptor::host`). Plane
//! 0 is uploaded (every mip and array slice), depth and stencil planes are
//! not.

use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;
use windows::core::Interface;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// shortcut: GPU copies between different packed layouts need a conversion pass.
pub(crate) fn incompatible_copy(src: Option<u32>, dst: Option<u32>) -> bool {
    if src == dst {
        return false;
    }
    once!(?src, ?dst, "GPU texture copy or resolve skipped: packed stand-in layouts differ");
    true
}

/// R11G11B10 float words to R9G9B9E5, in place in a buffer of `pitch`-byte
/// rows (`width` texels each): [`small_float`] and [`rgb9e5`] on the GPU.
const STAND_IN_TO_67: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), UAV(u0)"
cbuffer C : register(b0) { uint width; uint pitch; };
RWByteAddressBuffer buf : register(u0);
float small(uint bits, uint man) {
    uint e = (bits >> man) & 0x1f;
    uint m = bits & ((1u << man) - 1);
    if (e == 31) return m != 0 ? 0.0 : 65408.0;
    if (e == 0) return (float)m * asfloat((127 - 14 - man) << 23);
    return asfloat(((e + 112) << 23) | (m << (23 - man)));
}
uint pack(float3 c) {
    c = min(c, 65408.0);
    float mx = max(c.r, max(c.g, c.b));
    int e = max(-16, (int)(asuint(mx) >> 23) - 127) + 16;
    float denom = asfloat((uint)(e - 24 + 127) << 23);
    if (floor(mx / denom + 0.5) >= 512.0) { denom *= 2.0; e += 1; }
    uint3 q = (uint3)floor(c / denom + 0.5);
    return q.r | q.g << 9 | q.b << 18 | (uint)e << 27;
}
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= width) return;
    uint a = id.y * pitch + id.x * 4;
    uint v = buf.Load(a);
    buf.Store(a, pack(float3(small(v & 0x7ff, 6), small((v >> 11) & 0x7ff, 6), small(v >> 22, 5))));
}
"#;

/// Copies subresource `from_sub` (the region `b`, or all of it) of an
/// R9G9B9E5 stand-in (R11G11B10, [`crate::resource::target_format`]) into
/// subresource `to_sub` at `at` of a native R9G9B9E5 texture: through the
/// shared scratch buffer, converted there by a compute pass (the list's
/// compute state is set again after, `need_rebind`). Both textures are in
/// the copy states already. False if the pass is not available.
/// shortcut: 2D regions of one slice; the other direction (native 67 into
/// a stand-in) is still skipped by [`incompatible_copy`].
pub(crate) fn copy_stand_in_to_67(
    list: &HostList,
    to: &ID3D12Resource,
    to_sub: u32,
    at: [u32; 3],
    from: &ID3D12Resource,
    from_sub: u32,
    b: Option<&D3D12_BOX>,
) -> bool {
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
    static PASS: std::sync::OnceLock<Option<crate::twin::Converter>> = std::sync::OnceLock::new();
    let Some(c) = PASS
        .get_or_init(|| crate::twin::compile(STAND_IN_TO_67).inspect_err(|e| tracing::warn!(error = %e, "no R9G9B9E5 conversion")).ok())
        .as_ref()
    else {
        return false;
    };
    // SAFETY: live textures.
    let d = unsafe { from.GetDesc() };
    if !stand_in_copy_fits(&d, &unsafe { to.GetDesc() }, b) {
        return false;
    }
    let mip = from_sub % u32::from(d.MipLevels.max(1));
    let (width, height) = match b {
        Some(b) => (b.right.saturating_sub(b.left), b.bottom.saturating_sub(b.top)),
        None => (((d.Width >> mip).max(1)) as u32, (d.Height >> mip).max(1)),
    };
    if width == 0 || height == 0 {
        return true;
    }
    let pitch = (width * 4).next_multiple_of(256);
    let Some(scratch) = crate::twin::shared_scratch(u64::from(pitch) * u64::from(height)) else {
        return false;
    };
    let placed = |format: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the copy does not release.
        pResource: unsafe { std::mem::transmute_copy(&scratch) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: DXGI_FORMAT(format as i32),
                    Width: width,
                    Height: height,
                    Depth: 1,
                    RowPitch: pitch,
                },
            },
        },
    };
    let sub = |r: &ID3D12Resource, i: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i },
    };
    let state = |a, z| crate::host::transition(&scratch, a, z);
    let constants = [width, pitch];
    // SAFETY: an open list; live resources in the states named (the scratch
    // buffer starts common: its first copy promotes it).
    unsafe {
        let l = &list.list;
        l.CopyTextureRegion(&placed(26), 0, 0, 0, &sub(from, from_sub), b.map(std::ptr::from_ref));
        l.ResourceBarrier(&[state(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, scratch.GetGPUVirtualAddress());
        l.Dispatch(width.div_ceil(64), height, 1);
        l.ResourceBarrier(&[
            crate::twin::uav_barrier(),
            state(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        ]);
        l.CopyTextureRegion(&sub(to, to_sub), at[0], at[1], at[2], &placed(67), None);
        l.ResourceBarrier(&[state(D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
    }
    list.need_rebind();
    true
}

/// Whether [`copy_stand_in_to_67`] handles a copy between textures of
/// these descriptions: both 2D, the box (if any) one slice deep.
fn stand_in_copy_fits(from: &D3D12_RESOURCE_DESC, to: &D3D12_RESOURCE_DESC, b: Option<&D3D12_BOX>) -> bool {
    let two_d = |d: &D3D12_RESOURCE_DESC| d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    two_d(from) && two_d(to) && b.is_none_or(|b| b.back.saturating_sub(b.front) <= 1)
}

/// A float packed in `bits` with a 5-bit exponent and `man` mantissa bits
/// (the 11- and 10-bit floats of R11G11B10).
pub(crate) fn small_float(bits: u32, man: u32) -> f32 {
    let (exp, m) = ((bits >> man) & 0x1f, (bits & ((1 << man) - 1)) as f32);
    match exp {
        0 => m / (1 << man) as f32 * 2f32.powi(-14),
        31 if m != 0.0 => f32::NAN,
        31 => f32::INFINITY,
        e => (1.0 + m / (1 << man) as f32) * 2f32.powi(e as i32 - 15),
    }
}

/// `v` as a float of a 5-bit exponent and `man` mantissa bits, rounded to
/// nearest, denormals kept; NaN and negatives are 0, larger than the
/// largest finite (65024 for 11 bits, 64512 for 10) clamps to it.
fn small_bits(v: f32, man: u32) -> u32 {
    let max = 30 << man | ((1 << man) - 1);
    if v.is_nan() || v <= 0.0 {
        return 0;
    }
    let b = v.to_bits();
    let exp = (b >> 23) as i32 - 127;
    let bits = if exp < -14 {
        (v * 2f32.powi(14 + man as i32)).round() as u32
    } else {
        let shift = 23 - man;
        ((((exp + 15) as u32) << 23 | (b & 0x7f_ffff)) + (1 << (shift - 1))) >> shift
    };
    bits.min(max)
}

/// Channels (clamped to 0..=65408, NaN to 0) as R9G9B9E5 bits: the
/// standard shared exponent, bumped when the largest mantissa rounds over.
fn rgb9e5(c: [f32; 3]) -> u32 {
    let c = c.map(|v| if v > 0.0 { v.min(65408.0) } else { 0.0 });
    let max = c[0].max(c[1]).max(c[2]);
    let mut exp = ((max.to_bits() >> 23) as i32 - 127).max(-16) + 16;
    let mut denom = 2f32.powi(exp - 24);
    if (max / denom + 0.5).floor() >= 512.0 {
        denom *= 2.0;
        exp += 1;
    }
    let m = c.map(|v| (v / denom + 0.5).floor() as u32);
    m[0] | m[1] << 9 | m[2] << 18 | (exp as u32) << 27
}

/// R9G9B9E5 bits (`mantissa * 2^(e-24)`) as the R11G11B10 float stand-in.
fn rgb9e5_to_r11g11b10(bits: u32) -> u32 {
    let scale = 2f32.powi((bits >> 27) as i32 - 24);
    let c = |shift: u32| (bits >> shift & 0x1ff) as f32 * scale;
    small_bits(c(0), 6) | small_bits(c(9), 6) << 11 | small_bits(c(18), 5) << 22
}

fn r11g11b10_to_rgb9e5(bits: u32) -> u32 {
    rgb9e5([small_float(bits & 0x7ff, 6), small_float(bits >> 11 & 0x7ff, 6), small_float(bits >> 22, 5)])
}

/// Converts one row from the title's little-endian BGRA bits to host RGBA8
/// (R9G9B9E5: to R11G11B10 float, 4 bytes each).
pub(crate) fn expand_row(format: u32, packed: &[u8], rgba: &mut [u8]) {
    if format == 67 {
        for (p, c) in packed.as_chunks::<4>().0.iter().zip(rgba.as_chunks_mut::<4>().0.iter_mut()) {
            *c = rgb9e5_to_r11g11b10(u32::from_le_bytes(*p)).to_le_bytes();
        }
        return;
    }
    for (p, c) in packed.as_chunks::<2>().0.iter().zip(rgba.as_chunks_mut::<4>().0.iter_mut()) {
        let bits = u16::from_le_bytes(*p);
        let channel = |shift: u32, max: u16| (((bits >> shift & max) * 255 + max / 2) / max) as u8;
        let color = match format {
            86 => [channel(10, 31), channel(5, 31), channel(0, 31), channel(15, 1)],
            115 => [channel(8, 15), channel(4, 15), channel(0, 15), channel(12, 15)],
            _ => unreachable!(),
        };
        *c = color;
    }
}

pub(crate) fn pack_row(format: u32, rgba: &[u8], packed: &mut [u8]) {
    if format == 67 {
        for (c, p) in rgba.as_chunks::<4>().0.iter().zip(packed.as_chunks_mut::<4>().0.iter_mut()) {
            *p = r11g11b10_to_rgb9e5(u32::from_le_bytes(*c)).to_le_bytes();
        }
        return;
    }
    for (c, p) in rgba.as_chunks::<4>().0.iter().zip(packed.as_chunks_mut::<2>().0.iter_mut()) {
        let channel = |i: usize, max: u16| (u16::from(c[i]) * max + 127) / 255;
        let bits = match format {
            86 => channel(3, 1) << 15 | channel(0, 31) << 10 | channel(1, 31) << 5 | channel(2, 31),
            115 => channel(3, 15) << 12 | channel(0, 15) << 8 | channel(1, 15) << 4 | channel(2, 15),
            _ => unreachable!(),
        };
        *p = bits.to_le_bytes();
    }
}

/// A texture's upload state: how a staging buffer lays out the host
/// texture's subresources, and what was last detiled. The buffer itself is
/// taken for each upload from a small pool shared by all textures
/// ([`staging_buffer`]; kept for each texture's life, the buffers took its
/// size again in system memory: 2.6-2.8 GB with placed textures made on
/// read, run integ2).
pub(crate) struct Staging {
    /// A full upload's staging bytes.
    size: u64,
    packed: Option<u32>,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    /// The memory's pages changed since detiled (none: not the runtime's
    /// memory, hashed instead).
    tracked: Option<crate::watch::Tracked>,
    /// Per subresource, the bytes of the texture's memory it lies in (from
    /// its start, whole layout blocks); empty where the layout gives none.
    spans: Vec<(usize, usize)>,
    last: Mutex<Last>,
    /// The last detile failed after its changed pages were taken: the
    /// tracker no longer says so, so the next refresh reads everything.
    retry: std::sync::atomic::AtomicBool,
    /// Uploads [`record_upload`] recorded, and the subresources they
    /// copied.
    uploads: std::sync::atomic::AtomicU64,
    copied: std::sync::atomic::AtomicU64,
}

/// What was last detiled: the hash of the memory (not tracked), or of each
/// subresource's bytes (tracked).
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    hashes: Vec<Option<blake3::Hash>>,
}

/// Per subresource of `layout` (`mips` per array slice; a 3D texture's
/// are its mips), the bytes of the texture's memory it lies in, rounded out
/// to the layout's blocks; empty if the layout cannot say.
fn spans(layout: &victory_formats::texture::Layout, count: u32, mips: u32) -> Vec<(usize, usize)> {
    use victory_formats::texture::Dimension;
    let d = &layout.desc;
    let block = layout.alignment.max(256);
    (0..count)
        .map(|i| {
            let (mip, slice) = (i % mips, i / mips);
            let m = layout.mips.get(mip as usize)?;
            let (z0, z1) = if d.dimension == Dimension::Tex3d { (0, m.depth.max(1) - 1) } else { (slice, slice) };
            let first = layout.offset(mip, 0, 0, z0).ok()?;
            let last = layout.offset(mip, m.width.max(1) - 1, m.height.max(1) - 1, z1).ok()? + u64::from(d.element_bytes);
            let (from, to) = (first / block * block, last.div_ceil(block) * block);
            Some((from as usize, (to.min(layout.size)) as usize))
        })
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default()
}

/// Staging buffers for detiles to reuse, by size: the menu movie rewrites
/// the same textures every frame, and a fresh 12.7 MB of upload buffers a
/// frame cost ~5 ms (run textures8). Uploads run only inside
/// `Host::execute`, which waits for each submission, so the buffers of one
/// submission are free from the next on.
struct Buffers {
    /// (bytes, buffer, mapping), oldest first.
    free: std::collections::VecDeque<(u64, ID3D12Resource, usize)>,
    free_bytes: u64,
    /// Taken by the submission being prepared.
    used: Vec<(u64, ID3D12Resource, usize)>,
}

// SAFETY: the mappings are persistent mappings of the upload buffers.
unsafe impl Send for Buffers {}

static BUFFERS: Mutex<Buffers> = Mutex::new(Buffers {
    free: std::collections::VecDeque::new(),
    free_bytes: 0,
    used: Vec::new(),
});

/// Free staging buffers kept at most.
const KEPT_BYTES: u64 = 64 << 20;

/// Called before a submission's uploads: the buffers of the last one are
/// free again (the oldest dropped past `KEPT_BYTES`).
pub(crate) fn recycle() {
    {
        let mut c = CHUNKS.lock().unwrap_or_else(|e| e.into_inner());
        (c.current, c.used) = (0, 0);
        c.all.truncate(CHUNKS_KEPT);
    }
    let mut b = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
    let b = &mut *b;
    for e in b.used.drain(..) {
        b.free_bytes += e.0;
        b.free.push_back(e);
    }
    while b.free_bytes > KEPT_BYTES
        && let Some((n, ..)) = b.free.pop_front()
    {
        b.free_bytes -= n;
    }
}

/// A free staging buffer of `size` bytes, or a new one.
/// Sizes are taken in powers of two (at least 64 KiB), so a buffer serves
/// textures of other sizes too: a new shot's textures came in new sizes and
/// made buffers 10 % of the submitting thread (run evict5).
fn staging_buffer(size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    let class = size.max(1 << 16).next_power_of_two();
    let taken = {
        let mut b = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
        let i = b.free.iter().position(|e| e.0 == class);
        i.and_then(|i| b.free.remove(i)).inspect(|_| b.free_bytes -= class)
    };
    // Made without the pool's lock (refreshes run on several threads).
    let e = match taken {
        Some(e) => e,
        None => {
            let (buffer, data) = host()?.upload_buffer(class)?;
            crate::host::name(&buffer, || "texture staging".into());
            (class, buffer, data as usize)
        }
    };
    let made = (e.1.clone(), e.2 as *mut u8);
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner()).used.push(e);
    Some(made)
}

/// Staging space carved from large upload buffers, front to back: most
/// textures take a part of one instead of a buffer of their own (making
/// upload buffers was 10 % of the submitting thread in a new shot, run
/// evict5). Rewound at each submission ([`recycle`]): the last one has run.
struct Chunks {
    /// (buffer, mapping), [`CHUNK`] bytes each.
    all: Vec<(ID3D12Resource, usize)>,
    /// The chunk being carved and the bytes taken from it.
    current: usize,
    used: u64,
}

// SAFETY: the mappings are persistent mappings of the upload buffers.
unsafe impl Send for Chunks {}

static CHUNKS: Mutex<Chunks> = Mutex::new(Chunks { all: Vec::new(), current: 0, used: 0 });

const CHUNK: u64 = 32 << 20;
/// Chunks kept between submissions (more are made while one needs them).
const CHUNKS_KEPT: usize = 16;

/// `size` bytes of staging space: a buffer, the offset of the space in it
/// (a multiple of 512, as texture copies need) and its mapping there.
/// Space larger than a chunk is a buffer of its own ([`staging_buffer`]).
fn staging_space(size: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
    if size > CHUNK {
        return staging_buffer(size).map(|(b, data)| (b, 0, data));
    }
    let mut c = CHUNKS.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some((buffer, data)) = c.all.get(c.current) {
            let at = c.used.next_multiple_of(512);
            if at + size <= CHUNK {
                let made = (buffer.clone(), at, (*data + at as usize) as *mut u8);
                c.used = at + size;
                return Some(made);
            }
            (c.current, c.used) = (c.current + 1, 0);
            continue;
        }
        let (buffer, data) = host()?.upload_buffer(CHUNK)?;
        crate::host::name(&buffer, || "texture staging chunk".into());
        c.all.push((buffer, data as usize));
    }
}

// SAFETY: the title memory is only read, under `last`'s lock.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

/// RAII pin keeping a destination `HostTexture` alive against eviction while
/// a recorded list holds a pending `host_writes` mark for it. Dropping without
/// [`PendingHostWrite::record`] releases the pin without marking `host_written`.
pub(crate) struct PendingHostWrite(Arc<crate::resource::HostTexture>);

impl PendingHostWrite {
    /// The pinned host `ID3D12Resource`, guaranteed to match the pinned copy.
    pub(crate) fn resource(&self) -> &ID3D12Resource {
        &self.0.texture
    }

    /// Transfers this pin to `list`'s deferred step so `host_written` is set on
    /// submission and the pin is released when `list` resets or drops.
    pub(crate) fn record(self, list: &HostList) {
        use std::sync::atomic::Ordering::Release;
        let guard = self;
        list.defer_host(move || {
            // Held as in `still_host`, so it cannot clear a mark being made.
            let staging = guard.0.staging.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(s) = staging.as_ref() {
                s.drain();
            }
            guard
                .0
                .pending_host_writes
                .fetch_or(crate::resource::HOST_WRITTEN, Release);
            guard.0.host_written.store(true, Release);
        });
    }
}

impl Drop for PendingHostWrite {
    fn drop(&mut self) {
        self.0
            .pending_host_writes
            .fetch_sub(1, std::sync::atomic::Ordering::Release);
    }
}

/// Atomically acquires `resource`'s current host texture and pins it against
/// concurrent eviction before any GPU write is recorded into it. If eviction
/// already claimed the copy with `EVICTED`, detaches it and acquires a fresh
/// replacement.
pub(crate) fn prepare_host_write(resource: This) -> Option<PendingHostWrite> {
    use std::sync::atomic::Ordering::{AcqRel, Acquire};
    for _ in 0..16 {
        let copy = crate::resource::host_copy(resource)?;
        if copy
            .pending_host_writes
            .fetch_update(AcqRel, Acquire, |cur| {
                (cur & crate::resource::EVICTED == 0).then_some(cur + 1)
            })
            .is_ok()
        {
            // Made before the mark: one made after it would see every page changed.
            if title_written(&copy.texture) {
                staging(resource, &copy.texture);
            }
            return Some(PendingHostWrite(copy));
        }
        crate::resource::detach_if_same(resource, &copy);
    }
    None
}

/// A host copy or resolve that `list` records writes `resource`: from the
/// submission that runs `list` on, its host texture's contents are the
/// host's, not title memory's, which stays as it was. Marked when that
/// submission is made, not when `list` is recorded: a list recorded before
/// it but run after it must not upload title memory over the host's texels
/// (submissions run in the order they are made, which need not be the
/// order their lists were recorded in), and one run before it must still
/// get its upload.
#[cfg(test)]
pub(crate) fn host_writes(list: &HostList, resource: This) {
    if let Some(write) = prepare_host_write(resource) {
        write.record(list);
    }
}

fn is_host_written(resource: This) -> bool {
    crate::resource::host_copy(resource).is_some_and(|c| still_host(&c))
}

/// Whether `copy`'s contents are still the host's: a host write marked it and
/// the title has not overwritten all of its memory since (the mark then goes).
pub(crate) fn still_host(copy: &crate::resource::HostTexture) -> bool {
    use std::sync::atomic::Ordering::{Acquire, Release};
    if !copy.host_written.load(Acquire) {
        return false;
    }
    let staging = copy.staging.lock().unwrap_or_else(|e| e.into_inner());
    // Another thread may have cleared it before the lock was ours.
    if !copy.host_written.load(Acquire) {
        return false;
    }
    let Some((_memory, t)) = staging
        .as_ref()
        .and_then(|s| Some((s.memory, s.tracked.as_ref()?)))
    else {
        return true;
    };
    // Lock free: nothing changed at all.
    if t.clean() {
        return true;
    }
    // A render target is never uploaded from title memory.
    if !t.overwritten() || !title_written(&copy.texture) {
        #[cfg(feature = "diagnostics")]
        HOST_KEPT_STALE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return true;
    }
    copy.host_written.store(false, Release);
    copy.pending_host_writes.fetch_and(!crate::resource::HOST_WRITTEN, Release);
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
        static LOGGED: AtomicU32 = AtomicU32::new(0);
        HOST_MARKS_CLEARED.fetch_add(1, Relaxed);
        if LOGGED.fetch_update(Relaxed, Relaxed, |n| (n < 20).then_some(n + 1)).is_ok() {
            // SAFETY: a live resource.
            let desc = unsafe { copy.texture.GetDesc() };
            tracing::info!("host-written mark cleared: memory {_memory:p}, {desc:?}");
        }
    }
    false
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Copies [`alias_copy`] recorded since the last frame-rate line.
pub(crate) static ALIAS_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Reads since the last frame-rate line that kept a host copy although pages
/// of its memory had changed.
#[cfg(feature = "diagnostics")]
pub(crate) static HOST_KEPT_STALE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Host-written marks cleared since the last frame-rate line.
#[cfg(feature = "diagnostics")]
pub(crate) static HOST_MARKS_CLEARED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Records, once per list, the copy of `from`'s host texture into `to`'s:
/// two title textures at one memory that cannot share a host texture
/// (`resource::alias_source`: block-compressed against plain formats of
/// the same bytes, or a cast the host refused), `to` (a placed view's)
/// reading what the GPU wrote through `from`. Through a buffer, each
/// subresource with its own format, as many as have the same bytes per
/// row and rows in both; `to` no longer takes its contents from title
/// memory.
pub(crate) fn alias_copy(list: &HostList, from: This, to: This) {
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let (Some(a), Some(to_write), Some(host)) = (
        crate::resource::host_resource(from),
        prepare_host_write(to),
        host(),
    ) else {
        return;
    };
    let b = to_write.resource().clone();
    if incompatible_copy(crate::resource::packed_format(from), crate::resource::packed_format(to)) {
        return;
    }
    // An odd key: textures' own first uses are keyed by the host texture.
    if !list.first_use(to as usize | 1) {
        return;
    }
    // SAFETY: live resources.
    let (da, db) = unsafe { (a.GetDesc(), b.GetDesc()) };
    let count = |d: &D3D12_RESOURCE_DESC| {
        if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            u32::from(d.MipLevels)
        } else {
            u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize).max(1)
        }
    };
    let footprints = |d: &D3D12_RESOURCE_DESC, n: u32| {
        let mut f = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
        let mut rows = vec![0u32; n as usize];
        let mut total = 0;
        // SAFETY: outputs sized for `n` subresources.
        unsafe {
            host.device
                .GetCopyableFootprints(d, 0, n, 0, Some(f.as_mut_ptr()), Some(rows.as_mut_ptr()), None, Some(&mut total));
        }
        (f, rows, total)
    };
    let (fa, ra, total) = footprints(&da, count(&da));
    let (fb, rb, _) = footprints(&db, count(&db));
    // Subresources in order while both have the same bytes there (mips of
    // a block-compressed texture past its last whole block do not).
    let n = (0..fa.len().min(fb.len()))
        .take_while(|&i| {
            let (x, y) = (fa[i], fb[i]);
            x.Offset == y.Offset && x.Footprint.RowPitch == y.Footprint.RowPitch && ra[i] == rb[i] && x.Footprint.Depth == y.Footprint.Depth
        })
        .count() as u32;
    if n == 0 {
        once!(?da, ?db, "an aliased texture not copied: its layouts differ on the host");
        return;
    }
    let Some(buffer) = host.default_buffer(total) else { return };
    list.keep(&buffer);
    to_write.record(list);
    let bound = crate::list::ensure_state(list, from, COPY_SOURCE);
    crate::list::ensure_state(list, to, COPY_DEST);
    let place = |r: &ID3D12Resource, f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the copy does not release.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
    };
    let sub = |r: &ID3D12Resource, i: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i },
    };
    let l = &list.list;
    // SAFETY: an open list; live resources in the states named.
    unsafe {
        for i in 0..n {
            l.CopyTextureRegion(&place(&buffer, fa[i as usize]), 0, 0, 0, &sub(&a, i), None);
        }
        // The draw or dispatch reading `to` may have bound `from` as an
        // unordered or shader view already: back to that state. Targets are
        // set after the tables are bound, and restoring a depth write would
        // tell a twin of it of a write that never happened.
        // D3D12_RESOURCE_STATE_ UNORDERED_ACCESS and both shader resources.
        if let Some(state) = bound.filter(|&s| s != 0 && s & !(0x8 | 0x40 | 0x80) == 0) {
            crate::list::ensure_state(list, from, state);
        }
        l.ResourceBarrier(&[crate::host::transition(
            &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        for i in 0..n {
            l.CopyTextureRegion(&sub(&b, i), 0, 0, 0, &place(&buffer, fb[i as usize]), None);
        }
    }
    ALIAS_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static LOGGED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
    let mut logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
    if logged.len() < 64 && !logged.contains(&(from as usize, to as usize)) {
        logged.push((from as usize, to as usize));
        tracing::info!(from = ?from, to = ?to, a = da.Format.0, b = db.Format.0, w = da.Width, h = da.Height, subresources = n, "aliased texture copied");
    }
}

/// Has `resource`'s host copy brought up to date at the submission that
/// runs `list`: its staging buffer detiled again if the memory changed, and
/// then copied into the host texture by the submission's own upload list,
/// once for all its lists (`crate::bindless::uploads`).
///
/// shortcut: a title copy into texture memory recorded in the same
/// submission shows from the next one on.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    // shortcut: a texture the host copied into keeps the host's contents
    // until the title writes every page of its memory.
    // Once per list for each host texture, however many title textures of
    // its memory the list samples.
    if is_host_written(resource) {
        #[cfg(feature = "diagnostics")]
        {
            static SEEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);
            if SEEN.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default().insert(resource as usize) {
                let memory = crate::resource::texture_memory(resource).map(|(m, _)| m);
                // SAFETY: a live resource.
                let desc = unsafe { texture.GetDesc() };
                tracing::debug!(?resource, ?memory, ?desc, "a sampled texture keeps what the host copied into it: not uploaded from title memory");
            }
        }
        return;
    }
    if !list.first_use(texture.as_raw() as usize) {
        return;
    }
    let (Some(staging), Some(copy)) = (staging(resource, texture), crate::resource::host_copy(resource)) else {
        return;
    };
    // Each list recorded the copy before (run perf7: 149 textures uploaded
    // 3,195 times in a frame, 17 ms of GPU time), changed or not.
    list.defer_texture(staging, copy);
}

/// Records into `list` the copy of `buffer` (laid out as `staging` says)
/// into host texture `copy`. Returns the barrier that leaves it readable by
/// shaders, for the caller to record with the others after every copy: one
/// barrier per texture between the copies kept the GPU from overlapping
/// them (new shots copy hundreds of textures, run evict6).
#[must_use]
pub(crate) fn record_upload(
    list: &HostList,
    copy: &Arc<crate::resource::HostTexture>,
    staging: &Staging,
    (buffer, at, subresources): &Upload,
) -> Option<D3D12_RESOURCE_BARRIER> {
    const COPY_DEST: u32 = 0x400;
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    #[cfg(feature = "diagnostics")]
    list.note_host("host-upload");
    if let Some(b) = list.move_texture(copy, COPY_DEST) {
        // SAFETY: an open list; the barrier names a live texture.
        unsafe { list.list.ResourceBarrier(&[b]) };
    }
    copies(list, &copy.texture, staging, (buffer, *at), subresources, String::new);
    list.keep(buffer);
    staging.uploads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    staging.copied.fetch_add(subresources.len() as u64, std::sync::atomic::Ordering::Relaxed);
    list.move_texture(copy, SHADER_RESOURCE)
}

/// The uploads [`record_upload`] recorded into `resource`'s host texture
/// (tests: counts of their own texture, whatever other tests upload
/// meanwhile).
#[cfg(test)]
pub(crate) fn uploads(resource: This) -> u64 {
    let staging = crate::resource::host_copy(resource).and_then(|c| c.staging.lock().unwrap_or_else(|e| e.into_inner()).clone());
    staging.map_or(0, |s| s.uploads.load(std::sync::atomic::Ordering::Relaxed))
}

/// The subresources those uploads copied (tests, as [`uploads`]).
#[cfg(test)]
pub(crate) fn copied(resource: This) -> u64 {
    let staging = crate::resource::host_copy(resource).and_then(|c| c.staging.lock().unwrap_or_else(|e| e.into_inner()).clone());
    staging.map_or(0, |s| s.copied.load(std::sync::atomic::Ordering::Relaxed))
}


/// Records the copy of `subresources` of `buffer` (laid out as `staging`
/// says) into `texture` (in the copy destination state); `name` names it
/// in diagnostics notes.
#[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
fn copies(
    list: &HostList,
    texture: &ID3D12Resource,
    staging: &Staging,
    (buffer, at): (&ID3D12Resource, u64),
    subresources: &[u32],
    name: impl Fn() -> String,
) {
    let mut next = at;
    for &i in subresources {
        let span = staging.span(i);
        let i = i as usize;
        let mut footprint = staging.footprints[i];
        footprint.Offset = next;
        next += span;
        let footprint = &footprint;
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = span.min(u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth));
            list.note_read(
                || {
                    format!(
                        "texture upload {} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}: offset {:#x}+{len:#x} {}x{}x{} pitch {}",
                        name(),
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
}

/// What a refresh gives: a staging buffer, where its space starts in it,
/// and the subresources detiled into it.
pub(crate) type Upload = (ID3D12Resource, u64, Vec<u32>);

struct Job {
    call: fn(*const (), usize),
    data: *const (),
    n: usize,
    next: std::sync::atomic::AtomicUsize,
    panicked: Mutex<Option<Box<dyn std::any::Any + Send>>>,
}

// SAFETY: `Job` is shared across worker threads only while the caller's
// `WaitGuard` keeps the caller's stack frame alive until `remaining == 0`,
// and `data` points to an `F: Fn(usize) + Sync`.
unsafe impl Sync for Job {}

struct PoolState {
    job: *const Job,
    generation: u64,
    wanted: usize,
    claimed: usize,
    remaining: usize,
    shutdown: bool,
}

// SAFETY: `job` points to a `Sync` `Job` pinned on the caller's stack for the
// duration of the batch (`remaining > 0`).
unsafe impl Send for PoolState {}

/// Persistent worker pool for [`in_parallel`]: worker threads stay alive
/// across submissions so each thread keeps its `thread_local!` [`Scratch`]
/// buffer warm instead of re-faulting and zero-filling it on every call.
pub(crate) struct ParallelPool {
    workers: usize,
    run_lock: Mutex<()>,
    state: Arc<(Mutex<PoolState>, std::sync::Condvar, std::sync::Condvar)>,
    #[cfg(test)]
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl ParallelPool {
    pub(crate) fn new(workers: usize) -> Self {
        let state = Arc::new((
            Mutex::new(PoolState {
                job: std::ptr::null(),
                generation: 0,
                wanted: 0,
                claimed: 0,
                remaining: 0,
                shutdown: false,
            }),
            std::sync::Condvar::new(),
            std::sync::Condvar::new(),
        ));
        let mut live_workers = 0usize;
        #[cfg(test)]
        let mut handles = Vec::with_capacity(workers);
        for idx in 0..workers {
            let shared = Arc::clone(&state);
            let builder = std::thread::Builder::new().name(format!("victory-tex-{idx}"));
            if let Ok(h) = builder.spawn(move || {
                let (lock, work_cv, done_cv) = &*shared;
                let mut my_gen = 0u64;
                loop {
                    let job_ptr = {
                        let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
                        while !st.shutdown && st.generation == my_gen {
                            st = work_cv.wait(st).unwrap_or_else(|e| e.into_inner());
                        }
                        if st.shutdown {
                            break;
                        }
                        my_gen = st.generation;
                        if st.claimed >= st.wanted {
                            continue;
                        }
                        st.claimed += 1;
                        st.job
                    };
                    // SAFETY: `job_ptr` is non-null and valid until this worker
                    // decrements `st.remaining` below.
                    run_job(unsafe { &*job_ptr });
                    let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
                    st.remaining -= 1;
                    if st.remaining == 0 {
                        st.job = std::ptr::null();
                        done_cv.notify_one();
                    }
                }
            }) {
                live_workers += 1;
                #[cfg(test)]
                handles.push(h);
                #[cfg(not(test))]
                drop(h);
            }
        }
        Self {
            workers: live_workers,
            run_lock: Mutex::new(()),
            state,
            #[cfg(test)]
            handles,
        }
    }

    pub(crate) fn run<F: Fn(usize) + Sync>(&self, threads: usize, n: usize, f: F) {
        use std::sync::atomic::AtomicUsize;

        let workers = threads.saturating_sub(1).min(self.workers);
        if workers == 0 || n < 2 {
            (0..n).for_each(f);
            return;
        }
        // Contended or re-entrant call: run inline instead of blocking or
        // deadlocking against an outer `in_parallel` holding `run_lock`.
        let Ok(_run) = self.run_lock.try_lock() else {
            (0..n).for_each(f);
            return;
        };

        fn invoke<F: Fn(usize) + Sync>(data: *const (), i: usize) {
            // SAFETY: `data` points to `f: F` on `run`'s stack for the batch.
            unsafe { (&*data.cast::<F>())(i) };
        }

        let job = Job {
            call: invoke::<F>,
            data: (&raw const f).cast(),
            n,
            next: AtomicUsize::new(0),
            panicked: Mutex::new(None),
        };

        struct WaitGuard<'a> {
            state: &'a (Mutex<PoolState>, std::sync::Condvar, std::sync::Condvar),
        }
        impl Drop for WaitGuard<'_> {
            fn drop(&mut self) {
                let (lock, _, done_cv) = self.state;
                let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
                while st.remaining > 0 {
                    st = done_cv.wait(st).unwrap_or_else(|e| e.into_inner());
                }
                st.job = std::ptr::null();
            }
        }

        {
            let (lock, work_cv, _) = &*self.state;
            let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
            st.job = &raw const job;
            st.wanted = workers;
            st.claimed = 0;
            st.remaining = workers;
            st.generation = st.generation.wrapping_add(1);
            drop(st);
            work_cv.notify_all();
        }
        let guard = WaitGuard { state: &self.state };
        run_job(&job);
        drop(guard);
        drop(_run);

        if let Some(payload) = job.panicked.into_inner().unwrap_or_else(|e| e.into_inner()) {
            std::panic::resume_unwind(payload);
        }
    }
}

#[cfg(test)]
impl Drop for ParallelPool {
    fn drop(&mut self) {
        let (lock, work_cv, _) = &*self.state;
        {
            let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
            st.shutdown = true;
        }
        work_cv.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn run_job(job: &Job) {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::Ordering::Relaxed;

    loop {
        let i = job.next.fetch_add(1, Relaxed);
        if i >= job.n {
            break;
        }
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| (job.call)(job.data, i))) {
            let mut slot = job.panicked.lock().unwrap_or_else(|e| e.into_inner());
            if slot.is_none() {
                *slot = Some(payload);
            }
            job.next.store(job.n, Relaxed);
            break;
        }
    }
}

fn max_parallel_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(4, |n| n.get() / 4)
        .clamp(1, 4)
}

/// Runs `f(i)` for each `i` below `n` on up to a quarter of the CPUs (four
/// at most: the title's own threads need the rest), here when that is
/// fewer than two.
pub(crate) fn in_parallel(n: usize, f: impl Fn(usize) + Sync) {
    static POOL: std::sync::OnceLock<ParallelPool> = std::sync::OnceLock::new();
    let max = max_parallel_threads();
    let threads = max.min(n);
    if threads < 2 {
        (0..n).for_each(f);
        return;
    }
    let pool = POOL.get_or_init(|| ParallelPool::new(max - 1));
    pool.run(threads, n, f);
}

/// [`Staging::refresh`] of each of `work`, those whose memory may have
/// changed on several threads at once (reading title memory and detiling
/// cost ~6 ms of frame a MB on one thread in a new shot, run evict5).
pub(crate) fn refresh_all(work: &[(Arc<Staging>, Arc<crate::resource::HostTexture>)]) -> Vec<Option<Upload>> {
    // Most find nothing changed (their trackers' clean bits say so): those
    // are not handed to threads.
    let changed: Vec<usize> = (0..work.len()).filter(|&i| work[i].0.maybe_changed()).collect();
    let results: Vec<Mutex<Option<Upload>>> = changed.iter().map(|_| Mutex::new(None)).collect();
    in_parallel(changed.len(), |k| {
        *results[k].lock().unwrap_or_else(|e| e.into_inner()) = work[changed[k]].0.refresh();
    });
    let mut out: Vec<Option<Upload>> = vec![None; work.len()];
    for (k, r) in results.into_iter().enumerate() {
        out[changed[k]] = r.into_inner().unwrap_or_else(|e| e.into_inner());
    }
    out
}

impl Staging {
    /// Staging bytes of subresource `i`: its footprint with the driver's
    /// padding up to the next one. Only the texture's last span can end
    /// unaligned, and in an increasing list it comes last, so spans packed
    /// back to back keep copy offsets aligned.
    fn span(&self, i: u32) -> u64 {
        let i = i as usize;
        let end = self.footprints.get(i + 1).map_or(self.size, |next| next.Offset);
        end.saturating_sub(self.footprints[i].Offset)
    }

    /// Forgets what the title changed so far: the host is about to own the contents.
    fn drain(&self) {
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(t) = &self.tracked {
            t.take(self.memory as usize, self.memory as usize + self.layout.size as usize);
            t.watch_overwrite();
            // The host texture no longer holds what was hashed. Hashes on
            // untracked pages stay: unchanged memory must not replace it.
            // shortcut: the kept hash is the last refresh's, so a title write
            // between it and here uploads over the host's texels; and a
            // partly tracked texture hashes whole subresources on every
            // sampling submission. Re-hash here, and hash only the untracked
            // bytes, if a title has such textures (none logged so far).
            let (start, size) = (self.memory as usize, self.layout.size as usize);
            let untracked = t.untracked(start, start + size);
            for (i, hash) in last.hashes.iter_mut().enumerate() {
                let (a, b) = self.spans.get(i).copied().unwrap_or((0, size));
                if !untracked.iter().any(|&(f, e)| f - start < b && a < e - start) {
                    *hash = None;
                }
            }
        }
    }

    /// Whether its memory may have changed since it was last detiled (an
    /// untracked texture always may).
    pub(crate) fn maybe_changed(&self) -> bool {
        // After initialization a writable stand-in keeps the GPU's contents.
        (self.packed.is_none() || self.uploads.load(std::sync::atomic::Ordering::Relaxed) == 0)
            && (self.retry.load(std::sync::atomic::Ordering::Relaxed)
                || self.tracked.as_ref().is_none_or(|t| !t.clean()))
    }

    /// A staging buffer with the texture memory detiled into it if it
    /// changed since the last time (and is mapped at all), and the
    /// subresources detiled: those that changed and could be read, which
    /// are the ones to copy.
    pub(crate) fn refresh(&self) -> Option<Upload> {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let (read, buffer) = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
        buffer
    }

    /// Returns the bytes of title memory read, and the buffer it detiled
    /// into with the subresources detiled.
    fn refresh_now(&self) -> (usize, Option<Upload>) {
        let all = || (0..self.footprints.len() as u32).collect::<Vec<_>>();
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some(t) = &self.tracked {
            // Most refreshes find nothing: the tracker's dense bit says so
            // without its lock (123,000 refreshes a frame, run evict3).
            let retry = self.retry.load(std::sync::atomic::Ordering::Relaxed);
            if !retry && t.clean() {
                return (0, None);
            }
            let ranges = t.take(start, start + size);
            let untracked = t.untracked(start, start + size);
            if !retry && ranges.is_empty() && untracked.is_empty() {
                return (0, None);
            }
            #[cfg(feature = "diagnostics")]
            if !untracked.is_empty() {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| tracing::warn!(memory = ?self.memory, size, "a partly tracked texture is sampled"));
            }
            // Only the subresources in the pages that changed (streamed
            // textures change page by page; a whole re-read and detile of
            // each cost up to 430 MB a frame, run integ2), and those on
            // untracked pages, which can change unseen.
            let n = self.footprints.len();
            let span = |i: usize| self.spans.get(i).copied().unwrap_or((0, size));
            let touched: Vec<(usize, bool)> = (0..n)
                .filter_map(|i| {
                    let (a, b) = span(i);
                    let overlaps = |&(f, e): &(usize, usize)| f - start < b && a < e - start;
                    let unseen = untracked.iter().any(overlaps);
                    (retry || unseen || ranges.iter().any(overlaps)).then_some((i, unseen))
                })
                .collect();
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page. Not kept: it would
            // double each texture's system memory. Only the touched spans
            // are written, and only they are hashed and detiled.
            let mut scratch = Scratch::take(size);
            let read = &mut scratch.0[..size];
            let mut spans: Vec<(usize, usize)> = touched.iter().map(|&(i, _)| span(i)).collect();
            merge_spans(&mut spans);
            let mut bytes = 0;
            for (a, b) in spans {
                // SAFETY: inside the buffer of the texture's size.
                crate::host::copy_mapped(
                    unsafe { read.as_mut_ptr().add(a) },
                    self.memory.wrapping_add(a),
                    b - a,
                    false,
                );
                bytes += b - a;
            }
            // A changed page that holds only the subresource's bytes says it
            // changed. Pages it shares with other data the title writes
            // (small textures), and untracked pages: its own bytes decide.
            // (Hashing a movie frame cost 8 ms a frame.)
            let page = crate::watch::PAGE;
            last.hashes.resize(n, None);
            let mut fresh: Vec<u32> = touched
                .into_iter()
                .filter(|&(i, unseen)| {
                    let (a, b) = span(i);
                    let (a0, b0) = (start + a, start + b);
                    if !unseen
                        && (retry || ranges.iter().any(|&(f, e)| f.max(a0).next_multiple_of(page) + page <= e.min(b0)))
                    {
                        last.hashes[i] = None;
                        return true;
                    }
                    let hash = blake3::hash(&read[a..b]);
                    last.hashes[i].replace(hash) != Some(hash) || retry
                })
                .map(|(i, _)| i as u32)
                .collect();
            if fresh.is_empty() {
                return (bytes, None);
            }
            self.retry.store(false, std::sync::atomic::Ordering::Relaxed);
            let buffer = self.detile(read, &mut fresh);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, bytes as u64);
            return (bytes, buffer.map(|(b, at)| (b, at, fresh)));
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            // Never uploaded: zeros, as a committed texture starts (one in
            // a heap of textures starts with whatever the range held).
            if last.hash.is_some() {
                return (0, None);
            }
            let zeros = vec![0u8; size];
            let mut every = all();
            let buffer = self.detile(&zeros, &mut every);
            last.hash = buffer.is_some().then(|| blake3::hash(&zeros));
            return (0, buffer.map(|(b, at)| (b, at, every)));
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return (size, None);
        }
        let mut every = all();
        let buffer = self.detile(tiled, &mut every);
        last.hash = buffer.is_some().then_some(hash);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        (size, buffer.map(|(b, at)| (b, at, every)))
    }

    /// Detiles `subresources` (increasing) of `tiled` (the texture's memory)
    /// into staging space, those detiled packed back to back, dropping from the list those that cannot be read
    /// (`None` if no buffer could be had or none was detiled).
    fn detile(&self, tiled: &[u8], subresources: &mut Vec<u32>) -> Option<(ID3D12Resource, u64)> {
        let size: u64 = subresources.iter().map(|&i| self.span(i)).sum();
        let Some((buffer, at, data)) = staging_space(size) else {
            // Space can be had later; a layout error below would only recur.
            self.retry.store(true, std::sync::atomic::Ordering::Relaxed);
            return None;
        };
        #[cfg(feature = "diagnostics")]
        // Scanned only when its message is wanted: a pass over every
        // texture read, in timing runs too.
        if tracing::enabled!(tracing::Level::DEBUG) {
            if tiled.iter().all(|&b| b == 0) {
                tracing::debug!(
                    memory = ?self.memory,
                    desc = ?self.layout.desc,
                    "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
                );
            } else {
                tracing::debug!(memory = ?self.memory, desc = ?self.layout.desc, "a sampled texture detiled");
            }
        }
        // Advanced past detiled subresources only, as `copies` walks the list.
        let mut next = 0u64;
        subresources.retain(|&i| {
            let span = self.span(i);
            let (footprint, rows) = (&self.footprints[i as usize], self.rows[i as usize]);
            let (mip, slice) = (i % self.mips, i / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = subresource_len(pitch, rows, f.Depth, span);
            // SAFETY: `next + len` is at most the sum of the requested spans, so
            // the slice stays inside the staging space even when
            // GetCopyableFootprints does not pad the final row of the last subresource.
            let linear = unsafe { std::slice::from_raw_parts_mut(data.add(next as usize), len) };
            let detile = |dst: &mut [u8], pitch| self.layout.detile(tiled, mip, slice, dst, pitch, pitch * rows as usize);
            let result = if let Some(format) = self.packed {
                let row = f.Width as usize * crate::resource::element(format).map_or(2, |e| e.0 as usize);
                let mut packed = vec![0; row * rows as usize * f.Depth as usize];
                detile(&mut packed, row).map(|()| {
                    for (src, dst) in packed.chunks_exact(row).zip(linear.chunks_mut(pitch)) {
                        expand_row(format, src, dst);
                    }
                })
            } else {
                detile(linear, pitch)
            };
            let done = result
                .inspect_err(|e| tracing::warn!(?e, mip, slice, "texture not detiled"))
                .is_ok();
            if done {
                next += span;
            }
            done
        });
        (!subresources.is_empty()).then_some((buffer, at))
    }
}

/// Sorts `[a, b)` byte spans and merges overlapping or adjacent intervals
/// (dropping empty spans) so each covered byte is read once.
fn merge_spans(spans: &mut Vec<(usize, usize)>) {
    spans.retain(|&(a, b)| a < b);
    if spans.len() <= 1 {
        return;
    }
    spans.sort_unstable();
    let mut write = 0;
    for read in 1..spans.len() {
        let (a, b) = spans[read];
        if a <= spans[write].1 {
            spans[write].1 = spans[write].1.max(b);
        } else {
            write += 1;
            spans[write] = (a, b);
        }
    }
    spans.truncate(write + 1);
}

/// A buffer for the title memory a refresh reads, kept by the thread for
/// its next refresh: a fresh one for each texture cost a zero-fill and a
/// page fault per 4 KiB (129,000 faults in a frame, run placed1).
struct Scratch(Vec<u8>);

thread_local! {
    static SCRATCH: std::cell::Cell<Vec<u8>> = const { std::cell::Cell::new(Vec::new()) };
}

impl Scratch {
    /// The thread's buffer, at least `size` bytes.
    fn take(size: usize) -> Self {
        let mut v = SCRATCH.take();
        if v.len() < size {
            v.resize(size, 0);
        }
        Scratch(v)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // The largest textures' buffers are not kept.
        const KEEP: usize = 32 << 20;
        if self.0.len() <= KEEP {
            SCRATCH.set(std::mem::take(&mut self.0));
        }
    }
}

/// Bytes of a subresource's staging span of `span` bytes to detile into.
/// D3D12's `GetCopyableFootprints` omits trailing row-pitch padding on the
/// final row of the final subresource, so `pitch * rows * depth` can exceed
/// the span.
fn subresource_len(pitch: usize, rows: u32, depth: u32, span: u64) -> usize {
    let padded = pitch
        .saturating_mul(rows as usize)
        .saturating_mul(depth as usize);
    let remaining = usize::try_from(span).unwrap_or(usize::MAX);
    padded.min(remaining)
}

/// Whether `[at, at + len)` is committed, readable memory (streamed textures
/// may be unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let mut p = at as usize;
    let Some(end) = p.checked_add(len) else {
        return false;
    };
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let unreadable = info.Protect == 0
            || info.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0;
        if n == 0 || info.State != MEM_COMMIT || unreadable {
            return false;
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= p {
            return false;
        }
        p = next;
    }
    true
}

/// The staging state of a title texture's host texture, made on first use
/// (one for all title textures of its memory).
pub(crate) fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let copy = crate::resource::host_copy(resource)?;
    let mut held = copy.staging.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = held.as_ref() {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    // Render-target XOR layouts of one 2D slice and mip have addresses
    // (`Layout::addressable`); the other XOR modes do not.
    if !layout.addressable() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let layout_size = layout.size as usize;
    let spans = spans(&layout, count, u32::from(d.MipLevels));
    let s = Arc::new(Staging {
        size: total,
        packed: copy.packed(),
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        tracked: crate::watch::track(memory as usize, layout_size, true),
        spans,
        last: Mutex::default(),
        retry: Default::default(),
        uploads: Default::default(),
        copied: Default::default(),
    });
    *held = Some(s.clone());
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stand-in conversion takes 2D one-slice copies only.
    #[test]
    fn stand_in_copy_fits_2d_one_slice() {
        let tex = |dim| D3D12_RESOURCE_DESC { Dimension: dim, ..Default::default() };
        let (two, three) = (tex(D3D12_RESOURCE_DIMENSION_TEXTURE2D), tex(D3D12_RESOURCE_DIMENSION_TEXTURE3D));
        let deep = |front, back| D3D12_BOX { front, back, right: 4, bottom: 4, ..Default::default() };
        assert!(stand_in_copy_fits(&two, &two, None));
        assert!(stand_in_copy_fits(&two, &two, Some(&deep(0, 1))));
        assert!(!stand_in_copy_fits(&two, &two, Some(&deep(0, 2))));
        assert!(!stand_in_copy_fits(&three, &two, None));
        assert!(!stand_in_copy_fits(&two, &three, None));
    }

    /// A title-written texture in a render-target XOR layout (256x256
    /// RGBA8, `SW_64KB_R_X`) is staged: the bytes read back from the upload
    /// buffer are the formats crate's detile of its memory.
    #[test]
    fn a_render_target_xor_texture_is_staged() {
        let Some(mut staging) = xor_staging() else {
            eprintln!("no host device; skipped");
            return;
        };
        let memory: Vec<u8> = (0..staging.layout.size as usize)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 11) as u8)
            .collect();
        staging.memory = memory.as_ptr();
        let (footprint, rows) = (staging.footprints[0], staging.rows[0]);
        let (_, upload) = staging.refresh_now();
        let (buffer, at, list) = upload.expect("staged");
        assert_eq!(list, [0]);
        let pitch = footprint.Footprint.RowPitch as usize;
        let mut want = vec![0; pitch * rows as usize];
        let len = want.len();
        staging
            .layout
            .detile(&memory, 0, 0, &mut want, pitch, len)
            .unwrap();
        let mut mapped = std::ptr::null_mut();
        // SAFETY: an upload buffer, mapped to read what was staged.
        unsafe { buffer.Map(0, None, Some(&mut mapped)) }.unwrap();
        let at = (footprint.Offset + at) as usize;
        for row in 0..rows as usize {
            // SAFETY: the staged rows of the subresource.
            let got = unsafe {
                std::slice::from_raw_parts(mapped.cast::<u8>().add(at + row * pitch), 256 * 4)
            };
            assert_eq!(got, &want[row * pitch..][..256 * 4], "row {row}");
        }
        // SAFETY: unmaps what was mapped above.
        unsafe { buffer.Unmap(0, None) };
    }

    /// A detile that fails after the tracker's pages were taken is tried
    /// again, with no further write to the memory.
    #[test]
    fn a_failed_detile_of_tracked_memory_is_retried() {
        use crate::watch::{PAGE, tests as watch};
        let Some(mut staging) = xor_staging() else {
            eprintln!("no host device; skipped");
            return;
        };
        let size = staging.layout.size as usize;
        let p = watch::pool(size.div_ceil(PAGE));
        staging.memory = p as *const u8;
        staging.tracked = crate::watch::track(p, size, true);
        assert!(staging.tracked.is_some());
        // No staging space (no buffer is this large): the detile fails.
        let total = std::mem::replace(&mut staging.size, 1 << 62);
        assert!(staging.refresh_now().1.is_none());
        assert!(staging.tracked.as_ref().unwrap().clean());
        assert!(staging.maybe_changed());
        staging.size = total;
        let (_, _, list) = staging.refresh_now().1.expect("retried");
        assert_eq!(list, [0]);
        assert!(!staging.maybe_changed());
        assert!(staging.refresh_now().1.is_none());
        // No room for the subresource: a layout error, not tried again.
        staging.size = 0;
        drop(crate::watch::writing(p, PAGE));
        assert!(staging.maybe_changed());
        assert!(staging.refresh_now().1.is_none());
        assert!(!staging.maybe_changed());
        drop(staging);
        watch::free(p);
    }

    /// A write to the untracked page of a partly tracked texture is staged,
    /// though the tracker reports nothing.
    #[test]
    fn a_write_to_an_untracked_page_is_staged() {
        use crate::watch::PAGE;
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        let Some(mut staging) = xor_staging() else {
            eprintln!("no host device; skipped");
            return;
        };
        let size = staging.layout.size as usize;
        // SAFETY: a new reservation, decommitted below.
        let p = unsafe { VirtualAlloc(std::ptr::null(), size, MEM_RESERVE, PAGE_READWRITE) };
        assert!(!p.is_null());
        // Plain committed memory is untracked; the reserved tail is tracked.
        // SAFETY: committing the first page of the reservation above.
        assert_eq!(unsafe { VirtualAlloc(p, PAGE, MEM_COMMIT, PAGE_READWRITE) }, p);
        staging.memory = p as *const u8;
        staging.tracked = crate::watch::track(p as usize, size, true);
        assert!(staging.tracked.is_some());
        assert!(staging.refresh_now().1.is_some());
        assert!(staging.refresh_now().1.is_none());
        // SAFETY: inside the committed page.
        unsafe { p.cast::<u8>().write(0xff) };
        let (_, _, list) = staging.refresh_now().1.expect("staged");
        assert_eq!(list, [0]);
        // A host write: unchanged memory does not replace it.
        staging.drain();
        assert!(staging.refresh_now().1.is_none());
        drop(staging);
        crate::watch::tests::free(p as usize);
    }

    /// A subresource that cannot be detiled does not take the others of the
    /// same refresh with it.
    #[test]
    fn a_subresource_not_detiled_leaves_the_others_staged() {
        let Some(mut staging) = xor_staging() else {
            eprintln!("no host device; skipped");
            return;
        };
        let size = staging.layout.size as usize;
        let memory = vec![1u8; size];
        staging.memory = memory.as_ptr();
        // A second subresource the layout does not have.
        let mut second = staging.footprints[0];
        second.Offset = staging.size;
        staging.size *= 2;
        staging.footprints.push(second);
        staging.rows.push(staging.rows[0]);
        staging.spans = vec![(0, size); 2];
        let (_, _, list) = staging.refresh_now().1.expect("staged");
        assert_eq!(list, [0]);
    }

    /// The regression test of the packed staging space: only detiled
    /// subresources take it, so one after a failed one starts where
    /// [`copies`] looks for it.
    #[test]
    fn a_subresource_not_detiled_takes_no_staging_space() {
        use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
        let Some(mut staging) = xor_staging() else {
            eprintln!("no host device; skipped");
            return;
        };
        // Two linear slices; the first one's span is too small to detile into.
        staging.layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 256,
            height: 256,
            depth: 2,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::LINEAR,
        })
        .unwrap();
        let size = staging.layout.size as usize;
        let mut memory = vec![1u8; size];
        memory[size / 2..].fill(2);
        staging.memory = memory.as_ptr();
        let mut second = staging.footprints[0];
        second.Offset = 512;
        staging.size += 512;
        staging.footprints.push(second);
        staging.rows.push(staging.rows[0]);
        staging.spans = vec![(0, size); 2];
        let (buffer, at, list) = staging.refresh_now().1.expect("staged");
        assert_eq!(list, [1]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: an upload buffer, mapped to read what was staged.
        unsafe { buffer.Map(0, None, Some(&mut mapped)) }.unwrap();
        // SAFETY: the staged subresource's first row.
        let got = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>().add(at as usize), 256 * 4) };
        assert_eq!(got, &memory[size / 2..][..256 * 4]);
        // SAFETY: unmaps what was mapped above.
        unsafe { buffer.Unmap(0, None) };
    }

    /// The staging state of a 256x256 RGBA8 `SW_64KB_R_X` texture with no
    /// memory yet (`None` without a host device).
    fn xor_staging() -> Option<Staging> {
        use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
        };
        let host = host()?;
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 256,
            height: 256,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_64KB_R_X,
        })
        .unwrap();
        assert!(layout.addressable());
        let d = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let (mut rows, mut total) = (0u32, 0);
        // SAFETY: outputs for one subresource.
        unsafe {
            host.device.GetCopyableFootprints(
                &d,
                0,
                1,
                0,
                Some(&mut footprint),
                Some(&mut rows),
                None,
                Some(&mut total),
            );
        }
        Some(Staging {
            size: total,
            spans: spans(&layout, 1, 1),
            footprints: vec![footprint],
            rows: vec![rows],
            layout,
            mips: 1,
            memory: std::ptr::null(),
            tracked: None,
            last: Mutex::default(),
            retry: Default::default(),
            uploads: Default::default(),
            copied: Default::default(),
            packed: None,
        })
    }

    /// R9G9B9E5 texels expand to R11G11B10 float and pack back: exact
    /// where both formats hold the value, clamped to the 11- and 10-bit
    /// maxima, denormals kept, NaN and negatives 0.
    #[test]
    fn shared_exponent_texels_round_trip_through_the_stand_in() {
        let floats = |b: u32| {
            let scale = 2f32.powi((b >> 27) as i32 - 24);
            [0, 9, 18].map(|s| (b >> s & 0x1ff) as f32 * scale)
        };
        let tiny = 2f32.powi(-20); // the smallest 11-bit denormal
        for (c, want) in [
            ([1.0, 0.5, 0.25], [1.0, 0.5, 0.25]),
            ([0.0; 3], [0.0; 3]),
            ([65408.0; 3], [65024.0, 65024.0, 64512.0]),
            ([tiny, 0.0, 0.0], [tiny, 0.0, 0.0]),
            ([-1.0, f32::NAN, 1.0], [0.0, 0.0, 1.0]),
        ] {
            let word = rgb9e5(c);
            let mut host = [0; 4];
            expand_row(67, &word.to_le_bytes(), &mut host);
            let h = u32::from_le_bytes(host);
            let got = [small_float(h & 0x7ff, 6), small_float(h >> 11 & 0x7ff, 6), small_float(h >> 22, 5)];
            assert_eq!(got, want, "{c:?}");
            let mut back = [0; 4];
            pack_row(67, &host, &mut back);
            for (b, w) in floats(u32::from_le_bytes(back)).into_iter().zip(want) {
                assert!((b - w).abs() <= w / 256.0, "{c:?}: {b} for {w}");
            }
        }
        assert_eq!(small_bits(-1.0, 6), 0);
        assert_eq!(small_bits(f32::NAN, 5), 0);
        assert_eq!(small_bits(1e9, 6), 0x7bf);
        assert_eq!(small_bits(1e9, 5), 0x3df);
        // A NaN R11G11B10 channel packs to 0.
        assert_eq!(r11g11b10_to_rgb9e5(0x7c1) & 0x1ff, 0);
        // A mantissa rounding to 512 takes the next exponent.
        assert_eq!(floats(rgb9e5([511.9, 0.0, 0.0]))[0], 512.0);
    }

    #[test]
    fn packed_colors_round_trip_with_their_alpha() {
        for (format, word, expected) in [
            (86, 0x8443u16, [8, 16, 25, 255]),
            (115, 0xa123, [17, 34, 51, 170]),
        ] {
            let mut rgba = [0; 4];
            let mut packed = [0; 2];
            expand_row(format, &word.to_le_bytes(), &mut rgba);
            assert_eq!(rgba, expected);
            for word in 0..=u16::MAX {
                expand_row(format, &word.to_le_bytes(), &mut rgba);
                pack_row(format, &rgba, &mut packed);
                assert_eq!(u16::from_le_bytes(packed), word);
            }
        }
        for (alpha, high) in [(127, 0), (128, 0x80)] {
            let mut packed = [0; 2];
            pack_row(86, &[0, 0, 0, alpha], &mut packed);
            assert_eq!(packed, [0, high]);
        }
    }

    #[test]
    fn subresource_len_stays_inside_staging_total() {
        // A 4x4 RGBA8 texture has RowPitch=256, rows=4, last_row=16 -> total=784.
        // Padded size (1024) must be clamped to 784 so from_raw_parts_mut stays in bounds.
        assert_eq!(subresource_len(256, 4, 1, 784), 784);
        assert_eq!(subresource_len(256, 4, 1, 272), 272);
        assert_eq!(subresource_len(256, 4, 1, 2048), 1024);
    }

    #[test]
    fn committed_rejects_overflow_and_noaccess_pages() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        assert!(!committed(usize::MAX as *const u8, 16));
        // SAFETY: reserving and committing one private page for the test, freed below.
        let noaccess =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS) };
        assert!(!noaccess.is_null());
        assert!(!committed(noaccess.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(noaccess, 0, MEM_RELEASE) };

        // Negative control: a readable committed page still passes.
        // SAFETY: reserving and committing one private page for the test, freed below.
        let rw =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!rw.is_null());
        assert!(committed(rw.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(rw, 0, MEM_RELEASE) };
    }

    #[cfg(test)]
    struct HeldCom(This);
    #[cfg(test)]
    impl Drop for HeldCom {
        fn drop(&mut self) {
            crate::object::slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
        }
    }

    #[cfg(test)]
    fn test_title_texture(addr: u64, fill: u8) -> HeldCom {
        const SIDE: usize = 64;
        const BYTES: usize = SIDE * SIDE * 4;
        // SAFETY: `pool(1)` provides a 64 KiB RW page for each 64x64 RGBA8 linear texture.
        unsafe { std::ptr::write_bytes(addr as *mut u8, fill, BYTES) };
        let words: [u32; 14] = [3, 0, 0, 0, SIDE as u32, 0, SIDE as u32, 1 | (1 << 16), 28, 1, 0, 0x100, 0, 0];
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let mut out = std::ptr::null();
        let hr = crate::resource::create_at_address(std::ptr::null(), addr, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut out);
        assert_eq!(hr, crate::object::S_OK);
        HeldCom(out)
    }

    #[cfg(test)]
    fn seed_title_texture(host: &crate::host::Host, tex: This) {
        let seed_list = host.new_list().unwrap();
        let host_tex = crate::resource::host_resource(tex).unwrap();
        upload(&seed_list, tex, &host_tex);
        seed_list.close();
        host.execute(&[&seed_list]);
    }

    #[cfg(test)]
    fn readback_first_texel(host: &crate::host::Host, tex: This) -> [u8; 4] {
        let read_list = host.new_list().unwrap();
        let tex_after = crate::resource::host_resource(tex).unwrap();
        upload(&read_list, tex, &tex_after);
        crate::list::ensure_state(&read_list, tex, 0x800);
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0;
        // SAFETY: live host texture; outputs for subresource 0.
        unsafe {
            host.device.GetCopyableFootprints(&tex_after.GetDesc(), 0, 1, 0, Some(&mut footprint), None, None, Some(&mut total));
        }
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain readback buffer creation and copy recording.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: total,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed pointer for recording only.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
        };
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed pointer for recording only.
            pResource: unsafe { std::mem::transmute_copy(&tex_after) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        // SAFETY: open host command list and live resources in copy states.
        unsafe {
            read_list.list.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
        }
        read_list.close();
        host.execute(&[&read_list]);

        let mut mapped = std::ptr::null_mut();
        // SAFETY: synchronous execution finished; `readback` covers all rows.
        unsafe {
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            let first = mapped.cast::<u8>().add(footprint.Offset as usize).cast::<[u8; 4]>().read();
            readback.Unmap(0, None);
            first
        }
    }

    /// A texture fed from title memory cannot be evicted while a recorded list
    /// holds a pending `host_writes` mark for it; resetting/dropping an
    /// unexecuted list releases that pin, while executing the list keeps the
    /// copied GPU texels over stale title memory.
    #[test]
    fn pending_host_write_pins_texture_until_executed_and_unpins_on_reset() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let dst_mem = crate::watch::tests::pool(1);
        let src_mem = crate::watch::tests::pool(1);
        let dst = test_title_texture(dst_mem as u64, 0x40);
        let src = test_title_texture(src_mem as u64, 0xAA);

        let null = std::ptr::null();
        let (mut alloc, mut list_drop, mut list_exec) = (null, null, null);
        let alloc_iid = crate::object::guid(0x6102dee4, 0xaf59, 0x4b09, [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24]);
        let list_iid = crate::object::guid(0x5b160d0f, 0xac1b, 0x4185, [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55]);
        assert_eq!(crate::list::create_allocator(null, 0, &alloc_iid, &mut alloc), crate::object::S_OK);
        let _alloc = HeldCom(alloc);
        assert_eq!(crate::list::create_list(null, 0, 0, alloc, null, &list_iid, &mut list_drop), crate::object::S_OK);
        let _l_drop = HeldCom(list_drop);
        assert_eq!(crate::list::create_list(null, 0, 0, alloc, null, &list_iid, &mut list_exec), crate::object::S_OK);
        let _l_exec = HeldCom(list_exec);

        type CloseFn = extern "system" fn(This) -> crate::object::HRESULT;
        type ResetFn = extern "system" fn(This, This, This) -> crate::object::HRESULT;
        type CopyResourceFn = extern "system" fn(This, This, This);
        let close_drop = crate::object::slot::<CloseFn>(list_drop, 0x48);
        let reset_drop = crate::object::slot::<ResetFn>(list_drop, 0x50);
        let copy_drop = crate::object::slot::<CopyResourceFn>(list_drop, 0x88);
        let close_exec = crate::object::slot::<CloseFn>(list_exec, 0x48);
        let reset_exec = crate::object::slot::<ResetFn>(list_exec, 0x50);
        let copy_exec = crate::object::slot::<CopyResourceFn>(list_exec, 0x88);

        seed_title_texture(host, dst.0);

        copy_drop(list_drop, dst.0, src.0);
        assert!(!crate::resource::give_up_now(dst.0), "pending CopyResource must pin dst against eviction");
        assert_eq!(close_drop(list_drop), crate::object::S_OK);
        assert_eq!(reset_drop(list_drop, alloc, null), crate::object::S_OK);
        assert!(crate::resource::give_up_now(dst.0), "resetting unexecuted list must allow eviction again");

        seed_title_texture(host, dst.0);

        copy_exec(list_exec, dst.0, src.0);
        assert_eq!(close_exec(list_exec), crate::object::S_OK);
        assert!(!crate::resource::give_up_now(dst.0), "pending L1 CopyResource must block eviction before L1 runs");

        let recorded = crate::list::host_list(list_exec).unwrap();
        host.execute(&[recorded]);
        assert_eq!(reset_exec(list_exec, alloc, null), crate::object::S_OK);
        assert!(!crate::resource::give_up_now(dst.0), "executed host write must keep dst un-evictable after list reset");

        assert_eq!(readback_first_texel(host, dst.0), [0xAA, 0xAA, 0xAA, 0xAA]);
    }

    /// Window 2: an eviction candidate snapshot taken before a list pins `dst`
    /// must not evict `dst` in `give_up`, and the recorded copy must keep its
    /// host texels on subsequent reads.
    #[test]
    fn stale_eviction_snapshot_skips_texture_pinned_after_candidates() {
        use std::sync::atomic::Ordering::Acquire;
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let dst_mem = crate::watch::tests::pool(1);
        let src_mem = crate::watch::tests::pool(1);
        let dst = test_title_texture(dst_mem as u64, 0x40);
        let src = test_title_texture(src_mem as u64, 0xAA);

        let null = std::ptr::null();
        let (mut alloc, mut list) = (null, null);
        let alloc_iid = crate::object::guid(0x6102dee4, 0xaf59, 0x4b09, [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24]);
        let list_iid = crate::object::guid(0x5b160d0f, 0xac1b, 0x4185, [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55]);
        assert_eq!(crate::list::create_allocator(null, 0, &alloc_iid, &mut alloc), crate::object::S_OK);
        let _alloc = HeldCom(alloc);
        assert_eq!(crate::list::create_list(null, 0, 0, alloc, null, &list_iid, &mut list), crate::object::S_OK);
        let _list = HeldCom(list);

        type CloseFn = extern "system" fn(This) -> crate::object::HRESULT;
        type ResetFn = extern "system" fn(This, This, This) -> crate::object::HRESULT;
        type CopyResourceFn = extern "system" fn(This, This, This);
        let close = crate::object::slot::<CloseFn>(list, 0x48);
        let reset = crate::object::slot::<ResetFn>(list, 0x50);
        let copy = crate::object::slot::<CopyResourceFn>(list, 0x88);

        seed_title_texture(host, dst.0);
        let snap = crate::resource::owned(dst.0).expect("unpinned texture is an eviction candidate");

        // Record after `candidates()` snapshot was taken.
        copy(list, dst.0, src.0);
        let pinned = crate::resource::made_copy(dst.0).unwrap();
        assert_eq!(crate::resource::give_up_snapshot(&[snap]), (0, 0));

        let still = crate::resource::made_copy(dst.0).expect("stale snapshot must not evict pinned copy");
        assert!(Arc::ptr_eq(&pinned, &still));
        assert_eq!(still.pending_host_writes.load(Acquire) & crate::resource::EVICTED, 0);

        assert_eq!(close(list), crate::object::S_OK);
        let recorded = crate::list::host_list(list).unwrap();
        host.execute(&[recorded]);
        assert_eq!(reset(list, alloc, null), crate::object::S_OK);
        assert_eq!(readback_first_texel(host, dst.0), [0xAA, 0xAA, 0xAA, 0xAA]);
    }

    /// Window 1: when `give_up` has already claimed `old` with `EVICTED`
    /// before clearing `resource(dst).host`, `prepare_host_write` must never
    /// return `old`; it detaches `old` and atomically acquires + pins a fresh
    /// replacement `HostTexture`.
    #[test]
    fn prepare_host_write_detaches_claimed_evicted_copy_and_pins_replacement() {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        use windows::core::Interface;
        if host().is_none() {
            return; // no host GPU here
        }
        let dst_mem = crate::watch::tests::pool(1);
        let dst = test_title_texture(dst_mem as u64, 0x40);
        let old = crate::resource::host_copy(dst.0).unwrap();

        // Simulate `give_up` claiming `old` via CAS before clearing `resource.host`.
        assert_eq!(
            old.pending_host_writes.compare_exchange(0, crate::resource::EVICTED, AcqRel, Acquire),
            Ok(0)
        );

        let guard = prepare_host_write(dst.0).expect("must detach EVICTED copy and acquire replacement");
        assert_ne!(guard.resource().as_raw(), old.texture.as_raw());
        let current = crate::resource::made_copy(dst.0).unwrap();
        assert!(!Arc::ptr_eq(&current, &old));
        assert!(!crate::resource::give_up_now(dst.0), "replacement must be pinned while guard lives");

        drop(guard);
        assert!(crate::resource::give_up_now(dst.0), "dropping unrecorded guard must unpin replacement");
    }

    #[test]
    fn merge_spans_coalesces_overlapping_and_adjacent_intervals() {
        let mut s = vec![
            (100, 200),
            (50, 50),
            (0, 64),
            (64, 100),
            (300, 400),
            (350, 380),
            (380, 450),
            (0, 64),
        ];
        merge_spans(&mut s);
        assert_eq!(s, vec![(0, 200), (300, 450)]);

        let mut empty = vec![(10, 10), (20, 10)];
        merge_spans(&mut empty);
        assert!(empty.is_empty());
    }

    #[test]
    fn parallel_pool_reuses_worker_scratch_and_handles_reentrancy_and_panics() {
        use std::collections::HashSet;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let pool = ParallelPool::new(3);
        let sum = AtomicUsize::new(0);
        let first_ptrs: Mutex<HashSet<usize>> = Mutex::new(HashSet::new());
        pool.run(4, 64, |i| {
            let s = Scratch::take(64 * 1024);
            first_ptrs.lock().unwrap().insert(s.0.as_ptr() as usize);
            sum.fetch_add(i + 1, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 64 * 65 / 2);

        // Second batch reuses the already-allocated thread_local Scratch buffers.
        let mut second_ptrs = HashSet::new();
        let second_lock: Mutex<&mut HashSet<usize>> = Mutex::new(&mut second_ptrs);
        pool.run(4, 64, |_| {
            let s = Scratch::take(64 * 1024);
            second_lock.lock().unwrap().insert(s.0.as_ptr() as usize);
        });
        let first_ptrs = first_ptrs.into_inner().unwrap();
        assert!(
            second_ptrs.iter().any(|p| first_ptrs.contains(p)),
            "expected persistent worker Scratch pointer reuse across batches: first={first_ptrs:?} second={second_ptrs:?}"
        );

        // Nested call runs inline without deadlocking.
        let nested = AtomicUsize::new(0);
        pool.run(4, 8, |_| {
            pool.run(4, 4, |_| {
                nested.fetch_add(1, Ordering::Relaxed);
            });
        });
        assert_eq!(nested.load(Ordering::Relaxed), 32);

        // Panic propagates cleanly and leaves pool reusable afterward.
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.run(4, 16, |i| {
                assert_ne!(i, 5, "boom");
            });
        }));
        assert!(caught.is_err());

        let after = AtomicUsize::new(0);
        pool.run(4, 16, |_| {
            after.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(after.load(Ordering::Relaxed), 16);
    }

    #[test]
    #[ignore = "benchmark"]
    fn bench_in_parallel_scratch_reuse() {
        use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

        const BATCHES: usize = 40;
        const ITEMS: usize = 8;
        const BYTES: usize = 2 << 20; // 2 MiB texture read buffer

        let run_scoped = || {
            let t0 = std::time::Instant::now();
            for _ in 0..BATCHES {
                let next = AtomicUsize::new(0);
                std::thread::scope(|scope| {
                    for _ in 0..4 {
                        scope.spawn(|| loop {
                            let i = next.fetch_add(1, Relaxed);
                            if i >= ITEMS {
                                break;
                            }
                            let mut s = Scratch::take(BYTES);
                            // Touch one byte per 4 KiB page (as copy_mapped writes into read).
                            for p in (0..BYTES).step_by(4096) {
                                s.0[p] = (i as u8).wrapping_add(1);
                            }
                            std::hint::black_box(s.0[0]);
                        });
                    }
                });
            }
            t0.elapsed().as_secs_f64() * 1e3 / BATCHES as f64
        };

        let pool = ParallelPool::new(3);
        let run_pooled = || {
            // Warm once (simulating steady-state submissions after frame 1).
            pool.run(4, ITEMS, |_| {
                let _ = Scratch::take(BYTES);
            });
            let t0 = std::time::Instant::now();
            for _ in 0..BATCHES {
                pool.run(4, ITEMS, |i| {
                    let mut s = Scratch::take(BYTES);
                    for p in (0..BYTES).step_by(4096) {
                        s.0[p] = (i as u8).wrapping_add(1);
                    }
                    std::hint::black_box(s.0[0]);
                });
            }
            t0.elapsed().as_secs_f64() * 1e3 / BATCHES as f64
        };

        let scoped_ms = run_scoped();
        let pooled_ms = run_pooled();
        eprintln!(
            "in_parallel (4 threads, {ITEMS}x 2 MiB Scratch): scoped={scoped_ms:.3} ms/batch, pooled={pooled_ms:.3} ms/batch ({:.2}x)",
            scoped_ms / pooled_ms
        );
    }
}
