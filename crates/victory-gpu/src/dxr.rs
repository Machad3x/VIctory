//! The title's ray tracing on the host's DXR (`relatorios/design-dxr.md`).
//!
//! The title builds its acceleration structures with `XboxBvhBuilder.dll`,
//! whose interface is the D3D12 DXR build API (the title passes it
//! `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC`s), and walks them in
//! its ray shaders, one node at a time (`image_bvh64_intersect_ray`,
//! `victory_shader`'s `lower::bvh`). To trace on the host's ray tracing
//! hardware instead, the host needs structures of its own, built from the
//! same inputs.
//!
//! `VICTORY_DXR` ([`mode`]): `0` (the default) changes nothing; `1` and
//! `check` turn on the stages landed so far. Here, stages S0 and S1:
//! * the switch and the counters ([`log_frame`]);
//! * capture of the builder's calls: the title's imports of three of the
//!   builder's methods are bound to the wrappers below ([`hook`], from the
//!   application's import resolver), which read what the title asks for
//!   and then call the real method, so the builder runs as before.
//!
//! The batched build's desc (`D3D12XBOX_BUILD_RAYTRACING_ACCELERATION_
//! STRUCTURE_BATCHED_DESC2`) is the console's own and its layout is not
//! public: the wrapper looks for the standard desc in it ([`find_descs`])
//! and logs its raw bytes (the first calls, then one in [`RAW_EVERY`]),
//! with those of the memory its first qwords point to. Nothing is built on
//! the host yet.
//!
//! What a run's log showed of that layout ([`Element`]): elements of 80
//! bytes, the standard desc first, then the number of postbuild info descs
//! and their address (standard ones: where the answer goes, and what is
//! asked).
//!
//! **Sizes answered at the call** (`VICTORY_BVH_SIZE_ANSWER`, on unless
//! `0`, with the builder's stages on; needs no `VICTORY_DXR`). The title
//! asks for a new structure's compacted size in a postbuild info desc; the
//! builder has a program on the card store it, and the title reads the slot
//! a frame or two later without looking whether the answer is there, takes
//! memory of that size and has the structure copied into it. Our
//! submissions land late, so it read another structure's size and the copy
//! wrote over its neighbours. Here the slot holds an answer before the
//! batched build returns: the builder's own
//! `ResultDataMaxSizeInBytes` for the same inputs (what the title was told
//! by `GetRaytracingAccelerationStructurePrebuildInfo`, a fourth wrapper,
//! or the builder asked again), which no compacted structure exceeds. What
//! the builder's program stores into an answered slot later is kept out of
//! title memory ([`keep_answers`]): it may be an older structure's. The
//! builder's dispatches run as before.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// How far DXR is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Not at all: the title's own traversal, emulated.
    Off,
    /// Wherever a stage allows.
    On,
    /// Both ways, compared.
    Check,
}

/// `VICTORY_DXR`'s meaning: `1` on, `check` both ways compared, anything
/// else (and unset) off.
pub fn parse_mode(value: Option<&str>) -> Mode {
    match value.map(str::trim) {
        Some("1") => Mode::On,
        Some(v) if v.eq_ignore_ascii_case("check") => Mode::Check,
        _ => Mode::Off,
    }
}

/// The mode this run was started with (`VICTORY_DXR`).
pub fn mode() -> Mode {
    static MODE: std::sync::LazyLock<Mode> =
        std::sync::LazyLock::new(|| parse_mode(std::env::var("VICTORY_DXR").ok().as_deref()));
    *MODE
}

/// `VICTORY_BVH_SIZE_ANSWER`'s meaning: on unless `0`.
pub fn parse_answers(value: Option<&str>) -> bool {
    value.is_none_or(|v| v.trim() != "0")
}

/// Whether compacted sizes are answered at the call (see the module):
/// `VICTORY_BVH_SIZE_ANSWER` unless `0`, in runs with the builder's stages.
pub fn answers() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        parse_answers(std::env::var("VICTORY_BVH_SIZE_ANSWER").ok().as_deref()) && crate::pipeline::builder_allowed()
    });
    *ON
}

/// `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC`, as the title
/// passes it (48 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuildDesc {
    /// Where the structure goes (title address, 256-byte aligned).
    pub dest: u64,
    pub inputs: Inputs,
    /// The structure a refit or an update reads (0: none).
    pub source: u64,
    pub scratch: u64,
}

/// `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS` (24 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inputs {
    /// A top-level structure (instances), else a bottom-level one
    /// (geometry).
    pub top: bool,
    /// `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS`.
    pub flags: u32,
    /// Instances, or geometry descs.
    pub count: u32,
    /// The descs are an array of pointers (`DescsLayout` 1), else an array.
    pub pointers: bool,
    /// The instance descs' address (a GPU address), or the geometry descs'
    /// (title CPU memory).
    pub descs: u64,
}

/// `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE`.
pub const PERFORM_UPDATE: u32 = 0x20;
/// `…_ALLOW_COMPACTION`.
pub const ALLOW_COMPACTION: u32 = 0x2;
pub const BUILD_DESC_BYTES: usize = 48;
pub const INPUTS_BYTES: usize = 24;
pub const GEOMETRY_DESC_BYTES: usize = 56;
/// `D3D12_RAYTRACING_INSTANCE_DESC`.
pub const INSTANCE_DESC_BYTES: usize = 64;

fn dword(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn qword(bytes: &[u8], at: usize) -> u64 {
    u64::from(dword(bytes, at)) | u64::from(dword(bytes, at + 4)) << 32
}

impl BuildDesc {
    /// The desc in `bytes` (at least [`BUILD_DESC_BYTES`]), as written:
    /// nothing is judged ([`BuildDesc::plausible`]). None if a field that
    /// only has two values holds another.
    pub fn parse(bytes: &[u8]) -> Option<BuildDesc> {
        if bytes.len() < BUILD_DESC_BYTES {
            return None;
        }
        Some(BuildDesc {
            dest: qword(bytes, 0),
            inputs: Inputs::parse(&bytes[8..])?,
            source: qword(bytes, 32),
            scratch: qword(bytes, 40),
        })
    }

    /// Whether it reads as a desc the title's builder takes: structures
    /// below 2^40 and 256-byte aligned (the builder refuses others), no
    /// unknown build flag, descs where there are any, a source exactly
    /// when updating in place of building.
    pub fn plausible(&self) -> bool {
        let address = |a: u64| a != 0 && a < 1 << 40 && a.is_multiple_of(256);
        address(self.dest)
            && self.inputs.flags < 0x100
            && self.inputs.count < 1 << 24
            && (self.inputs.count == 0 || self.inputs.descs != 0)
            && (self.source == 0 || address(self.source))
            && (self.inputs.flags & PERFORM_UPDATE == 0 || self.source != 0)
    }
}

impl Inputs {
    /// The inputs in `bytes` (at least [`INPUTS_BYTES`]). None if a field
    /// that only has two values holds another.
    pub fn parse(bytes: &[u8]) -> Option<Inputs> {
        if bytes.len() < INPUTS_BYTES {
            return None;
        }
        let (kind, layout) = (dword(bytes, 0), dword(bytes, 12));
        if kind > 1 || layout > 1 {
            return None;
        }
        Some(Inputs {
            top: kind == 0,
            flags: dword(bytes, 4),
            count: dword(bytes, 8),
            pointers: layout == 1,
            descs: qword(bytes, 16),
        })
    }

    /// What a structure's sizes follow from: the kind, the flags, and of
    /// each geometry its kind, flags, formats and counts. Not where the
    /// data lies: the title is told sizes before it has placed anything.
    pub fn key(&self, geometries: &[Geometry]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.top, self.flags, self.count).hash(&mut h);
        for g in geometries {
            match *g {
                Geometry::Triangles { flags, index_format, vertex_format, indices, vertices, .. } => {
                    (0u8, flags, index_format, vertex_format, indices, vertices).hash(&mut h);
                }
                Geometry::Boxes { flags, count, .. } => (1u8, flags, count).hash(&mut h),
            }
        }
        h.finish()
    }
}

/// One element of the batched build's array (`D3D12XBOX_BUILD_RAYTRACING_
/// ACCELERATION_STRUCTURE_BATCHED_DESC2`), as a run's log showed it: the
/// standard desc, then at 48 the number of postbuild info descs and at 56
/// their address; 16 bytes more that were zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Element {
    pub desc: BuildDesc,
    pub postbuilds: u32,
    /// `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC`s
    /// ([`POSTBUILD_DESC_BYTES`] each): where the answer goes (a GPU
    /// address, title memory), and in a dword what is asked.
    pub postbuild_descs: u64,
}

pub const ELEMENT_BYTES: usize = 80;
pub const POSTBUILD_DESC_BYTES: usize = 16;
/// `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE`:
/// the answer is one qword, the size in bytes.
pub const COMPACTED_SIZE: u32 = 0;

impl Element {
    pub fn parse(bytes: &[u8]) -> Option<Element> {
        if bytes.len() < ELEMENT_BYTES {
            return None;
        }
        Some(Element { desc: BuildDesc::parse(bytes)?, postbuilds: dword(bytes, 48), postbuild_descs: qword(bytes, 56) })
    }
}

/// `D3D12_RAYTRACING_GEOMETRY_DESC` (56 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Geometry {
    Triangles {
        /// `D3D12_RAYTRACING_GEOMETRY_FLAGS` (1: opaque).
        flags: u32,
        /// A 3×4 matrix's address, or 0.
        transform: u64,
        /// `DXGI_FORMAT`s (index 0: no indices).
        index_format: u32,
        vertex_format: u32,
        indices: u32,
        vertices: u32,
        index_buffer: u64,
        vertex_buffer: u64,
        vertex_stride: u64,
    },
    /// Procedural primitives: boxes at an address.
    Boxes { flags: u32, count: u64, boxes: u64, stride: u64 },
}

impl Geometry {
    pub fn parse(bytes: &[u8]) -> Option<Geometry> {
        if bytes.len() < GEOMETRY_DESC_BYTES {
            return None;
        }
        let flags = dword(bytes, 4);
        match dword(bytes, 0) {
            0 => Some(Geometry::Triangles {
                flags,
                transform: qword(bytes, 8),
                index_format: dword(bytes, 16),
                vertex_format: dword(bytes, 20),
                indices: dword(bytes, 24),
                vertices: dword(bytes, 28),
                index_buffer: qword(bytes, 32),
                vertex_buffer: qword(bytes, 40),
                vertex_stride: qword(bytes, 48),
            }),
            1 => Some(Geometry::Boxes {
                flags,
                count: qword(bytes, 8),
                boxes: qword(bytes, 16),
                stride: qword(bytes, 24),
            }),
            _ => None,
        }
    }

    /// Triangles it holds.
    pub fn triangles(&self) -> u64 {
        match *self {
            Geometry::Triangles { index_format: 0, vertices, .. } => u64::from(vertices) / 3,
            Geometry::Triangles { indices, .. } => u64::from(indices) / 3,
            Geometry::Boxes { .. } => 0,
        }
    }
}

/// Where standard build descs lie in the batched build's array, found by
/// [`find_descs`]: the desc's offset in an element, and the element's size
/// (0 when a single element gave no second desc to measure it by).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub offset: usize,
    pub stride: usize,
}

/// Looks for the standard build desc in `bytes`, the first bytes of the
/// batched build's array of `count` elements: the first offset (by 8) at
/// which a [`BuildDesc::plausible`] desc lies, and with a second element
/// the first size (by 8) that puts another one at the same offset in it.
pub fn find_descs(bytes: &[u8], count: u32) -> Option<Layout> {
    let at = |o: usize| bytes.get(o..o + BUILD_DESC_BYTES).and_then(BuildDesc::parse).filter(BuildDesc::plausible);
    let offset = (0..bytes.len()).step_by(8).find(|&o| at(o).is_some())?;
    if count < 2 {
        return Some(Layout { offset, stride: 0 });
    }
    let stride = (offset + BUILD_DESC_BYTES..bytes.len()).step_by(8).map(|o| o - offset).find(|&s| at(offset + s).is_some())?;
    Some(Layout { offset, stride })
}

/// Counters since the last frame line, in the order of [`NAMES`].
static COUNTS: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
const NAMES: [&str; 16] = [
    "batched build calls",
    "batched elements",
    "top builds",
    "bottom builds",
    "updates",
    "builds allowing compaction",
    "geometries",
    "triangles",
    "rehydrate calls",
    "rehydrated structures",
    "copies",
    "descs not understood",
    "prebuild calls",
    "sizes answered",
    "sizes not answered",
    "late size stores kept off",
];
const CALLS: usize = 0;
const ELEMENTS: usize = 1;
const TOP: usize = 2;
const BOTTOM: usize = 3;
const UPDATES: usize = 4;
const COMPACTING: usize = 5;
const GEOMETRIES: usize = 6;
const TRIANGLES: usize = 7;
const REHYDRATES: usize = 8;
const REHYDRATED: usize = 9;
const COPIES: usize = 10;
const UNKNOWN: usize = 11;
const PREBUILDS: usize = 12;
const ANSWERED: usize = 13;
const UNANSWERED: usize = 14;
const KEPT_OFF: usize = 15;

fn count(what: usize, n: u64) {
    COUNTS[what].fetch_add(n, Ordering::Relaxed);
}

/// The counters' line: what the title asked of its builder over the last
/// `frames` frames (from the 10 s line of title memory by address).
pub(crate) fn log_frame(frames: u64) {
    if mode() == Mode::Off && !answers() {
        return;
    }
    let counts = COUNTS.each_ref().map(|c| c.swap(0, Ordering::Relaxed));
    if counts.iter().all(|&c| c == 0) {
        return;
    }
    let text: Vec<String> = NAMES.iter().zip(counts).filter(|&(_, c)| c != 0).map(|(name, c)| format!("{c} {name}")).collect();
    tracing::info!("dxr builder over {frames} frames: {}", text.join(", "));
}

/// `len` bytes of title memory at `at` the title's CPU passed us, if they
/// are committed (a pointer in a desc may be stale or ours to distrust).
fn title_bytes(at: u64, len: usize) -> Option<Vec<u8>> {
    if at == 0 || len == 0 || !crate::texture::committed(at as *const u8, len) {
        return None;
    }
    // SAFETY: committed memory of this process, read while the title's
    // calling thread waits in the wrapper.
    Some(unsafe { std::slice::from_raw_parts(at as *const u8, len) }.to_vec())
}

/// The first [`VERBOSE`] descs of each kind (bottom, top, copy, prebuild,
/// size store kept off, size answered) are logged whole.
const VERBOSE: u64 = 16;
static LOGGED: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
/// After the first [`VERBOSE`] batched calls, one in this many is dumped.
const RAW_EVERY: u64 = 1024;
static BATCHED_SEEN: AtomicU64 = AtomicU64::new(0);

/// `bytes` in hex, 8 bytes a group, its offset before every 64.
fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2 + bytes.len() / 4);
    for (k, chunk) in bytes.chunks(8).enumerate() {
        if k % 8 == 0 {
            let _ = write!(out, "{}+{:03x}:", if k == 0 { "" } else { " | " }, k * 8);
        }
        out.push(' ');
        for b in chunk {
            let _ = write!(out, "{b:02x}");
        }
    }
    out
}

/// How many bytes of each pointed-to block a raw dump shows.
const POINTED_BYTES: usize = 128;

/// The batched build's raw bytes, for its layout to be decoded from a log:
/// the element count, how much of the array could be read, the array's
/// first bytes, and for each of its first 16 qwords that points to readable
/// memory, that memory's first [`POINTED_BYTES`].
fn dump_batched(n: u32, descs: u64, bytes: &[u8], layout: Option<Layout>) {
    tracing::info!(
        n,
        descs = format_args!("{descs:#x}"),
        readable = bytes.len(),
        ?layout,
        "dxr builder batched raw: {}",
        hex(bytes)
    );
    for k in 0..bytes.len().min(128) / 8 {
        let at = qword(bytes, 8 * k);
        if at == descs || at < 0x1_0000 {
            continue;
        }
        if let Some(pointed) = title_bytes(at, POINTED_BYTES) {
            tracing::info!(
                field = 8 * k,
                at = format_args!("{at:#x}"),
                "dxr builder batched raw, pointed to: {}",
                hex(&pointed)
            );
        }
    }
}

fn verbose(kind: usize) -> bool {
    LOGGED[kind].fetch_add(1, Ordering::Relaxed) < VERBOSE
}

/// Counts one build desc, and its geometry descs (title CPU memory) for a
/// bottom-level one.
fn note_build(desc: &BuildDesc, how: &str) {
    count(if desc.inputs.top { TOP } else { BOTTOM }, 1);
    if desc.inputs.flags & PERFORM_UPDATE != 0 {
        count(UPDATES, 1);
    }
    if desc.inputs.flags & ALLOW_COMPACTION != 0 {
        count(COMPACTING, 1);
    }
    let mut formats: Vec<(u32, u32, u64)> = Vec::new();
    if !desc.inputs.top {
        let n = desc.inputs.count as usize;
        for k in 0..n.min(4096) {
            let at = if desc.inputs.pointers {
                title_bytes(desc.inputs.descs + 8 * k as u64, 8).map_or(0, |p| qword(&p, 0))
            } else {
                desc.inputs.descs + (GEOMETRY_DESC_BYTES * k) as u64
            };
            match title_bytes(at, GEOMETRY_DESC_BYTES).as_deref().and_then(Geometry::parse) {
                Some(g) => {
                    count(GEOMETRIES, 1);
                    count(TRIANGLES, g.triangles());
                    if let Geometry::Triangles { index_format, vertex_format, vertex_stride, .. } = g
                        && !formats.contains(&(vertex_format, index_format, vertex_stride))
                    {
                        formats.push((vertex_format, index_format, vertex_stride));
                    }
                }
                None => count(UNKNOWN, 1),
            }
        }
    }
    if verbose(usize::from(desc.inputs.top)) {
        tracing::info!(
            how,
            dest = format_args!("{:#x}", desc.dest),
            top = desc.inputs.top,
            flags = format_args!("{:#x}", desc.inputs.flags),
            count = desc.inputs.count,
            pointers = desc.inputs.pointers,
            descs = format_args!("{:#x}", desc.inputs.descs),
            source = format_args!("{:#x}", desc.source),
            scratch = format_args!("{:#x}", desc.scratch),
            formats = ?formats,
            "dxr builder desc (vertex format, index format, vertex stride)"
        );
    }
}

/// The real methods, where the title's imports pointed ([`hook`]).
static REAL: [AtomicUsize; 4] = [const { AtomicUsize::new(0) }; 4];
const BATCHED: usize = 0;
const COPY: usize = 1;
const REHYDRATE: usize = 2;
const PREBUILD: usize = 3;

/// The geometry descs of bottom-level `inputs` (title CPU memory), all of
/// them or none; none for a top level.
fn geometries(inputs: &Inputs) -> Option<Vec<Geometry>> {
    let n = inputs.count as usize;
    if inputs.top {
        return Some(Vec::new());
    }
    if n > 4096 {
        return None;
    }
    if inputs.pointers {
        let pointers = title_bytes(inputs.descs, 8 * n)?;
        return (0..n).map(|k| Geometry::parse(&title_bytes(qword(&pointers, 8 * k), GEOMETRY_DESC_BYTES)?)).collect();
    }
    title_bytes(inputs.descs, GEOMETRY_DESC_BYTES * n)?.chunks_exact(GEOMETRY_DESC_BYTES).map(Geometry::parse).collect()
}

/// What the builder told the title a structure may take at most
/// (`ResultDataMaxSizeInBytes`), by [`Inputs::key`]: the last [`TOLD`]
/// keys.
#[derive(Default)]
struct Told {
    sizes: HashMap<u64, u64>,
    order: VecDeque<u64>,
}

const TOLD: usize = 4096;

impl Told {
    fn put(&mut self, key: u64, size: u64) {
        if self.sizes.insert(key, size).is_some() {
            return;
        }
        self.order.push_back(key);
        if self.order.len() > TOLD
            && let Some(oldest) = self.order.pop_front()
        {
            self.sizes.remove(&oldest);
        }
    }
}

static TOLD_SIZES: std::sync::LazyLock<Mutex<Told>> = std::sync::LazyLock::new(Mutex::default);
/// The title's prebuild calls without a stream desc, and with one.
static STREAMS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

type Prebuild = extern "system" fn(*mut c_void, *const c_void, *const c_void, *mut u64);

/// `XboxBvhBuilder::GetRaytracingAccelerationStructurePrebuildInfo(inputs,
/// stream, info)`: the sizes of a structure and of its scratch memory, on
/// the CPU. What it said of the structure is kept by the inputs' key.
extern "system" fn prebuild(this: *mut c_void, inputs: *const c_void, stream: *const c_void, info: *mut u64) {
    let real = REAL[PREBUILD].load(Ordering::Relaxed);
    // SAFETY: the method the title imported, with the title's arguments.
    let real: Prebuild = unsafe { std::mem::transmute(real) };
    real(this, inputs, stream, info);
    count(PREBUILDS, 1);
    STREAMS[usize::from(!stream.is_null())].fetch_add(1, Ordering::Relaxed);
    let read = title_bytes(inputs as u64, INPUTS_BYTES).as_deref().and_then(Inputs::parse);
    let said = title_bytes(info as u64, 24).map(|b| [0, 8, 16].map(|at| qword(&b, at)));
    let key = read.as_ref().and_then(|i| Some(i.key(&geometries(i)?)));
    if let (Some(key), Some([size, ..])) = (key, said) {
        TOLD_SIZES.lock().unwrap_or_else(|e| e.into_inner()).put(key, size);
    }
    if verbose(3) {
        tracing::info!(
            inputs = ?read,
            stream = format_args!("{:#x}", stream as usize),
            stream_bytes = %title_bytes(stream as u64, 16).as_deref().map(hex).unwrap_or_default(),
            sizes = ?said,
            key = ?key,
            "dxr builder prebuild (result, scratch, update scratch)"
        );
    }
}

/// The most a structure of `inputs` (which lie at `at`, in a batched
/// build's element) takes, as the builder itself says, and where that is
/// from: what it told the title for the same inputs, else asked now, in
/// the form the title asks (without a stream desc, and only once the title
/// was seen to ask so and never otherwise).
fn max_size(this: *mut c_void, inputs: &Inputs, at: u64) -> Option<(u64, &'static str)> {
    let told = geometries(inputs).and_then(|g| TOLD_SIZES.lock().unwrap_or_else(|e| e.into_inner()).sizes.get(&inputs.key(&g)).copied());
    let size = match told {
        Some(size) => (size, "told to the title"),
        None => {
            let real = REAL[PREBUILD].load(Ordering::Relaxed);
            if real == 0 || STREAMS[0].load(Ordering::Relaxed) == 0 || STREAMS[1].load(Ordering::Relaxed) != 0 {
                return None;
            }
            let mut info = [0u64; 3];
            // SAFETY: the builder's method as the title calls it, with the
            // builder the title is calling now and the inputs it passes.
            let real: Prebuild = unsafe { std::mem::transmute(real) };
            real(this, at as *const c_void, std::ptr::null(), info.as_mut_ptr());
            (info[0], "asked now")
        }
    };
    (size.0 != 0 && size.0 < 1 << 32).then_some(size)
}

/// The slots (title addresses of a qword) that hold an answer of ours, and
/// the span they lie in, for [`keep_answers`] to look no further in most
/// calls.
static SLOTS: Mutex<BTreeSet<u64>> = Mutex::new(BTreeSet::new());
static SPAN: [AtomicU64; 2] = [AtomicU64::new(u64::MAX), AtomicU64::new(0)];
/// The most slots held: the title's are a ring in a small buffer.
const SLOTS_MOST: usize = 1 << 16;

/// `slot` holds an answer of ours from now on; false if no more are held.
fn hold_slot(slot: u64) -> bool {
    let mut slots = SLOTS.lock().unwrap_or_else(|e| e.into_inner());
    if slots.len() >= SLOTS_MOST && !slots.contains(&slot) {
        return false;
    }
    slots.insert(slot);
    SPAN[0].fetch_min(slot, Ordering::Relaxed);
    SPAN[1].fetch_max(slot + 8, Ordering::Relaxed);
    true
}

/// `slot` is the builder's to store into again.
fn free_slot(slot: u64) {
    if slot < SPAN[1].load(Ordering::Relaxed) {
        SLOTS.lock().unwrap_or_else(|e| e.into_inner()).remove(&slot);
    }
}

/// Before what shaders wrote goes into title memory at `at` (`title`: what
/// is there, `merged`: the same with their bytes put in): the bytes of
/// slots that hold an answer of ours stay as they are. The builder's store
/// of a compacted size lands frames after the title asked, by when the
/// slot may be another structure's. None if `merged` is right as it is.
pub(crate) fn keep_answers(at: usize, title: &[u8], merged: &[u8]) -> Option<Vec<u8>> {
    let (from, to) = (at as u64, (at + merged.len().min(title.len())) as u64);
    if from >= SPAN[1].load(Ordering::Relaxed) || to <= SPAN[0].load(Ordering::Relaxed) {
        return None;
    }
    let slots = SLOTS.lock().unwrap_or_else(|e| e.into_inner());
    let mut kept: Option<Vec<u8>> = None;
    for &slot in slots.range(from.saturating_sub(7)..to) {
        let (a, b) = ((slot.max(from) - from) as usize, ((slot + 8).min(to) - from) as usize);
        if merged[a..b] == title[a..b] {
            continue;
        }
        count(KEPT_OFF, 1);
        if b - a == 8 && verbose(4) {
            tracing::info!(
                slot = format_args!("{slot:#x}"),
                answer = format_args!("{:#x}", qword(title, a)),
                store = format_args!("{:#x}", qword(merged, a)),
                "dxr: the builder's late size store is kept off an answered slot"
            );
        }
        kept.get_or_insert_with(|| merged.to_vec())[a..b].copy_from_slice(&title[a..b]);
    }
    kept
}

/// Writes the answers of a batched build's compacted-size postbuild descs
/// (see the module) and returns them (slot, size). A slot that gets no
/// answer of ours (another kind of info, or no size known) is the
/// builder's again.
fn answer_sizes(this: *mut c_void, n: u32, descs: u64) -> Vec<(u64, u64)> {
    let mut answered = Vec::new();
    let Some(bytes) = title_bytes(descs, ELEMENT_BYTES * (n as usize).min(4096)) else {
        return answered;
    };
    for (k, element) in bytes.chunks_exact(ELEMENT_BYTES).enumerate() {
        let Some(element) = Element::parse(element).filter(|e| e.desc.plausible() && (1..=8).contains(&e.postbuilds)) else {
            continue;
        };
        let Some(posts) = title_bytes(element.postbuild_descs, POSTBUILD_DESC_BYTES * element.postbuilds as usize) else {
            continue;
        };
        for post in posts.chunks_exact(POSTBUILD_DESC_BYTES) {
            let (slot, kind) = (qword(post, 0), dword(post, 8));
            let size = (kind == COMPACTED_SIZE && title_bytes(slot, 8).is_some())
                .then(|| max_size(this, &element.desc.inputs, descs + (k * ELEMENT_BYTES + 8) as u64))
                .flatten()
                .filter(|_| hold_slot(slot));
            let Some((size, from)) = size else {
                free_slot(slot);
                if kind == COMPACTED_SIZE {
                    count(UNANSWERED, 1);
                }
                continue;
            };
            put_answer(slot, size);
            count(ANSWERED, 1);
            if verbose(5) {
                tracing::info!(
                    structure = format_args!("{:#x}", element.desc.dest),
                    slot = format_args!("{slot:#x}"),
                    size = format_args!("{size:#x}"),
                    from,
                    "dxr: compacted size answered at the call with the builder's most"
                );
            }
            answered.push((slot, size));
        }
    }
    answered
}

fn put_answer(slot: u64, size: u64) {
    crate::host::write_mapped(slot as *mut u8, size.to_le_bytes().as_ptr(), 8);
}

/// The most bytes read of a batched build's array to find its layout.
const PROBE_BYTES: usize = 512;

/// `XboxBvhBuilder::BuildRaytracingAccelerationStructureBatched2X(list,
/// count, descs)`: every build and refit of the title's.
extern "system" fn batched(this: *mut c_void, list: *mut c_void, n: u32, descs: *const c_void) {
    if mode() != Mode::Off {
        watch_batched(n, descs);
    }
    let answered = if answers() { answer_sizes(this, n, descs as u64) } else { Vec::new() };
    let real = REAL[BATCHED].load(Ordering::Relaxed);
    // SAFETY: the method the title imported, with the title's arguments.
    let real: extern "system" fn(*mut c_void, *mut c_void, u32, *const c_void) = unsafe { std::mem::transmute(real) };
    real(this, list, n, descs);
    // Once more: whatever the builder's own code put there while it
    // recorded, the title reads the answer.
    for (slot, size) in answered {
        if title_bytes(slot, 8).is_some_and(|b| qword(&b, 0) != size) {
            put_answer(slot, size);
        }
    }
}

/// What a batched build asks, counted and logged.
fn watch_batched(n: u32, descs: *const c_void) {
    count(CALLS, 1);
    count(ELEMENTS, u64::from(n));
    // The layout is the console's own: looked for in the first bytes, at
    // every call (cheap), and logged for the first ones.
    let probe = (1..=PROBE_BYTES / 64).rev().find_map(|k| title_bytes(descs as u64, 64 * k));
    let found = probe.as_deref().and_then(|bytes| find_descs(bytes, n));
    let seen = BATCHED_SEEN.fetch_add(1, Ordering::Relaxed);
    if seen < VERBOSE || seen.is_multiple_of(RAW_EVERY) {
        dump_batched(n, descs as u64, probe.as_deref().unwrap_or_default(), found);
    }
    match probe.as_deref().zip(found) {
        Some((bytes, layout)) => {
            let stride = if layout.stride == 0 { bytes.len() } else { layout.stride };
            for k in 0..n as usize {
                let at = descs as u64 + (k * stride + layout.offset) as u64;
                match title_bytes(at, BUILD_DESC_BYTES).as_deref().and_then(BuildDesc::parse) {
                    Some(desc) if desc.plausible() => note_build(&desc, "batched"),
                    _ => count(UNKNOWN, 1),
                }
                if layout.stride == 0 {
                    break;
                }
            }
        }
        None => count(UNKNOWN, u64::from(n)),
    }
}

/// `XboxBvhBuilder::CopyRaytracingAccelerationStructure(list, dest,
/// source, mode)`: clones and compactions.
extern "system" fn copy(this: *mut c_void, list: *mut c_void, dest: u64, source: u64, how: u32) {
    count(COPIES, 1);
    if verbose(2) {
        tracing::info!(dest = format_args!("{dest:#x}"), source = format_args!("{source:#x}"), mode = how, "dxr builder copy");
    }
    let real = REAL[COPY].load(Ordering::Relaxed);
    // SAFETY: the method the title imported, with the title's arguments.
    let real: extern "system" fn(*mut c_void, *mut c_void, u64, u64, u32) = unsafe { std::mem::transmute(real) };
    real(this, list, dest, source, how);
}

/// `XboxBvhBuilder::RehydrateRaytracingAccelerationStructureBatched2X(
/// list, count, descs, values)`: structures built offline, placed in
/// memory. Its descs are the standard ones.
extern "system" fn rehydrate(this: *mut c_void, list: *mut c_void, n: u32, descs: *const c_void, values: *const u64) -> i32 {
    count(REHYDRATES, 1);
    for k in 0..n as usize {
        let at = descs as u64 + (k * BUILD_DESC_BYTES) as u64;
        match title_bytes(at, BUILD_DESC_BYTES).as_deref().and_then(BuildDesc::parse) {
            Some(desc) => {
                count(REHYDRATED, 1);
                note_build(&desc, "rehydrate");
            }
            None => count(UNKNOWN, 1),
        }
    }
    let real = REAL[REHYDRATE].load(Ordering::Relaxed);
    // SAFETY: the method the title imported, with the title's arguments.
    let real: extern "system" fn(*mut c_void, *mut c_void, u32, *const c_void, *const u64) -> i32 = unsafe { std::mem::transmute(real) };
    real(this, list, n, descs, values)
}

/// Which wrapper stands for the builder's export `symbol` (its decorated
/// name), if one does.
fn wrapper_of(symbol: &str) -> Option<(usize, usize)> {
    let is = |method: &str| symbol.starts_with(&format!("?{method}@XboxBvhBuilder@@"));
    if is("BuildRaytracingAccelerationStructureBatched2X") {
        Some((BATCHED, batched as *const () as usize))
    } else if is("CopyRaytracingAccelerationStructure") {
        Some((COPY, copy as *const () as usize))
    } else if is("RehydrateRaytracingAccelerationStructureBatched2X") {
        Some((REHYDRATE, rehydrate as *const () as usize))
    } else if is("GetRaytracingAccelerationStructurePrebuildInfo") {
        Some((PREBUILD, prebuild as *const () as usize))
    } else {
        None
    }
}

/// The title imports `symbol` from its acceleration-structure builder,
/// which exports it at `real`: with DXR on ([`mode`]), the address to
/// bind instead, a wrapper that reads the call and passes it on. None
/// (bind the builder's own) for other symbols and with DXR off; the two
/// methods sizes are answered from ([`answers`]) are wrapped without it.
pub fn hook(symbol: &str, real: usize) -> Option<usize> {
    let (which, wrapper) = wrapper_of(symbol)?;
    if mode() == Mode::Off && !(answers() && matches!(which, BATCHED | PREBUILD)) {
        return None;
    }
    REAL[which].store(real, Ordering::Relaxed);
    tracing::info!(symbol, real = format_args!("{real:#x}"), "dxr: the builder's method is watched");
    Some(wrapper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_is_off_unless_asked() {
        assert_eq!(parse_mode(None), Mode::Off);
        assert_eq!(parse_mode(Some("0")), Mode::Off);
        assert_eq!(parse_mode(Some("yes")), Mode::Off);
        assert_eq!(parse_mode(Some("1")), Mode::On);
        assert_eq!(parse_mode(Some(" check ")), Mode::Check);
        assert_eq!(parse_mode(Some("CHECK")), Mode::Check);
    }

    /// A build desc as D3D12 lays it out: destination, type, flags, count,
    /// layout, descs, source, scratch.
    #[allow(clippy::too_many_arguments)]
    fn desc_bytes(dest: u64, kind: u32, flags: u32, n: u32, layout: u32, descs: u64, source: u64, scratch: u64) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend(dest.to_le_bytes());
        for x in [kind, flags, n, layout] {
            b.extend(x.to_le_bytes());
        }
        for x in [descs, source, scratch] {
            b.extend(x.to_le_bytes());
        }
        assert_eq!(b.len(), BUILD_DESC_BYTES);
        b
    }

    #[test]
    fn a_build_desc_is_read_field_by_field() {
        let b = desc_bytes(0xd_5f1c_0000, 1, 0x26, 3, 1, 0x1234_5678, 0xd_5f00_0000, 0xd_6000_0000);
        let d = BuildDesc::parse(&b).unwrap();
        assert_eq!(d.dest, 0xd_5f1c_0000);
        assert!(!d.inputs.top);
        assert_eq!((d.inputs.flags, d.inputs.count, d.inputs.pointers), (0x26, 3, true));
        assert_eq!((d.inputs.descs, d.source, d.scratch), (0x1234_5678, 0xd_5f00_0000, 0xd_6000_0000));
        assert!(d.plausible());
        // A top-level build of 2,000 instances, no source.
        let top = BuildDesc::parse(&desc_bytes(0xd_6000_0100, 0, 4, 2000, 0, 0x7_0000_1000, 0, 0xd_6100_0000)).unwrap();
        assert!(top.inputs.top && top.plausible());
        // Too short, or a type that does not exist.
        assert_eq!(BuildDesc::parse(&b[..40]), None);
        assert_eq!(BuildDesc::parse(&desc_bytes(0x100, 2, 0, 0, 0, 0, 0, 0)), None);
    }

    #[test]
    fn descs_the_builder_would_refuse_are_not_plausible() {
        let ok = |b: Vec<u8>| BuildDesc::parse(&b).is_some_and(|d| d.plausible());
        assert!(ok(desc_bytes(0x1_0000_0100, 1, 0, 1, 0, 0x1000, 0, 0x2_0000_0000)));
        assert!(!ok(desc_bytes(0x1_0000_0101, 1, 0, 1, 0, 0x1000, 0, 0)), "not 256-byte aligned");
        assert!(!ok(desc_bytes(1 << 40, 1, 0, 1, 0, 0x1000, 0, 0)), "above 2^40");
        assert!(!ok(desc_bytes(0, 1, 0, 1, 0, 0x1000, 0, 0)), "no destination");
        assert!(!ok(desc_bytes(0x1_0000_0100, 1, 0, 1, 0, 0, 0, 0)), "geometry with no descs");
        assert!(!ok(desc_bytes(0x1_0000_0100, 1, PERFORM_UPDATE, 1, 0, 0x1000, 0, 0)), "an update with no source");
        assert!(!ok(desc_bytes(0x1_0000_0100, 1, 0x1_0000, 1, 0, 0x1000, 0, 0)), "an unknown flag");
    }

    #[test]
    fn geometry_descs_are_read_by_type() {
        let mut b = vec![0u8; GEOMETRY_DESC_BYTES];
        let put = |b: &mut [u8], at: usize, x: u64, n: usize| b[at..at + n].copy_from_slice(&x.to_le_bytes()[..n]);
        // Triangles, opaque: R16_UINT indices (57), R32G32B32_FLOAT vertices (6).
        for (at, x, n) in [(0, 0, 4), (4, 1, 4), (8, 0, 8), (16, 57, 4), (20, 6, 4), (24, 300, 4), (28, 120, 4), (32, 0x7000, 8), (40, 0x8000, 8), (48, 12, 8)] {
            put(&mut b, at, x, n);
        }
        let g = Geometry::parse(&b).unwrap();
        assert_eq!(
            g,
            Geometry::Triangles {
                flags: 1,
                transform: 0,
                index_format: 57,
                vertex_format: 6,
                indices: 300,
                vertices: 120,
                index_buffer: 0x7000,
                vertex_buffer: 0x8000,
                vertex_stride: 12
            }
        );
        assert_eq!(g.triangles(), 100);
        // No indices: three vertices a triangle.
        put(&mut b, 16, 0, 4);
        assert_eq!(Geometry::parse(&b).unwrap().triangles(), 40);
        // Boxes.
        let mut b = vec![0u8; GEOMETRY_DESC_BYTES];
        for (at, x, n) in [(0, 1, 4), (8, 9, 8), (16, 0x9000, 8), (24, 24, 8)] {
            put(&mut b, at, x, n);
        }
        assert_eq!(Geometry::parse(&b), Some(Geometry::Boxes { flags: 0, count: 9, boxes: 0x9000, stride: 24 }));
        put(&mut b, 0, 2, 4);
        assert_eq!(Geometry::parse(&b), None);
    }

    /// The standard desc is found inside a larger element of unknown
    /// layout, and the element's size from the second one.
    #[test]
    fn the_standard_desc_is_found_in_a_batched_element() {
        // An element: 16 bytes of something, the desc, 24 bytes more.
        let element = |dest: u64| {
            let mut e = vec![0xEEu8; 16];
            e.extend(desc_bytes(dest, 1, 4, 2, 0, 0x7_0000_2000, 0, 0xd_7000_0000));
            e.extend([0x11u8; 24]);
            e
        };
        let mut bytes = element(0xd_5f1c_0000);
        assert_eq!(find_descs(&bytes, 1), Some(Layout { offset: 16, stride: 0 }));
        // Two elements asked for, one readable: the size cannot be told.
        assert_eq!(find_descs(&bytes, 2), None);
        bytes.extend(element(0xd_5f2c_0000));
        assert_eq!(find_descs(&bytes, 2), Some(Layout { offset: 16, stride: 88 }));
        // The plain array of standard descs.
        let mut plain = desc_bytes(0xd_5f1c_0000, 0, 0, 10, 0, 0x7_0000_3000, 0, 0xd_7000_0000);
        plain.extend(desc_bytes(0xd_5f2c_0000, 0, 0, 10, 0, 0x7_0000_4000, 0, 0xd_7000_0000));
        assert_eq!(find_descs(&plain, 2), Some(Layout { offset: 0, stride: 48 }));
        assert_eq!(find_descs(&[0u8; 256], 1), None);
    }

    #[test]
    fn raw_bytes_are_dumped_in_groups_of_eight() {
        let bytes: Vec<u8> = (0u8..72).collect();
        let h = hex(&bytes);
        assert!(h.starts_with("+000: 0001020304050607 08090a0b0c0d0e0f"), "{h}");
        assert!(h.ends_with(" | +040: 4041424344454647"), "{h}");
        assert_eq!(hex(&[]), "");
    }

    #[test]
    fn sizes_are_answered_unless_switched_off() {
        assert!(parse_answers(None));
        assert!(parse_answers(Some("1")));
        assert!(!parse_answers(Some(" 0 ")));
    }

    /// An element as a run logged it: the desc, one postbuild desc, zeros.
    fn element_bytes(dest: u64, flags: u32, n: u32, descs: u64, postbuilds: u32, postbuild_descs: u64) -> Vec<u8> {
        let mut e = desc_bytes(dest, 1, flags, n, 0, descs, 0, 0x1d_c000_0000);
        e.extend(u64::from(postbuilds).to_le_bytes());
        e.extend(postbuild_descs.to_le_bytes());
        e.extend([0u8; 16]);
        assert_eq!(e.len(), ELEMENT_BYTES);
        e
    }

    #[test]
    fn a_batched_element_holds_its_postbuild_descs() {
        let e = element_bytes(0xd_6000_8700, 0xa, 1, 0xd_1000_6ed0, 1, 0x5f_f000_5d50);
        let got = Element::parse(&e).unwrap();
        assert_eq!((got.desc.dest, got.postbuilds, got.postbuild_descs), (0xd_6000_8700, 1, 0x5f_f000_5d50));
        assert!(got.desc.plausible());
        assert_eq!(got.desc.inputs, Inputs::parse(&e[8..]).unwrap());
        assert_eq!(Element::parse(&e[..72]), None);
    }

    fn triangles(indices: u32, vertices: u32, buffer: u64) -> Geometry {
        Geometry::Triangles {
            flags: 1,
            transform: 0,
            index_format: 57,
            vertex_format: 6,
            indices,
            vertices,
            index_buffer: buffer,
            vertex_buffer: buffer + 0x100,
            vertex_stride: 12,
        }
    }

    #[test]
    fn sizes_follow_from_counts_not_from_addresses() {
        let inputs = |flags, descs| Inputs { top: false, flags, count: 1, pointers: false, descs };
        let key = inputs(0xa, 0x1000).key(&[triangles(300, 120, 0x7000)]);
        assert_eq!(key, inputs(0xa, 0x2000).key(&[triangles(300, 120, 0x9000)]));
        assert_ne!(key, inputs(0xa, 0x1000).key(&[triangles(303, 120, 0x7000)]));
        assert_ne!(key, inputs(0xa, 0x1000).key(&[triangles(300, 121, 0x7000)]));
        assert_ne!(key, inputs(0xb, 0x1000).key(&[triangles(300, 120, 0x7000)]));
        assert_ne!(key, inputs(0xa, 0x1000).key(&[Geometry::Boxes { flags: 1, count: 100, boxes: 0x7000, stride: 24 }]));
    }

    #[test]
    fn what_the_title_was_told_is_bounded() {
        let mut told = Told::default();
        for key in 0..TOLD as u64 + 10 {
            told.put(key, key + 1);
        }
        // Told again: the newer size, in the same place.
        told.put(TOLD as u64 + 9, 7);
        assert_eq!((told.sizes.len(), told.order.len()), (TOLD, TOLD));
        assert_eq!(told.sizes.get(&9), None);
        assert_eq!(told.sizes.get(&10), Some(&11));
        assert_eq!(told.sizes.get(&(TOLD as u64 + 9)), Some(&7));
    }

    #[test]
    fn a_late_store_stays_off_an_answered_slot() {
        let at = 0x7_1234_0000usize;
        let title = [0x11u8; 32];
        let mut merged = title;
        merged[4..20].fill(0x22);
        assert_eq!(keep_answers(at, &title, &merged), None);
        assert!(hold_slot(at as u64 + 8));
        // The slot's bytes stay; what is stored beside it lands.
        let kept = keep_answers(at, &title, &merged).unwrap();
        assert_eq!((&kept[4..8], &kept[8..16], &kept[16..20]), (&[0x22; 4][..], &[0x11; 8][..], &[0x22; 4][..]));
        // A stretch that begins inside the slot.
        let kept = keep_answers(at + 12, &title[12..], &merged[12..]).unwrap();
        assert_eq!((&kept[..4], &kept[4..8]), (&[0x11; 4][..], &[0x22; 4][..]));
        // Nothing stored there, and stretches beside it.
        assert_eq!(keep_answers(at, &title, &title), None);
        assert_eq!(keep_answers(at + 16, &title[16..], &merged[16..]), None);
        assert_eq!(keep_answers(at, &title[..8], &merged[..8]), None);
        // Given back to the builder.
        free_slot(at as u64 + 8);
        assert_eq!(keep_answers(at, &title, &merged), None);
    }

    /// The builder's prebuild method, for the test below: 0x1000 a geometry.
    extern "system" fn told(_: *mut c_void, inputs: *const c_void, _: *const c_void, info: *mut u64) {
        // SAFETY: the test's inputs and its three qwords.
        unsafe {
            let n = inputs.cast::<u32>().add(2).read_unaligned();
            info.write(u64::from(n) * 0x1000);
            info.add(1).write(0x40);
            info.add(2).write(0x20);
        }
    }

    /// From the title's prebuild call to its batched build, in memory of
    /// the test's: the slot of a compacted size holds the builder's most
    /// before the build goes on, whether the title was told that size or
    /// not; a slot asked something else is left alone.
    #[test]
    fn a_compacted_size_is_answered_with_what_the_builder_said() {
        fn geometry_bytes(n: usize, indices: u32) -> Vec<u8> {
            let mut b = vec![0u8; GEOMETRY_DESC_BYTES * n];
            for g in b.chunks_exact_mut(GEOMETRY_DESC_BYTES) {
                g[16..20].copy_from_slice(&57u32.to_le_bytes());
                g[20..24].copy_from_slice(&6u32.to_le_bytes());
                g[24..28].copy_from_slice(&indices.to_le_bytes());
                g[28..32].copy_from_slice(&120u32.to_le_bytes());
            }
            b
        }
        REAL[PREBUILD].store(told as *const () as usize, Ordering::Relaxed);
        let slots = Box::new([0xAAAA_u64; 4]);
        let slot = |k: usize| std::ptr::from_ref(&slots[k]) as u64;
        let post = |slot: u64, kind: u32| {
            let mut b = slot.to_le_bytes().to_vec();
            b.extend(u64::from(kind).to_le_bytes());
            b
        };
        let (told_of, new, third) = (geometry_bytes(2, 300), geometry_bytes(3, 600), geometry_bytes(1, 30));
        let posts = [post(slot(0), COMPACTED_SIZE), post(slot(1), COMPACTED_SIZE), [post(slot(2), COMPACTED_SIZE), post(slot(3), 3)].concat()];
        // The title asks the sizes of the first structure, in other memory
        // than it builds from later.
        let asked = geometry_bytes(2, 300);
        let inputs = element_bytes(0x100, 0xa, 2, asked.as_ptr() as u64, 0, 0);
        let mut info = [0u64; 3];
        prebuild(std::ptr::null_mut(), inputs[8..].as_ptr().cast(), std::ptr::null(), info.as_mut_ptr());
        assert_eq!(info, [0x2000, 0x40, 0x20]);
        // Nothing was said of the second and the third: the builder is asked.
        let mut elements = element_bytes(0xd_6000_0100, 0xa, 2, told_of.as_ptr() as u64, 1, posts[0].as_ptr() as u64);
        elements.extend(element_bytes(0xd_6000_4100, 0xa, 3, new.as_ptr() as u64, 1, posts[1].as_ptr() as u64));
        elements.extend(element_bytes(0xd_6000_8100, 0xa, 1, third.as_ptr() as u64, 2, posts[2].as_ptr() as u64));
        // A slot that held an older answer and is asked something else now
        // is given back.
        assert!(hold_slot(slot(3)));
        let answered = answer_sizes(std::ptr::null_mut(), 3, elements.as_ptr() as u64);
        assert_eq!(answered, [(slot(0), 0x2000), (slot(1), 0x3000), (slot(2), 0x1000)]);
        assert_eq!(*slots, [0x2000, 0x3000, 0x1000, 0xAAAA]);
        // The builder's stores, landing late: off the answered slots only.
        let title: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
        let kept = keep_answers(slot(0) as usize, &title, &[0x77; 32]).unwrap();
        assert_eq!((&kept[..24], &kept[24..]), (&title[..24], &[0x77; 8][..]));
    }

    #[test]
    fn only_the_four_watched_methods_have_wrappers() {
        let prebuild = "?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z";
        assert_eq!(wrapper_of(prebuild).map(|w| w.0), Some(PREBUILD));
        let batched = "?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z";
        assert_eq!(wrapper_of(batched).map(|w| w.0), Some(BATCHED));
        let copy = "?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE@@@Z";
        assert_eq!(wrapper_of(copy).map(|w| w.0), Some(COPY));
        let rehydrate = "?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z";
        assert_eq!(wrapper_of(rehydrate).map(|w| w.0), Some(REHYDRATE));
        // The plain build (not imported by the title) and the rest pass.
        assert_eq!(wrapper_of("?BuildRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@PEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@IPEBUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC@@@Z"), None);
        assert_eq!(wrapper_of("?Set64BitMortonCodes@XboxBvhBuilder@@QEAAX_N@Z"), None);
        assert_eq!(wrapper_of("??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z"), None);
    }
}
