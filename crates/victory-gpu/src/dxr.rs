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
//! with those of the memory its first qwords point to, for the next stage
//! to be written against what a run shows. Nothing is built on the host yet.

use std::ffi::c_void;
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
        let (kind, layout) = (dword(bytes, 8), dword(bytes, 20));
        if kind > 1 || layout > 1 {
            return None;
        }
        Some(BuildDesc {
            dest: qword(bytes, 0),
            inputs: Inputs {
                top: kind == 0,
                flags: dword(bytes, 12),
                count: dword(bytes, 16),
                pointers: layout == 1,
                descs: qword(bytes, 24),
            },
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
static COUNTS: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
const NAMES: [&str; 12] = [
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

fn count(what: usize, n: u64) {
    COUNTS[what].fetch_add(n, Ordering::Relaxed);
}

/// The counters' line: what the title asked of its builder over the last
/// `frames` frames (from the 10 s line of title memory by address).
pub(crate) fn log_frame(frames: u64) {
    if mode() == Mode::Off {
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

/// The first [`VERBOSE`] descs of each kind (bottom, top, copy) are
/// logged whole.
const VERBOSE: u64 = 16;
static LOGGED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
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
static REAL: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];
const BATCHED: usize = 0;
const COPY: usize = 1;
const REHYDRATE: usize = 2;

/// The most bytes read of a batched build's array to find its layout.
const PROBE_BYTES: usize = 512;

/// `XboxBvhBuilder::BuildRaytracingAccelerationStructureBatched2X(list,
/// count, descs)`: every build and refit of the title's.
extern "system" fn batched(this: *mut c_void, list: *mut c_void, n: u32, descs: *const c_void) {
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
    let real = REAL[BATCHED].load(Ordering::Relaxed);
    // SAFETY: the method the title imported, with the title's arguments.
    let real: extern "system" fn(*mut c_void, *mut c_void, u32, *const c_void) = unsafe { std::mem::transmute(real) };
    real(this, list, n, descs);
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
    } else {
        None
    }
}

/// The title imports `symbol` from its acceleration-structure builder,
/// which exports it at `real`: with DXR on ([`mode`]), the address to
/// bind instead, a wrapper that reads the call and passes it on. None
/// (bind the builder's own) for other symbols and with DXR off.
pub fn hook(symbol: &str, real: usize) -> Option<usize> {
    if mode() == Mode::Off {
        return None;
    }
    let (which, wrapper) = wrapper_of(symbol)?;
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
    fn only_the_three_watched_methods_have_wrappers() {
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
