//! Device methods 0x1c8 and 0x218: the console's pipeline state
//! deserializers (Marvisak/VIctory#3586, adapted).
//!
//! Compute state is the driver's FSPX payload without its container: the
//! payload is rebuilt here, wrapped in the ordinary shader container and
//! made through `create_compute`, so translation, cache and host pipeline
//! see the same bytes as any other compute shader.
//!
//! Graphics state carries fixed-function blobs whose format is not decoded.
//! Unlike upstream, which hands out a pipeline without shaders (every draw
//! with it dropped), the description is validated and counted, then the
//! call fails as before (`E_NOTIMPL`), so the title keeps whatever it does
//! without a deserialized pipeline.
//!
//! Descriptions are ABI input: every pair is range-checked before it is
//! copied, and a pair of size zero is absent even with a non-null pointer.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::object::{E_INVALIDARG, E_NOTIMPL, HRESULT, Iid, S_OK, This};

/// Guards the copy against a corrupt size; one pipeline's blobs are small.
const MAX_PAIR_SIZE: u64 = 16 << 20;

#[repr(C)]
#[derive(Clone, Copy)]
struct Pair {
    data: *const u8,
    size: u64,
}

/// Method 0x218's description: root signature, then the dispatch state
/// (kind 0), opaque driver state (1), machine code (6) and records (7).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ComputeDesc {
    root: This,
    kind0: Pair,
    kind1: Pair,
    kind6: Pair,
    kind7: Pair,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct StageDesc {
    code: Pair,
    records: Pair,
}

/// Method 0x1c8's description: root signature, two fixed-function state
/// blobs, seven stage blocks.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GraphicsDesc {
    root: This,
    state_a: Pair,
    state_b: Pair,
    stages: [StageDesc; 7],
}

/// `pipeline::ComputeHead`'s layout (D3D12's root signature then
/// bytecode), whose fields that module keeps private.
#[repr(C)]
struct ComputeHead {
    root: This,
    code: *const u8,
    size: usize,
}

const _: () = assert!(size_of::<ComputeDesc>() == 0x48);
const _: () = assert!(size_of::<GraphicsDesc>() == 0x108);
const _: () = assert!(std::mem::offset_of!(GraphicsDesc, stages) == 0x28);
const _: () = assert!(size_of::<ComputeHead>() == size_of::<crate::pipeline::ComputeHead>());

/// Compute pipelines made, compute descriptions refused, graphics
/// descriptions read (and left to the title), graphics descriptions refused.
static COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The totals line, logged once (at frame 1024).
pub(crate) fn log_totals() {
    let [made, refused, read, malformed] = COUNTS.each_ref().map(|c| c.load(Relaxed));
    tracing::info!(
        compute_made = made,
        compute_failed = refused,
        graphics_left_to_title = read,
        graphics_malformed = malformed,
        "pipeline state deserialization totals"
    );
}

fn copied(pair: Pair) -> Result<Option<Vec<u8>>, &'static str> {
    if pair.size == 0 {
        return Ok(None);
    }
    if pair.data.is_null() {
        return Err("null data with a size");
    }
    if pair.size > MAX_PAIR_SIZE {
        return Err("size over cap");
    }
    // SAFETY: a non-null pair names `size` readable bytes under the console
    // ABI; the size is capped above.
    Ok(Some(
        unsafe { std::slice::from_raw_parts(pair.data, pair.size as usize) }.to_vec(),
    ))
}

fn required(pair: Pair) -> Result<Vec<u8>, &'static str> {
    copied(pair)?.ok_or("required pair absent")
}

fn le32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

fn put32(out: &mut [u8], at: usize, value: u32) {
    out[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn align16(at: usize) -> usize {
    (at + 15) & !15
}

/// The FSPX payload a compute description carries in parts: a 16-byte
/// header, a 16-byte directory entry per kind, each blob 16-byte aligned.
fn compute_payload(desc: &ComputeDesc) -> Result<Vec<u8>, &'static str> {
    let state = required(desc.kind0)?;
    let opaque = copied(desc.kind1)?;
    let code = required(desc.kind6)?;
    let records = copied(desc.kind7)?;
    // Seven dispatch-packet words are what the translator reads of kind 0.
    if state.len() < 0x1c {
        return Err("kind0 shorter than the dispatch packet");
    }
    if code.len() % 4 != 0 {
        return Err("kind6 not dword-aligned");
    }
    if let Some(records) = &records {
        if records.len() < 0x18 {
            return Err("kind7 shorter than the FSPX metadata header");
        }
        if le32(records, 8) as usize != code.len() / 4 {
            return Err("kind7 code length differs from kind6");
        }
    }

    let mut pieces: Vec<(u32, &[u8])> = vec![(0, &state)];
    if let Some(opaque) = &opaque {
        pieces.push((1, opaque));
    }
    pieces.push((6, &code));
    if let Some(records) = &records {
        pieces.push((7, records));
    }
    let mut at = align16(16 + 16 * pieces.len());
    let mut entries = Vec::with_capacity(pieces.len());
    for &(_, bytes) in &pieces {
        entries.push(at);
        at = align16(at + bytes.len());
    }
    let mut out = vec![0; at];
    out[..4].copy_from_slice(b"FSPX");
    put32(&mut out, 4, 1);
    put32(&mut out, 8, pieces.len() as u32);
    put32(&mut out, 12, at as u32);
    for (index, (&(kind, bytes), &at)) in pieces.iter().zip(&entries).enumerate() {
        let entry = 16 + 16 * index;
        put32(&mut out, entry, kind);
        put32(&mut out, entry + 4, at as u32);
        put32(&mut out, entry + 8, bytes.len() as u32);
        out[at..at + bytes.len()].copy_from_slice(bytes);
    }
    Ok(out)
}

/// `payload` as the single ACCC part of a shader container.
fn accc_container(payload: &[u8]) -> Vec<u8> {
    let part_at = 0x20 + 4;
    let total = part_at + 8 + payload.len();
    let mut out = vec![0; total];
    out[..4].copy_from_slice(b"DXBC");
    // Bytes 4..20, the digest, are not checked for ACCC and stay zero.
    put32(&mut out, 20, 1); // version 1.0
    put32(&mut out, 24, total as u32);
    put32(&mut out, 28, 1); // one part
    put32(&mut out, 32, part_at as u32);
    out[part_at..part_at + 4].copy_from_slice(b"ACCC");
    put32(&mut out, part_at + 4, payload.len() as u32);
    out[part_at + 8..].copy_from_slice(payload);
    out
}

fn null_out(out: *mut This) {
    if !out.is_null() {
        // SAFETY: the title's out pointer.
        unsafe { *out = std::ptr::null() };
    }
}

/// Device method 0x218 `(this, description, iid, out)`.
pub(crate) extern "system" fn compute(
    this: This,
    desc: *const ComputeDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    null_out(out);
    // SAFETY: the title passes a description or null.
    let payload = match unsafe { desc.as_ref() } {
        Some(desc) => compute_payload(desc),
        None => Err("null description"),
    };
    let hr = match payload {
        Ok(payload) => {
            let container = accc_container(&payload);
            let head = ComputeHead {
                // SAFETY: checked non-null above.
                root: unsafe { (*desc).root },
                code: container.as_ptr(),
                size: container.len(),
            };
            // `create_compute` copies the container into the shared table.
            crate::pipeline::create_compute(this, (&raw const head).cast(), iid, out)
        }
        Err(why) => {
            tracing::debug!(why, "compute pipeline state not deserialized");
            E_INVALIDARG
        }
    };
    COUNTS[usize::from(hr != S_OK)].fetch_add(1, Relaxed);
    hr
}

/// The two measured shapes of state blob A: 380 or 464 bytes plus up to
/// five or seven 8-byte records, word 0 a tag whose bits 3 and 4 vary.
fn state_a_shape(blob: &[u8]) -> bool {
    let size = blob.len();
    let measured = (380..=420).contains(&size) && (size - 380).is_multiple_of(8)
        || (464..=520).contains(&size) && (size - 464).is_multiple_of(8);
    measured && (le32(blob, 0) & !0x18) == 0x4005_d800
}

/// Validates a graphics description; the stage sizes, `index:code:records`.
fn read_graphics(desc: &GraphicsDesc) -> Result<String, &'static str> {
    if let Some(a) = copied(desc.state_a)?
        && !state_a_shape(&a)
    {
        return Err("state A size or tag not a measured shape");
    }
    if required(desc.state_b)?.len() < 0x50 {
        return Err("state B shorter than 0x50 bytes");
    }
    let mut stages = Vec::new();
    for (index, stage) in desc.stages.iter().enumerate() {
        let code = copied(stage.code)?;
        let records = copied(stage.records)?;
        if code.as_ref().is_some_and(|c| c.len() % 4 != 0) {
            return Err("stage code not dword-aligned");
        }
        if code.is_some() || records.is_some() {
            stages.push(format!("{index}:{}:{}", stage.code.size, stage.records.size));
        }
    }
    if stages.is_empty() {
        return Err("no stage has code or records");
    }
    Ok(stages.join(","))
}

/// Device method 0x1c8 `(this, description, iid, out)`.
/// shortcut: fails as before once the description is read; a host
/// pipeline needs the fixed-function blobs decoded first.
pub(crate) extern "system" fn graphics(
    _: This,
    desc: *const GraphicsDesc,
    _iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    null_out(out);
    // SAFETY: the title passes a description or null.
    let read = match unsafe { desc.as_ref() } {
        Some(d) => read_graphics(d).map(|stages| (d.state_a.size, d.state_b.size, stages)),
        None => Err("null description"),
    };
    let count = COUNTS[2 + usize::from(read.is_err())].fetch_add(1, Relaxed) + 1;
    if count <= 4 || count.is_power_of_two() {
        match read {
            Ok((state_a, state_b, stages)) => tracing::info!(
                count,
                state_a,
                state_b,
                stages,
                "graphics pipeline state read, left to the title (E_NOTIMPL)"
            ),
            Err(why) => tracing::warn!(count, why, "graphics pipeline state malformed"),
        }
    }
    E_NOTIMPL
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{E_NOINTERFACE, slot};
    use victory_shader::native::StageIo;
    use victory_shader::{Container, FourCC, HwStage};

    /// `ID3D12PipelineState`.
    const PIPELINE: Iid = crate::object::guid(
        0x765a30f3,
        0xf624,
        0x4c6f,
        [0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45],
    );

    type Deserialize<D> = extern "system" fn(This, *const D, *const Iid, *mut This) -> HRESULT;

    fn pair(bytes: &[u8]) -> Pair {
        Pair {
            data: bytes.as_ptr(),
            size: bytes.len() as u64,
        }
    }

    fn words(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    struct Compute {
        state: Vec<u8>,
        opaque: Vec<u8>,
        code: Vec<u8>,
        records: Vec<u8>,
    }

    impl Compute {
        fn new() -> Self {
            // Threads 32x1x1, wave32, sixteen VGPRs, a nonzero RSRC2.
            let state = words(&[0, 0, 0, 32 | 1 << 16, 1 | 0x10 << 16, 1, 0x9c]);
            let code = vec![0x00, 0x00, 0x81, 0xbf]; // s_endpgm
            let mut records = words(&[0x403, 0, code.len() as u32 / 4, 0, 0]);
            records.extend_from_slice(&[0x40, 4, 0, 0]);
            let mut hash_record = [0; 32];
            hash_record[0] = 0x43;
            records.extend_from_slice(&hash_record);
            let len = records.len() as u32 - 0x14;
            records[4..8].copy_from_slice(&len.to_le_bytes());
            Self {
                state,
                opaque: vec![0; 0x14],
                code,
                records,
            }
        }

        fn desc(&self) -> ComputeDesc {
            ComputeDesc {
                root: std::ptr::null(),
                kind0: pair(&self.state),
                kind1: pair(&self.opaque),
                kind6: pair(&self.code),
                kind7: pair(&self.records),
            }
        }
    }

    struct Graphics {
        state_a: Vec<u8>,
        state_b: Vec<u8>,
        code: Vec<u8>,
    }

    impl Graphics {
        fn new() -> Self {
            let mut state_a = vec![0; 380];
            state_a[..4].copy_from_slice(&0x4005_d800u32.to_le_bytes());
            Self {
                state_a,
                state_b: vec![0; 0x50],
                code: vec![0x00, 0x00, 0x81, 0xbf],
            }
        }

        fn desc(&self) -> GraphicsDesc {
            let absent = Pair {
                data: std::ptr::null(),
                size: 0,
            };
            let mut stages = [StageDesc {
                code: absent,
                records: absent,
            }; 7];
            stages[0].code = pair(&self.code);
            GraphicsDesc {
                root: std::ptr::null(),
                state_a: pair(&self.state_a),
                state_b: pair(&self.state_b),
                stages,
            }
        }
    }

    #[test]
    fn compute_payload_is_a_translatable_fspx_program() {
        let input = Compute::new();
        let payload = compute_payload(&input.desc()).expect("well-formed");
        let container = accc_container(&payload);
        let parsed = Container::parse(&container).expect("container");
        assert_eq!(parsed.part(FourCC::ACCC).expect("ACCC").data, payload);
        let programs = victory_shader::native_programs(&payload).expect("FSPX");
        victory_shader::emit::translate(&container).expect("translates");
        let program = &programs[0];
        assert_eq!(program.stage, HwStage::Cs);
        assert!(!program.wave64);
        assert_eq!(program.vgprs, 16);
        assert_eq!(program.code, input.code);
        let StageIo::Cs(io) = &program.io else {
            panic!("not compute");
        };
        assert_eq!((io.threads, io.rsrc2), ([32, 1, 1], 0x9c));
    }

    #[test]
    fn compute_slot_makes_pipelines_and_refuses_malformed_ones() {
        let device = crate::device::device();
        let deserialize = slot::<Deserialize<ComputeDesc>>(device, 0x218);
        let input = Compute::new();
        let desc = input.desc();
        let mut p = std::ptr::null();
        assert_eq!(deserialize(device, &desc, &PIPELINE, &mut p), S_OK);
        assert!(!p.is_null());
        assert_eq!(slot::<extern "system" fn(This) -> u32>(p, 0x10)(p), 0);
        let mut out = device;
        assert_eq!(deserialize(device, &desc, &[0; 16], &mut out), E_NOINTERFACE);
        assert!(out.is_null());

        let mut out = device;
        assert_eq!(
            deserialize(device, std::ptr::null(), &PIPELINE, &mut out),
            E_INVALIDARG
        );
        assert!(out.is_null());
        let mut records = input.records.clone();
        records[8] = 2; // two code dwords; kind 6 has one
        let edits: [fn(&mut ComputeDesc); 5] = [
            |d| d.kind0.size = 0,
            |d| d.kind6.size = 0,
            |d| d.kind0.size = 0x1b,
            |d| d.kind6.size = 3,
            |d| d.kind1.data = std::ptr::null(),
        ];
        for edit in edits {
            let mut desc = input.desc();
            edit(&mut desc);
            assert!(compute_payload(&desc).is_err());
        }
        let mut desc = input.desc();
        desc.kind7 = pair(&records);
        assert!(compute_payload(&desc).is_err());
        // A zero size is absent even with a pointer.
        let mut desc = input.desc();
        desc.kind1 = Pair {
            data: std::ptr::dangling(),
            size: 0,
        };
        assert!(compute_payload(&desc).is_ok());
        assert!(
            copied(Pair {
                data: std::ptr::dangling(),
                size: MAX_PAIR_SIZE + 1
            })
            .is_err()
        );
    }

    #[test]
    fn graphics_descriptions_are_read_and_left_to_the_title() {
        let device = crate::device::device();
        let deserialize = slot::<Deserialize<GraphicsDesc>>(device, 0x1c8);
        let input = Graphics::new();
        assert_eq!(read_graphics(&input.desc()).as_deref(), Ok("0:4:0"));
        let mut out = device;
        assert_eq!(
            deserialize(device, &input.desc(), &PIPELINE, &mut out),
            E_NOTIMPL
        );
        assert!(out.is_null());
        let edits: [fn(&mut GraphicsDesc); 4] = [
            |d| d.state_b.size = 0x4f,
            |d| d.state_a.size = 421,
            |d| d.stages[0].code.size = 3,
            |d| d.stages[0].code.size = 0,
        ];
        for edit in edits {
            let mut desc = input.desc();
            edit(&mut desc);
            assert!(read_graphics(&desc).is_err());
        }
        for size in [388, 420, 464, 472, 520] {
            let mut blob = vec![0; size];
            blob[..4].copy_from_slice(&0x4005_d810u32.to_le_bytes());
            assert!(state_a_shape(&blob), "{size}");
        }
        for size in [379, 421, 463, 521] {
            let mut blob = vec![0; size];
            blob[..4].copy_from_slice(&0x4005_d800u32.to_le_bytes());
            assert!(!state_a_shape(&blob), "{size}");
        }
    }
}
