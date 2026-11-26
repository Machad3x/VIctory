//! Copies between textures of one memory that cannot share a host texture:
//! R32G32_UINT blocks the GPU may write, read as BC1.
use super::*;
use crate::object::slot;
use std::ptr::null;

const BLOCKS: u32 = 17;
const BC1: u32 = 71;
const RED: u64 = 0xF800;
const GREEN: u64 = 0x07E0;

/// A title texture at `address`: `BLOCKS` 16x16 with unordered access, or
/// `BC1` 64x64.
fn texture(address: u64, format: u32) -> This {
    let (size, flags) = if format == BLOCKS { (16, 4) } else { (64, 0) };
    let words: [u32; 14] = [3, 0, 0, 0, size, 0, size, 1 | 1 << 16, format, 1, 0, 0x105, flags, 0];
    let iid = guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
    let mut out = null();
    let made = crate::resource::create_at_address(null(), address, words.as_ptr().cast(), 0, null(), &iid, &mut out);
    assert_eq!(made, S_OK);
    out
}

/// Zeroed title memory, leaked as title memory outlives a test.
fn memory() -> u64 {
    let layout = std::alloc::Layout::from_size_align(1 << 20, 1 << 16).unwrap();
    // SAFETY: a non-zero size.
    unsafe { std::alloc::alloc_zeroed(layout) as u64 }
}

fn release(object: This) {
    slot::<extern "system" fn(This) -> u32>(object, 0x10)(object);
}

/// A texture the GPU writes, its aliased view and a title list.
struct Alias {
    from: This,
    to: This,
    title: This,
    allocator: This,
}

impl Alias {
    fn new() -> Option<Self> {
        crate::host::host()?;
        let mut allocator = null();
        assert_eq!(create_allocator(null(), 0, &ALLOCATOR_IIDS[0], &mut allocator), S_OK);
        let mut title = null();
        assert_eq!(create_list(null(), 0, 0, allocator, null(), &LIST_IIDS[1], &mut title), S_OK);
        let address = memory();
        let (from, to) = (texture(address, BLOCKS), texture(address, BC1));
        crate::resource::host_copy(from).unwrap();
        crate::resource::host_copy(to).unwrap();
        assert_eq!(crate::resource::alias_source(to), Some(from));
        Some(Alias { from, to, title, allocator })
    }

    fn submit(&self) {
        let list = host_list(self.title).unwrap();
        list.close();
        crate::host::host().unwrap().execute(&[list]);
    }
}

impl Drop for Alias {
    fn drop(&mut self) {
        [self.to, self.from, self.title, self.allocator].into_iter().for_each(release);
    }
}

/// A footprint of a `size` texel square of `format` in a new title buffer.
fn footprint(format: u32, size: u32) -> CopyLocation {
    CopyLocation {
        resource: crate::resource::test_buffer(256 * 16),
        kind: 1,
        footprint: crate::resource::PlacedFootprint { offset: 0, format, width: size, height: size, depth: 1, row_pitch: 256 },
    }
}

fn subresource(resource: This) -> CopyLocation {
    // SAFETY: all integers; zero is subresource index 0.
    CopyLocation { resource, kind: 0, footprint: unsafe { std::mem::zeroed() } }
}

/// Records a copy filling the top left `size` texels of `texture` with `block`.
fn write(a: &Alias, texture: This, format: u32, size: u32, block: u64) {
    let from = footprint(format, size);
    let (data, _) = crate::resource::buffer_bytes(from.resource).unwrap();
    let blocks = if format == BC1 { size as usize / 4 } else { size as usize };
    for y in 0..blocks {
        for x in 0..blocks {
            // SAFETY: rows of 256 bytes inside the buffer.
            unsafe { data.add(y * 256 + x * 8).cast::<u64>().write_unaligned(block) };
        }
    }
    copy_texture_region(a.title, &subresource(texture), 0, 0, 0, &from, null());
}

/// Records a read back of a `BC1` texture; the result, once submitted, gives
/// the block at (`x`, `y`).
fn read(a: &Alias, texture: This) -> impl Fn(usize, usize) -> u64 + use<> {
    let to = footprint(BC1, 64);
    copy_texture_region(a.title, &to, 0, 0, 0, &subresource(texture), null());
    let (data, _) = crate::resource::buffer_bytes(to.resource).unwrap();
    // SAFETY: a block inside the buffer, which is never released.
    move |x, y| unsafe { data.add(y * 256 + x * 8).cast::<u64>().read_unaligned() }
}

fn copies() -> u64 {
    crate::texture::OWN_ALIAS_COPIES.get()
}

#[test]
fn a_source_rewritten_within_a_list_is_copied_again() {
    let Some(a) = Alias::new() else { return };
    write(&a, a.from, BLOCKS, 16, RED);
    let first = read(&a, a.to);
    write(&a, a.from, BLOCKS, 16, GREEN);
    let second = read(&a, a.to);
    a.submit();
    assert_eq!((first(0, 0), second(0, 0)), (RED, GREEN));
}

#[test]
fn a_destination_written_after_its_alias_is_kept() {
    let Some(a) = Alias::new() else { return };
    write(&a, a.from, BLOCKS, 16, RED);
    write(&a, a.to, BC1, 64, GREEN);
    let before = copies();
    let got = read(&a, a.to);
    a.submit();
    assert_eq!((got(0, 0), copies() - before), (GREEN, 0));
}

#[test]
fn each_copy_path_reads_what_an_alias_wrote() {
    // Whole resource, texture region, read back.
    for path in 0..3 {
        let Some(a) = Alias::new() else { return };
        write(&a, a.from, BLOCKS, 16, RED);
        let other = texture(memory(), BC1);
        match path {
            0 => copy_resource(a.title, other, a.to),
            1 => copy_texture_region(a.title, &subresource(other), 0, 0, 0, &subresource(a.to), null()),
            _ => {}
        }
        let got = read(&a, if path == 2 { a.to } else { other });
        a.submit();
        assert_eq!(got(15, 15), RED, "path {path}");
        release(other);
    }
}

#[test]
fn a_box_copy_keeps_the_alias_data_outside_the_box() {
    let Some(a) = Alias::new() else { return };
    write(&a, a.from, BLOCKS, 16, RED);
    write(&a, a.to, BC1, 4, GREEN);
    let got = read(&a, a.to);
    a.submit();
    assert_eq!((got(0, 0), got(1, 0), got(15, 15)), (GREEN, RED, RED));
}

#[test]
fn two_reads_without_a_rewrite_copy_once() {
    let Some(a) = Alias::new() else { return };
    write(&a, a.from, BLOCKS, 16, RED);
    let before = copies();
    let (first, second) = (read(&a, a.to), read(&a, a.to));
    a.submit();
    assert_eq!((first(0, 0), second(0, 0), copies() - before), (RED, RED, 1));
}
