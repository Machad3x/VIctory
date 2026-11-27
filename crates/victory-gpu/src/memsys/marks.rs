//! A submission's write marks, its own from the moment it ends on the GPU.
//!
//! Shaders mark the pages they write through views in one bitmap
//! (`crate::shadow`), which the CPU reads and clears once the submission
//! has run. That makes the bitmap the property of whichever submission
//! ran last: nothing else may reach the GPU before the CPU has read it.
//!
//! Here a list of ours rides behind the submission's own lists: it copies
//! the bitmaps' used words into a ring of read-back memory and clears
//! them, on the GPU, in queue order. The span of the ring belongs to the
//! submission's ticket ([`Snap`]) until it is retired, and the bitmap is
//! clear for whatever runs next.
//!
//! `VICTORY_MEMSYS_GATE=1` ([`super::gating`]).
//! shortcut: the bitmaps are system memory, so each submission copies
//! their used words twice over the bus; with the bitmap in video memory
//! (#2465) the clear stays on the card.

use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::ring::Ring;
use crate::host::{Host, transition};

/// Bytes of the ring: eight submissions' worth of both bitmaps whole.
const RING: u64 = 8 * 2 * BITMAP;
/// Bytes of a bitmap's write marks.
const BITMAP: u64 = (victory_shader::emit::VIEW_WRITE_BITS / 8) as u64;
/// Tickets with a span in the ring at once.
const TICKETS: usize = 8;

/// The marks one submission's shaders left: where each bitmap's used
/// words lie in the ring, until [`done`].
pub(crate) struct Snap {
    ticket: u64,
    /// By bitmap (`memory`): the mapped words.
    spans: [(usize, usize); 2],
}

// SAFETY: the spans point into the ring's persistent mapping, which only
// the snap's holder reads until it gives the span back.
unsafe impl Send for Snap {}

impl Snap {
    /// The words of the bitmap of other shaders, or (`memory`) of shaders
    /// that use title memory, as the submission left them. Only once the
    /// submission has run.
    pub(crate) fn words(&self, memory: bool) -> &[u32] {
        let (at, words) = self.spans[usize::from(memory)];
        if words == 0 {
            return &[];
        }
        // SAFETY: a span of the mapped ring handed to this snap alone, at
        // a multiple of 4 bytes; the GPU filled it before the wait ended.
        unsafe { std::slice::from_raw_parts(at as *const u32, words) }
    }
}

struct Taker {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    ring: Ring,
    back: ID3D12Resource,
    data: usize,
    zeros: ID3D12Resource,
    /// The submission its list last rode.
    value: u64,
}

// SAFETY: D3D12 objects are free-threaded; used under the mutex only.
unsafe impl Send for Taker {}

static TAKER: OnceLock<Option<Mutex<Taker>>> = OnceLock::new();

fn taker(host: &Host) -> Option<&'static Mutex<Taker>> {
    TAKER
        .get_or_init(|| {
            // SAFETY: plain creation; the list is closed until used.
            let made = unsafe {
                (|| -> windows::core::Result<Taker> {
                    let none = windows::core::Error::empty;
                    let allocator: ID3D12CommandAllocator =
                        host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                    let list: ID3D12GraphicsCommandList =
                        host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                    list.Close()?;
                    let (back, data) = host.readback_buffer(RING).ok_or_else(none)?;
                    let zeros = host.uav_buffer(BITMAP).ok_or_else(none)?;
                    crate::host::name(&back, || "write marks by submission".into());
                    crate::host::name(&zeros, || "zeros for the write marks".into());
                    Ok(Taker { allocator, list, ring: Ring::new(RING, TICKETS), back, data: data as usize, zeros, value: 0 })
                })()
            };
            made.inspect_err(|e| tracing::warn!(error = %e, "write marks are not taken by submission: read after each"))
                .ok()
                .map(Mutex::new)
        })
        .as_ref()
}

/// Records into `list` the copies that take the bitmaps' used words into
/// the ring and clear them: the snap, for the ticket the list runs with.
/// None when there is nothing to take or no room (the bitmap is then read
/// where it lies, as without this).
fn record(t: &mut Taker, list: &ID3D12GraphicsCommandList, ticket: u64) -> Option<Snap> {
    debug_assert!(BITMAP.is_multiple_of(4) && RING.is_multiple_of(4));
    let bitmaps = crate::shadow::mark_bitmaps();
    let mut spans = [(0, 0); 2];
    let (common, source, dest) =
        (D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    for (memory, (bitmap, words)) in bitmaps.iter().enumerate().filter_map(|(m, b)| Some((m, b.as_ref()?))) {
        let bytes = 4 * *words as u64;
        if bytes == 0 {
            continue;
        }
        let Some(at) = t.ring.alloc(bytes, 4) else {
            // What this list took so far is freed with the next retire.
            t.ring.seal(ticket);
            super::add(super::N::MarksNoRoom, 1);
            return None;
        };
        // SAFETY: an open list of ours; the bitmap, the ring and the zeros
        // are live and the ranges inside them (`words` is cut to the
        // bitmap, a span of the ring is `bytes` long). The bitmap is
        // common between submissions; the ring is always a copy's
        // destination.
        unsafe {
            list.ResourceBarrier(&[transition(bitmap, common, source)]);
            list.CopyBufferRegion(&t.back, at, bitmap, 0, bytes);
            list.ResourceBarrier(&[transition(bitmap, source, dest)]);
            list.CopyBufferRegion(bitmap, 0, &t.zeros, 0, bytes);
            // Marks of offsets past the bitmap land in its last word.
            list.CopyBufferRegion(bitmap, BITMAP - 4, &t.zeros, 0, 4);
            list.ResourceBarrier(&[transition(bitmap, dest, common)]);
        }
        spans[memory] = (t.data + at as usize, *words);
        super::add(super::N::MarksBytes, bytes);
    }
    t.ring.seal(ticket);
    if spans == [(0, 0); 2] {
        return None;
    }
    super::add(super::N::MarksSnaps, 1);
    Some(Snap { ticket, spans })
}

/// A list to run right after the lists of the submission numbered
/// `ticket`, before its fence is signalled, and the snap it fills.
pub(crate) fn take(host: &Host, ticket: u64) -> Option<(ID3D12CommandList, Snap)> {
    let mut t = taker(host)?.lock().unwrap_or_else(|e| e.into_inner());
    // Its last run was not waited for: the allocator may be in use.
    if host.completed() < t.value {
        return None;
    }
    // SAFETY: the allocator's last list has run (checked above).
    unsafe {
        t.allocator.Reset().ok()?;
        t.list.Reset(&t.allocator, None).ok()?;
    }
    let list = t.list.clone();
    let snap = record(&mut t, &list, ticket);
    // SAFETY: our open list.
    unsafe { list.Close() }.ok()?;
    let snap = snap?;
    t.value = ticket;
    Some((list.cast().ok()?, snap))
}

/// The submission was retired: its span of the ring is free.
pub(crate) fn done(snap: Snap) {
    if let Some(t) = TAKER.get().and_then(|t| t.as_ref()) {
        t.lock().unwrap_or_else(|e| e.into_inner()).ring.retire(snap.ticket);
    }
}

/// Whether marks can be taken on this host (tests).
#[cfg(test)]
pub(crate) fn ready(host: &Host) -> bool {
    taker(host).is_some()
}
