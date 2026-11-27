//! A ring of bytes handed out front to back and freed by ticket.
//!
//! What a submission stages (uploads on the way in, pages read back on the
//! way out) lives until the submission is retired, and submissions retire
//! in order. So the bytes need no free list: [`Ring::alloc`] bumps a
//! position, [`Ring::seal`] notes where a ticket's bytes end, and
//! [`Ring::retire`] frees everything up to a ticket. Nothing is allocated
//! after [`Ring::new`].
//!
//! A full ring gives nothing: the caller retires its oldest ticket and
//! asks again, which is the back-pressure that keeps a title from running
//! ahead of the GPU.

/// Positions grow without wrapping (2^64 bytes outlast any run); the
/// offset in the buffer is the position modulo the capacity.
pub(crate) struct Ring {
    capacity: u64,
    /// Where the next bytes go.
    head: u64,
    /// Bytes before this are free.
    tail: u64,
    /// Sealed spans, oldest first from `first`: (ticket, the head then).
    marks: Box<[(u64, u64)]>,
    first: usize,
    count: usize,
}

impl Ring {
    /// A ring over `capacity` bytes, for at most `tickets` sealed tickets
    /// in flight (more share the newest one's span).
    pub(crate) fn new(capacity: u64, tickets: usize) -> Self {
        Self {
            capacity,
            head: 0,
            tail: 0,
            marks: vec![(0, 0); tickets.max(1)].into_boxed_slice(),
            first: 0,
            count: 0,
        }
    }

    /// Bytes in use, the padding at the ring's end included.
    pub(crate) fn used(&self) -> u64 {
        self.head - self.tail
    }

    /// `size` bytes in one piece at a multiple of `align` (a power of two
    /// that divides the capacity): their offset in the buffer. None when
    /// they do not fit until a ticket is retired, or never fit.
    pub(crate) fn alloc(&mut self, size: u64, align: u64) -> Option<u64> {
        debug_assert!(align.is_power_of_two() && self.capacity.is_multiple_of(align));
        if size == 0 || size > self.capacity {
            return None;
        }
        let mut at = self.head.checked_next_multiple_of(align)?;
        let offset = at % self.capacity;
        // Not across the end: from the start (a multiple of every `align`).
        if offset + size > self.capacity {
            at += self.capacity - offset;
        }
        if at + size - self.tail > self.capacity {
            return None;
        }
        self.head = at + size;
        Some(at % self.capacity)
    }

    /// The bytes handed out since the last seal belong to `ticket`
    /// (tickets never decrease).
    pub(crate) fn seal(&mut self, ticket: u64) {
        let slots = self.marks.len();
        if self.count > 0 {
            let last = (self.first + self.count - 1) % slots;
            debug_assert!(self.marks[last].0 <= ticket, "tickets in order");
            // The same ticket again, or no slot left: the newest span
            // grows, and is freed with the later ticket.
            if self.marks[last].0 == ticket || self.count == slots {
                self.marks[last] = (ticket, self.head);
                return;
            }
        }
        self.marks[(self.first + self.count) % slots] = (ticket, self.head);
        self.count += 1;
    }

    /// Frees the bytes of every ticket up to `ticket`.
    pub(crate) fn retire(&mut self, ticket: u64) {
        while self.count > 0 && self.marks[self.first].0 <= ticket {
            self.tail = self.marks[self.first].1;
            self.first = (self.first + 1) % self.marks.len();
            self.count -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_are_handed_out_in_order_and_freed_by_ticket() {
        let mut ring = Ring::new(1024, 4);
        assert_eq!(ring.alloc(100, 1), Some(0));
        assert_eq!(ring.alloc(100, 256), Some(256), "aligned");
        ring.seal(1);
        assert_eq!(ring.alloc(600, 1), Some(356));
        ring.seal(2);
        assert_eq!(ring.used(), 956);
        assert_eq!(ring.alloc(100, 1), None, "68 bytes left");
        ring.retire(1);
        assert_eq!(ring.used(), 600);
        // 68 bytes at the end do not hold it: from the start.
        assert_eq!(ring.alloc(100, 1), Some(0));
        assert_eq!(ring.used(), 768, "the padding at the end counts");
        assert_eq!(ring.alloc(300, 1), None, "it would run into ticket 2's bytes");
        ring.retire(2);
        assert_eq!(ring.used(), 168, "unsealed bytes stay");
        assert_eq!(ring.alloc(300, 1), Some(100));
    }

    #[test]
    fn nothing_for_no_bytes_or_more_than_the_ring() {
        let mut ring = Ring::new(256, 2);
        assert_eq!(ring.alloc(0, 1), None);
        assert_eq!(ring.alloc(257, 1), None);
        assert_eq!(ring.alloc(256, 256), Some(0));
        assert_eq!(ring.alloc(1, 1), None);
        // Unsealed bytes are never freed.
        ring.retire(u64::MAX);
        assert_eq!(ring.used(), 256);
        ring.seal(7);
        ring.retire(6);
        assert_eq!(ring.used(), 256);
        ring.retire(7);
        assert_eq!(ring.used(), 0);
        assert_eq!(ring.alloc(256, 256), Some(0), "the whole ring again");
    }

    #[test]
    fn tickets_past_the_slots_share_the_newest_span() {
        let mut ring = Ring::new(1024, 2);
        for ticket in 1..=4 {
            assert!(ring.alloc(100, 1).is_some());
            ring.seal(ticket);
        }
        ring.retire(1);
        assert_eq!(ring.used(), 300);
        // Tickets 2 to 4 are one span, freed with ticket 4.
        ring.retire(3);
        assert_eq!(ring.used(), 300);
        ring.retire(4);
        assert_eq!(ring.used(), 0);
        // The same ticket sealed twice is one span too.
        assert!(ring.alloc(10, 1).is_some());
        ring.seal(5);
        assert!(ring.alloc(10, 1).is_some());
        ring.seal(5);
        ring.retire(5);
        assert_eq!(ring.used(), 0);
    }
}
