//! What the title may see of submissions still on their way.
//!
//! The title learns that the GPU is done only from fence values, labels
//! (words lists write at the bottom of the pipe) and query results. So a
//! submission need not be finished before the next is prepared, as long
//! as none of those is visible before the bytes it stands for are in
//! title memory. The gate keeps that order: each host submission takes a
//! ticket ([`Gate::submit`]), and its signals are raised when the ticket
//! is retired ([`Gate::retire`]), after its copy-back.
//!
//! Lists of other queues wait on labels. The CPU cannot test such a wait
//! while the list that writes the label is still on the GPU, but it need
//! not: all host work runs on one queue, so a list sent after the writer
//! runs after it. [`Gate::wait`] says which it is.
//!
//! Nothing is allocated after [`Gate::new`].

/// A word (`bytes`: 4 or 8) a submission in flight writes at the bottom
/// of the pipe.
#[derive(Debug, Clone, Copy)]
struct Label {
    address: u64,
    bytes: u8,
    /// None: only the GPU knows it (a query result, a timestamp).
    value: Option<u64>,
    ticket: u64,
}

/// What is known of a wait on a word of title memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Seen {
    /// The word holds a value the wait accepts, and nothing in flight
    /// writes it.
    Now,
    /// The submission with this ticket writes a value the wait accepts:
    /// work sent to the host queue after it runs after the write.
    After(u64),
    /// Not known until what is in flight is retired, or not met.
    Unknown,
}

/// Why [`Gate::label`] kept no label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Refused {
    /// No room: the caller retires what is in flight first.
    Full,
    /// The ticket is not in flight. Retired, its write is in title memory
    /// and the word itself answers waits; a label kept for it would
    /// answer `After` a submission nothing waits behind any more. Not
    /// submitted yet, it names no submission to run after.
    Ticket,
}

/// Tickets and the labels of those in flight.
pub(crate) struct Gate {
    /// The last ticket given.
    submitted: u64,
    /// Every ticket up to this is retired.
    retired: u64,
    labels: Vec<Label>,
    most: usize,
    /// A label of this ticket found no room: until it is retired, no wait
    /// is answered from the labels.
    lost: u64,
}

impl Gate {
    /// A gate for at most `labels` label writes in flight.
    pub(crate) fn new(labels: usize) -> Self {
        Self { submitted: 0, retired: 0, labels: Vec::with_capacity(labels), most: labels, lost: 0 }
    }

    /// A ticket for the next submission (from 1, in order).
    pub(crate) fn submit(&mut self) -> u64 {
        self.submitted += 1;
        self.submitted
    }

    /// The submission numbered `ticket` by its caller is sent (numbers
    /// only grow; those skipped count as sent with it).
    pub(crate) fn submit_as(&mut self, ticket: u64) {
        self.submitted = self.submitted.max(ticket);
    }

    /// Submissions not retired.
    pub(crate) fn in_flight(&self) -> u64 {
        self.submitted - self.retired
    }

    /// Whether the results of `ticket` are in title memory, so that its
    /// fence values, labels and query results may be shown to the title.
    pub(crate) fn visible(&self, ticket: u64) -> bool {
        ticket <= self.retired
    }

    /// The submission `ticket`, in flight, writes the `bytes` bytes of
    /// `value` at `address` at the bottom of the pipe (None: a value only
    /// the GPU knows).
    pub(crate) fn label(
        &mut self,
        address: u64,
        bytes: u8,
        value: Option<u64>,
        ticket: u64,
    ) -> Result<(), Refused> {
        if ticket <= self.retired || ticket > self.submitted {
            return Err(Refused::Ticket);
        }
        if self.labels.len() == self.most {
            self.lost = self.lost.max(ticket);
            return Err(Refused::Full);
        }
        self.labels.push(Label { address, bytes, value, ticket });
        Ok(())
    }

    /// The copy-back of every submission up to `ticket` is in title
    /// memory, and their labels are written.
    pub(crate) fn retire(&mut self, ticket: u64) {
        self.retired = self.retired.max(ticket.min(self.submitted));
        let retired = self.retired;
        self.labels.retain(|l| l.ticket > retired);
    }

    /// What the word of `bytes` bytes at `address` will hold once every
    /// label in flight is written: `None` when none touches it (the word
    /// itself answers), the value and the ticket that writes it, or
    /// `Err(ticket)` when it is known only once that ticket is retired.
    pub(crate) fn foreseen(&self, address: u64, bytes: u8) -> Option<Result<(u64, u64), u64>> {
        // A label was not kept: any word may be the one it writes.
        if self.lost > self.retired {
            return Some(Err(self.lost));
        }
        // The last write in flight that touches any byte of the word
        // decides: the word's value now is about to be replaced.
        let end = address.saturating_add(u64::from(bytes));
        let l = self
            .labels
            .iter()
            .filter(|l| l.address < end && address < l.address.saturating_add(u64::from(l.bytes)))
            .fold(None, |last: Option<&Label>, l| match last {
                Some(found) if found.ticket > l.ticket => Some(found),
                _ => Some(l),
            })?;
        // The same word: its value is the write's. Part of the word, or
        // more than it: not worked out here.
        Some(l.value.filter(|_| l.address == address && l.bytes == bytes).map(|v| (v, l.ticket)).ok_or(l.ticket))
    }

    /// A wait on the word of `bytes` bytes at `address`, which holds
    /// `current` now and is met by the values `holds` accepts.
    pub(crate) fn wait(&self, address: u64, bytes: u8, current: u64, holds: impl Fn(u64) -> bool) -> Seen {
        match self.foreseen(address, bytes) {
            Some(Ok((value, ticket))) if holds(value) => Seen::After(ticket),
            Some(_) => Seen::Unknown,
            None if holds(current) => Seen::Now,
            None => Seen::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ticket_is_visible_once_retired() {
        let mut gate = Gate::new(4);
        let (a, b, c) = (gate.submit(), gate.submit(), gate.submit());
        assert_eq!((a, b, c), (1, 2, 3));
        assert_eq!(gate.in_flight(), 3);
        assert!(!gate.visible(a));
        gate.retire(b);
        assert!(gate.visible(a) && gate.visible(b) && !gate.visible(c));
        assert_eq!(gate.in_flight(), 1);
        // Neither backwards nor past what was submitted.
        gate.retire(a);
        gate.retire(100);
        assert!(gate.visible(c) && !gate.visible(4));
        assert_eq!(gate.in_flight(), 0);
        assert_eq!(gate.submit(), 4);
    }

    #[test]
    fn a_wait_is_met_by_the_last_label_in_flight() {
        let mut gate = Gate::new(8);
        let is = |want: u64| move |value: u64| value == want;
        // Nothing in flight: the word itself.
        assert_eq!(gate.wait(0x1000, 4, 7, is(7)), Seen::Now);
        assert_eq!(gate.wait(0x1000, 4, 6, is(7)), Seen::Unknown);
        let (a, b) = (gate.submit(), gate.submit());
        assert_eq!(gate.label(0x1000, 4, Some(7), a), Ok(()));
        assert_eq!(gate.label(0x2000, 4, Some(1), a), Ok(()));
        // The word still holds the old value; the writer is in flight.
        assert_eq!(gate.wait(0x1000, 4, 6, is(7)), Seen::After(a));
        // A later write replaces it: the value now does not count.
        assert_eq!(gate.label(0x1000, 4, Some(8), b), Ok(()));
        assert_eq!(gate.wait(0x1000, 4, 7, is(7)), Seen::Unknown);
        assert_eq!(gate.wait(0x1000, 4, 7, is(8)), Seen::After(b));
        assert_eq!(gate.wait(0x2000, 4, 0, is(1)), Seen::After(a));
        // Retired: in title memory, so the word decides again.
        gate.retire(a);
        assert_eq!(gate.wait(0x2000, 4, 1, is(1)), Seen::Now);
        assert_eq!(gate.wait(0x1000, 4, 7, is(8)), Seen::After(b));
        gate.retire(b);
        assert_eq!(gate.wait(0x1000, 4, 8, is(8)), Seen::Now);
    }

    /// A write in flight that shares bytes with the word waited on makes
    /// its value unknown, whatever the word holds now; one next to it
    /// does not.
    #[test]
    fn a_label_over_part_of_the_word_leaves_it_unknown() {
        let mut gate = Gate::new(8);
        let a = gate.submit();
        // A 64-bit label over 0x1000..0x1008.
        assert_eq!(gate.label(0x1000, 8, Some(7), a), Ok(()));
        assert_eq!(gate.wait(0x1000, 8, 0, |v| v == 7), Seen::After(a));
        // Its low and high halves, and a word across its end.
        assert_eq!(gate.wait(0x1000, 4, 7, |v| v == 7), Seen::Unknown);
        assert_eq!(gate.wait(0x1004, 4, 0, |v| v == 0), Seen::Unknown);
        assert_eq!(gate.wait(0x1006, 4, 0, |v| v == 0), Seen::Unknown);
        assert_eq!(gate.wait(0xffc, 8, 0, |v| v == 0), Seen::Unknown);
        // The words on either side are their own.
        assert_eq!(gate.wait(0x1008, 4, 3, |v| v == 3), Seen::Now);
        assert_eq!(gate.wait(0xffc, 4, 3, |v| v == 3), Seen::Now);
        // A newer write of a half hides the whole word's value.
        let b = gate.submit();
        assert_eq!(gate.label(0x1004, 4, Some(1), b), Ok(()));
        assert_eq!(gate.wait(0x1000, 8, 0, |v| v == 7), Seen::Unknown);
        assert_eq!(gate.wait(0x1004, 4, 0, |v| v == 1), Seen::After(b));
    }

    /// Only a submission in flight has labels: one retired has written
    /// its own, and one not submitted is nothing to run after.
    #[test]
    fn a_label_needs_a_ticket_in_flight() {
        let mut gate = Gate::new(8);
        assert_eq!(gate.label(0x1000, 4, Some(1), 1), Err(Refused::Ticket), "not submitted");
        let (a, b) = (gate.submit(), gate.submit());
        assert_eq!(gate.label(0x1000, 4, Some(1), 0), Err(Refused::Ticket), "no such ticket");
        assert_eq!(gate.label(0x1000, 4, Some(1), b + 1), Err(Refused::Ticket), "not submitted");
        gate.retire(a);
        assert_eq!(gate.label(0x1000, 4, Some(1), a), Err(Refused::Ticket), "retired");
        // Nothing was kept: the word itself answers.
        assert_eq!(gate.wait(0x1000, 4, 1, |v| v == 1), Seen::Now);
        assert_eq!(gate.label(0x1000, 4, Some(2), b), Ok(()));
        assert_eq!(gate.wait(0x1000, 4, 1, |v| v == 2), Seen::After(b));
    }

    #[test]
    fn a_value_only_the_gpu_knows_is_not_guessed() {
        let mut gate = Gate::new(2);
        let a = gate.submit();
        assert_eq!(gate.label(0x3000, 8, None, a), Ok(()));
        assert_eq!(gate.wait(0x3000, 8, 5, |v| v == 5), Seen::Unknown);
        assert_eq!(gate.label(0x3008, 8, Some(1), a), Ok(()));
        assert_eq!(gate.label(0x3010, 8, Some(1), a), Err(Refused::Full), "no room: retire first");
        gate.retire(a);
        assert_eq!(gate.wait(0x3000, 8, 5, |v| v == 5), Seen::Now);
        let b = gate.submit();
        assert_eq!(gate.label(0x3010, 8, Some(1), b), Ok(()), "room again");
    }

    /// A label that found no room may write any word: nothing is answered
    /// from the labels until its ticket is retired.
    #[test]
    fn a_label_not_kept_leaves_every_wait_unknown() {
        let mut gate = Gate::new(1);
        gate.submit_as(7);
        assert_eq!(gate.in_flight(), 7, "numbers skipped count as sent");
        assert_eq!(gate.label(0x1000, 4, Some(1), 7), Ok(()));
        assert_eq!(gate.wait(0x2000, 4, 3, |v| v == 3), Seen::Now);
        assert_eq!(gate.label(0x2000, 4, Some(4), 7), Err(Refused::Full));
        assert_eq!(gate.wait(0x2000, 4, 3, |v| v == 3), Seen::Unknown, "it may be about to change");
        assert_eq!(gate.wait(0x1000, 4, 0, |v| v == 1), Seen::Unknown);
        assert_eq!(gate.foreseen(0x9000, 8), Some(Err(7)));
        gate.retire(6);
        assert_eq!(gate.wait(0x2000, 4, 3, |v| v == 3), Seen::Unknown, "not its ticket yet");
        gate.retire(7);
        assert_eq!(gate.wait(0x2000, 4, 4, |v| v == 4), Seen::Now);
        assert_eq!(gate.foreseen(0x1000, 4), None);
    }
}
