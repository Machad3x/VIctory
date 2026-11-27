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

/// A word a submission in flight writes at the bottom of the pipe.
#[derive(Debug, Clone, Copy)]
struct Label {
    address: u64,
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

/// Tickets and the labels of those in flight.
pub(crate) struct Gate {
    /// The last ticket given.
    submitted: u64,
    /// Every ticket up to this is retired.
    retired: u64,
    labels: Vec<Label>,
    most: usize,
}

impl Gate {
    /// A gate for at most `labels` label writes in flight.
    pub(crate) fn new(labels: usize) -> Self {
        Self { submitted: 0, retired: 0, labels: Vec::with_capacity(labels), most: labels }
    }

    /// A ticket for the next submission (from 1, in order).
    pub(crate) fn submit(&mut self) -> u64 {
        self.submitted += 1;
        self.submitted
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

    /// The submission `ticket` writes `value` at `address` at the bottom
    /// of the pipe (None: a value only the GPU knows). False when no more
    /// labels can be kept: the caller retires what is in flight first.
    pub(crate) fn label(&mut self, address: u64, value: Option<u64>, ticket: u64) -> bool {
        if self.labels.len() == self.most {
            return false;
        }
        self.labels.push(Label { address, value, ticket });
        true
    }

    /// The copy-back of every submission up to `ticket` is in title
    /// memory, and their labels are written.
    pub(crate) fn retire(&mut self, ticket: u64) {
        self.retired = self.retired.max(ticket.min(self.submitted));
        let retired = self.retired;
        self.labels.retain(|l| l.ticket > retired);
    }

    /// A wait on the word at `address`, which holds `current` now and is
    /// met by the values `holds` accepts.
    pub(crate) fn wait(&self, address: u64, current: u64, holds: impl Fn(u64) -> bool) -> Seen {
        // The last write in flight decides: the word's value now is about
        // to be replaced.
        let last = self
            .labels
            .iter()
            .filter(|l| l.address == address)
            .fold(None, |last: Option<&Label>, l| match last {
                Some(found) if found.ticket > l.ticket => Some(found),
                _ => Some(l),
            });
        match last {
            Some(Label { value: Some(value), ticket, .. }) if holds(*value) => Seen::After(*ticket),
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
        assert_eq!(gate.wait(0x1000, 7, is(7)), Seen::Now);
        assert_eq!(gate.wait(0x1000, 6, is(7)), Seen::Unknown);
        let (a, b) = (gate.submit(), gate.submit());
        assert!(gate.label(0x1000, Some(7), a));
        assert!(gate.label(0x2000, Some(1), a));
        // The word still holds the old value; the writer is in flight.
        assert_eq!(gate.wait(0x1000, 6, is(7)), Seen::After(a));
        // A later write replaces it: the value now does not count.
        assert!(gate.label(0x1000, Some(8), b));
        assert_eq!(gate.wait(0x1000, 7, is(7)), Seen::Unknown);
        assert_eq!(gate.wait(0x1000, 7, is(8)), Seen::After(b));
        assert_eq!(gate.wait(0x2000, 0, is(1)), Seen::After(a));
        // Retired: in title memory, so the word decides again.
        gate.retire(a);
        assert_eq!(gate.wait(0x2000, 1, is(1)), Seen::Now);
        assert_eq!(gate.wait(0x1000, 7, is(8)), Seen::After(b));
        gate.retire(b);
        assert_eq!(gate.wait(0x1000, 8, is(8)), Seen::Now);
    }

    #[test]
    fn a_value_only_the_gpu_knows_is_not_guessed() {
        let mut gate = Gate::new(2);
        let a = gate.submit();
        assert!(gate.label(0x3000, None, a));
        assert_eq!(gate.wait(0x3000, 5, |v| v == 5), Seen::Unknown);
        assert!(gate.label(0x3008, Some(1), a));
        assert!(!gate.label(0x3010, Some(1), a), "no room: retire first");
        gate.retire(a);
        assert_eq!(gate.wait(0x3000, 5, |v| v == 5), Seen::Now);
        assert!(gate.label(0x3010, Some(1), gate.submitted + 1), "room again");
    }
}
