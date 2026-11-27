//! Copies waiting for a worker or for the GPU's copy queue.
//!
//! One thread asks for copies (the one preparing a submission) and one
//! takes them. The queue is a fixed ring of slots between the two: no
//! lock, and nothing allocated after [`CopyQueue::new`]. A full queue
//! refuses the copy, and the asking thread makes it itself or waits: the
//! copies never pile up behind a slow taker.

use std::sync::atomic::{AtomicU64, Ordering};

/// A copy of `len` bytes from `src` to `dst` for the submission `ticket`.
/// What the two addresses mean is the queue's user's: pointers for a copy
/// on the CPU, offsets in a staging ring and a resource for one on the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Job {
    pub(crate) dst: u64,
    pub(crate) src: u64,
    pub(crate) len: u64,
    pub(crate) ticket: u64,
}

/// A bounded queue of [`Job`]s from one thread to one other.
pub(crate) struct CopyQueue {
    slots: Box<[[AtomicU64; 4]]>,
    /// Jobs taken so far (written by the taking thread only).
    head: AtomicU64,
    /// Jobs pushed so far (written by the asking thread only).
    tail: AtomicU64,
    /// The most jobs that waited at once.
    most: AtomicU64,
}

impl CopyQueue {
    /// A queue of at most `capacity` waiting jobs.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            slots: (0..capacity.max(1)).map(|_| std::array::from_fn(|_| AtomicU64::new(0))).collect(),
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            most: AtomicU64::new(0),
        }
    }

    /// Jobs waiting.
    pub(crate) fn waiting(&self) -> usize {
        // The head first: read after the tail it could pass it.
        let head = self.head.load(Ordering::Acquire);
        (self.tail.load(Ordering::Acquire) - head) as usize
    }

    /// The most jobs that waited at once.
    pub(crate) fn most(&self) -> u64 {
        self.most.load(Ordering::Relaxed)
    }

    /// Queues `job`; false when the queue is full. From the one asking
    /// thread only.
    pub(crate) fn push(&self, job: Job) -> bool {
        let tail = self.tail.load(Ordering::Relaxed);
        let waiting = tail - self.head.load(Ordering::Acquire);
        if waiting as usize == self.slots.len() {
            return false;
        }
        let slot = &self.slots[(tail % self.slots.len() as u64) as usize];
        for (cell, value) in slot.iter().zip([job.dst, job.src, job.len, job.ticket]) {
            cell.store(value, Ordering::Relaxed);
        }
        // The taker reads the slot only after it sees the new tail.
        self.tail.store(tail + 1, Ordering::Release);
        self.most.fetch_max(waiting + 1, Ordering::Relaxed);
        true
    }

    /// The oldest job, if one waits. From the one taking thread only.
    pub(crate) fn pop(&self) -> Option<Job> {
        let head = self.head.load(Ordering::Relaxed);
        if head == self.tail.load(Ordering::Acquire) {
            return None;
        }
        let slot = &self.slots[(head % self.slots.len() as u64) as usize];
        let [dst, src, len, ticket] = std::array::from_fn(|i| slot[i].load(Ordering::Relaxed));
        // The asking thread reuses the slot only after it sees the new head.
        self.head.store(head + 1, Ordering::Release);
        Some(Job { dst, src, len, ticket })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(n: u64) -> Job {
        Job { dst: n, src: n + 1, len: n + 2, ticket: n + 3 }
    }

    #[test]
    fn jobs_come_out_in_order_and_a_full_queue_refuses() {
        let queue = CopyQueue::new(3);
        assert_eq!(queue.pop(), None);
        assert!(queue.push(job(1)) && queue.push(job(2)) && queue.push(job(3)));
        assert!(!queue.push(job(4)), "full");
        assert_eq!(queue.waiting(), 3);
        assert_eq!(queue.pop(), Some(job(1)));
        assert!(queue.push(job(4)), "a slot is free again");
        assert_eq!([queue.pop(), queue.pop(), queue.pop()], [Some(job(2)), Some(job(3)), Some(job(4))]);
        assert_eq!(queue.pop(), None);
        assert_eq!(queue.most(), 3);
    }

    /// One thread pushing and one taking: every job arrives once, in order.
    #[test]
    fn two_threads_pass_every_job_in_order() {
        const JOBS: u64 = 100_000;
        let queue = CopyQueue::new(64);
        std::thread::scope(|s| {
            s.spawn(|| {
                for n in 0..JOBS {
                    while !queue.push(job(n)) {
                        std::thread::yield_now();
                    }
                }
            });
            let mut next = 0;
            while next < JOBS {
                match queue.pop() {
                    Some(got) => {
                        assert_eq!(got, job(next));
                        next += 1;
                    }
                    None => std::thread::yield_now(),
                }
            }
        });
        assert_eq!(queue.pop(), None);
        assert!(queue.most() <= 64);
    }
}
