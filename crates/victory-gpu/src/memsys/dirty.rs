//! Pages written since they were last taken: a bit per page.
//!
//! Whoever learns of a write sets bits from any thread ([`Dirty::mark`]
//! for the CPU's trackers, [`Dirty::merge`] for a bitmap shaders marked
//! and the GPU copied back), and one reader takes them as runs
//! ([`Dirty::take`]). A bit set while the reader takes is in this take or
//! the next, never lost. Nothing is allocated after [`Dirty::new`].

use std::sync::atomic::{AtomicU64, Ordering};

/// A set of pages.
pub(crate) struct Dirty {
    bits: Box<[AtomicU64]>,
    pages: usize,
}

/// The bits `[lo, hi)` of a word (`lo < hi <= 64`).
fn mask(lo: usize, hi: usize) -> u64 {
    (u64::MAX >> (64 - (hi - lo))) << lo
}

impl Dirty {
    /// `pages` pages, none written.
    pub(crate) fn new(pages: usize) -> Self {
        Self { bits: (0..pages.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(), pages }
    }

    /// The `count` pages from `first` were written (cut to the set).
    pub(crate) fn mark(&self, first: usize, count: usize) {
        let end = first.saturating_add(count).min(self.pages);
        let mut page = first;
        while page < end {
            let word = page / 64;
            let hi = (end - word * 64).min(64);
            self.bits[word].fetch_or(mask(page % 64, hi), Ordering::Release);
            page = (word + 1) * 64;
        }
    }

    /// The pages `words` has bits for were written, `words[0]` being the
    /// set's word `at` (64 pages a word; cut to the set).
    pub(crate) fn merge(&self, at: usize, words: &[u64]) {
        for (cell, &word) in self.bits.iter().skip(at).zip(words) {
            if word != 0 {
                cell.fetch_or(word, Ordering::Release);
            }
        }
    }

    /// Whether any page is written.
    pub(crate) fn any(&self) -> bool {
        self.bits.iter().any(|w| w.load(Ordering::Acquire) != 0)
    }

    /// Takes every written page, calling `run(first, count)` for each run
    /// of them in order; they are not written after. Returns the pages.
    pub(crate) fn take(&self, mut run: impl FnMut(usize, usize)) -> usize {
        let mut open: Option<(usize, usize)> = None;
        let mut total = 0;
        for (w, cell) in self.bits.iter().enumerate() {
            if cell.load(Ordering::Relaxed) == 0 {
                continue;
            }
            let mut word = cell.swap(0, Ordering::AcqRel);
            // A bit past the last page (from `merge`) names no page.
            if w == self.pages / 64 {
                word &= mask(0, self.pages % 64);
            }
            while word != 0 {
                let lo = word.trailing_zeros() as usize;
                let n = (word >> lo).trailing_ones() as usize;
                let first = w * 64 + lo;
                total += n;
                open = match open {
                    Some((start, len)) if start + len == first => Some((start, len + n)),
                    Some((start, len)) => {
                        run(start, len);
                        Some((first, n))
                    }
                    None => Some((first, n)),
                };
                word = if lo + n == 64 { 0 } else { word & !mask(lo, lo + n) };
            }
        }
        if let Some((start, len)) = open {
            run(start, len);
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs(dirty: &Dirty) -> (Vec<(usize, usize)>, usize) {
        let mut runs = Vec::new();
        let total = dirty.take(|first, count| runs.push((first, count)));
        (runs, total)
    }

    #[test]
    fn marked_pages_come_out_as_runs_once() {
        let dirty = Dirty::new(300);
        assert!(!dirty.any());
        dirty.mark(3, 2);
        dirty.mark(4, 3);
        dirty.mark(60, 10);
        dirty.mark(128, 64);
        dirty.mark(192, 1);
        dirty.mark(299, 1);
        assert!(dirty.any());
        // Runs join over word boundaries (60..70, 128..193).
        assert_eq!(runs(&dirty), (vec![(3, 4), (60, 10), (128, 65), (299, 1)], 80));
        assert!(!dirty.any());
        assert_eq!(runs(&dirty), (vec![], 0), "taken once");
    }

    #[test]
    fn marks_are_cut_to_the_set() {
        let dirty = Dirty::new(70);
        dirty.mark(68, 100);
        dirty.mark(500, 3);
        dirty.mark(0, 0);
        assert_eq!(runs(&dirty), (vec![(68, 2)], 2));
        dirty.mark(0, usize::MAX);
        assert_eq!(runs(&dirty), (vec![(0, 70)], 70));
    }

    #[test]
    fn a_bitmap_read_back_is_merged_in() {
        let dirty = Dirty::new(200);
        dirty.mark(64, 1);
        // Words 1 to 3: pages 65 and 66, page 128, pages 192 and 194; the
        // fourth word is past the set, as are bits past its last page.
        dirty.merge(1, &[0b110, 1, 5, 9]);
        dirty.merge(3, &[u64::MAX]);
        assert_eq!(runs(&dirty), (vec![(64, 3), (128, 1), (192, 8)], 12));
    }
}
