//! Runs of pages in a heap: a bit per page, first fit.
//!
//! A heap of 64 MiB in 64 KiB pages is 16 words, and a search goes over
//! them a run of bits at a time, not a page at a time, so it reads a few
//! cache lines; nothing is allocated after [`Pages::new`]. Sizes are
//! not rounded past the page (rounding to powers of two would leave about
//! a quarter of the video memory unused for the sizes host copies have).

/// Which pages of a heap are taken.
pub(crate) struct Pages {
    /// A bit per page, set while it is taken.
    used: Box<[u64]>,
    pages: usize,
    free: usize,
}

impl Pages {
    /// `pages` pages, all free.
    pub(crate) fn new(pages: usize) -> Self {
        let mut made = Self { used: vec![0; pages.div_ceil(64)].into_boxed_slice(), pages, free: pages };
        // The last word's bits past the last page are never free.
        made.set(pages, made.used.len() * 64 - pages, true);
        made
    }

    /// Pages not taken.
    pub(crate) fn free(&self) -> usize {
        self.free
    }

    pub(crate) fn is_used(&self, page: usize) -> bool {
        (self.used[page / 64] >> (page % 64)) & 1 == 1
    }

    fn set(&mut self, at: usize, n: usize, used: bool) {
        for page in at..at + n {
            let bit = 1u64 << (page % 64);
            if used {
                self.used[page / 64] |= bit;
            } else {
                self.used[page / 64] &= !bit;
            }
        }
    }

    /// Takes the first run of `n` free pages: its first page.
    pub(crate) fn take(&mut self, n: usize) -> Option<usize> {
        if n == 0 || n > self.free {
            return None;
        }
        // The free run so far: its first page and its pages.
        let (mut at, mut run) = (0, 0);
        let mut found = None;
        'words: for (w, &word) in self.used.iter().enumerate() {
            let mut bit = 0;
            while bit < 64 {
                let rest = word >> bit;
                // Free pages from `bit` on in this word, else taken ones.
                let free = (rest.trailing_zeros() as usize).min(64 - bit);
                if free == 0 {
                    run = 0;
                    bit += rest.trailing_ones() as usize;
                    continue;
                }
                if run == 0 {
                    at = w * 64 + bit;
                }
                run += free;
                if run >= n {
                    found = Some(at);
                    break 'words;
                }
                bit += free;
            }
        }
        let at = found?;
        self.set(at, n, true);
        self.free -= n;
        Some(at)
    }

    /// Gives back the `n` pages from `at` that [`Pages::take`] gave.
    pub(crate) fn give(&mut self, at: usize, n: usize) {
        debug_assert!(at + n <= self.pages && (at..at + n).all(|p| self.is_used(p)), "pages given back twice");
        if at + n > self.pages {
            return;
        }
        self.set(at, n, false);
        self.free += n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_are_taken_first_fit_and_reused_once_given_back() {
        let mut pages = Pages::new(10);
        let (a, b, c) = (pages.take(3).unwrap(), pages.take(2).unwrap(), pages.take(4).unwrap());
        assert_eq!((a, b, c), (0, 3, 5));
        assert_eq!(pages.free(), 1);
        assert_eq!(pages.take(2), None, "one page left");
        pages.give(b, 2);
        assert_eq!(pages.take(3), None, "two free runs of 2 and 1 pages, apart");
        pages.give(c, 4);
        // Joined with both neighbours: pages 3..10.
        assert_eq!(pages.take(7), Some(3));
        pages.give(a, 3);
        assert_eq!(pages.take(1), Some(0), "the first free run");
        assert_eq!(pages.free(), 2);
    }

    #[test]
    fn a_run_crosses_words_and_stops_at_the_last_page() {
        // 100 pages: the second word has 36.
        let mut pages = Pages::new(100);
        assert_eq!(pages.take(60), Some(0));
        assert_eq!(pages.take(40), Some(60), "over the word boundary");
        assert_eq!(pages.take(1), None, "full, and no page past the last");
        pages.give(60, 40);
        assert_eq!(pages.take(41), None, "40 pages, not the word's 64 bits");
        // A full first word is skipped whole.
        pages.give(0, 60);
        assert_eq!(pages.take(64), Some(0));
        assert_eq!(pages.take(36), Some(64));
        assert_eq!(pages.free(), 0);
    }

    /// Holes of every size in words that are neither empty nor full.
    #[test]
    fn the_first_hole_that_fits_is_taken() {
        let mut pages = Pages::new(192);
        assert_eq!(pages.take(192), Some(0));
        // Holes of 2 at 10, of 5 at 60 (over a word boundary), of 3 at 130.
        pages.give(10, 2);
        pages.give(60, 5);
        pages.give(130, 3);
        assert_eq!(pages.take(4), Some(60));
        assert_eq!(pages.take(3), Some(130));
        assert_eq!(pages.take(2), Some(10));
        assert_eq!(pages.take(1), Some(64), "what was left of the hole of 5");
        assert_eq!(pages.free(), 0);
    }

    #[test]
    fn nothing_is_taken_for_no_pages_or_too_many() {
        let mut pages = Pages::new(4);
        assert_eq!(pages.take(0), None);
        assert_eq!(pages.take(5), None);
        assert_eq!(pages.take(4), Some(0));
        assert_eq!(pages.free(), 0);
    }
}
