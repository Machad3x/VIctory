//! Which physical pages the title's addresses show.
//!
//! The console maps one physical page at several addresses (a pool view
//! and a tiled heap, say). A host copy per address range then holds the
//! same bytes more than once: each is refreshed from the same title
//! memory, and a write through one address marks the copies at the
//! others, which fetch it again over the bus.
//!
//! The runtime shows every section of physical pages whole in a view of
//! its own, so a page's address there names the page itself, whatever
//! addresses the title maps it at. [`Phys`] keeps, per run of title
//! addresses, where its pages lie in those views. Host copies of buffers
//! are then found by that ([`Phys::home`]): a buffer whose pages another
//! buffer's copy already holds takes that copy, at the offset of its
//! pages, and none of its own (`crate::shadow::find`).
//!
//! `VICTORY_MEMSYS_ALIAS=1` ([`on`]), off unless set.
//! shortcut: a buffer shares a copy only when all its pages lie in one
//! physical run that the other copy holds whole; pages mapped scattered,
//! views outside buffer objects and textures keep copies of their own.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

#[cfg(test)]
thread_local! {
    /// Tests: [`on`] for this thread, whatever the environment says.
    pub(crate) static ON_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether buffers showing the same physical pages share one host copy.
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if ON_HERE.get() {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    super::alone("VICTORY_MEMSYS_ALIAS", &ON)
}

/// Runs of title addresses by their first: (bytes, where the run's pages
/// lie in the runtime's own views). Runs next to each other in both are
/// one.
#[derive(Default)]
pub(crate) struct Phys {
    runs: BTreeMap<u64, (u64, u64)>,
}

impl Phys {
    /// The title's `[at, at + len)` shows the pages at `own` from now on.
    pub(crate) fn shown(&mut self, mut at: u64, mut len: u64, mut own: u64) {
        if len == 0 {
            return;
        }
        self.remapped(at, len);
        // Joined to the run before it and the run after it, when they
        // continue each other in the own view as well.
        if let Some((&before, &(n, o))) = self.runs.range(..at).next_back()
            && before + n == at
            && o + n == own
        {
            self.runs.remove(&before);
            (at, len, own) = (before, len + n, o);
        }
        if let Some(&(n, o)) = self.runs.get(&(at + len))
            && own + len == o
        {
            self.runs.remove(&(at + len));
            len += n;
        }
        self.runs.insert(at, (len, own));
    }

    /// The title's `[at, at + len)` shows something else, or nothing.
    pub(crate) fn remapped(&mut self, at: u64, len: u64) {
        let end = at.saturating_add(len);
        // The run reaching in from below, then those starting inside.
        let first = self.runs.range(..at).next_back().filter(|&(&s, &(n, _))| s + n > at).map(|(&s, _)| s);
        let hit: Vec<u64> = first.into_iter().chain(self.runs.range(at..end).map(|(&s, _)| s)).collect();
        for start in hit {
            let (n, own) = self.runs.remove(&start).expect("a run just found");
            if start < at {
                self.runs.insert(start, (at - start, own));
            }
            if start + n > end {
                self.runs.insert(end, (start + n - end, own + (end - start)));
            }
        }
    }

    /// Where the pages of `[address, address + len)` lie in the runtime's
    /// own views, when they lie there in one piece.
    pub(crate) fn home(&self, address: u64, len: u64) -> Option<u64> {
        let (&start, &(n, own)) = self.runs.range(..=address).next_back()?;
        (address.checked_add(len)? <= start + n).then(|| own + (address - start))
    }
}

fn phys() -> std::sync::MutexGuard<'static, Phys> {
    static PHYS: Mutex<Phys> = Mutex::new(Phys { runs: BTreeMap::new() });
    PHYS.lock().unwrap_or_else(|e| e.into_inner())
}

/// [`Phys::shown`] of the title's memory.
pub(crate) fn shown(at: u64, len: u64, own: u64) {
    phys().shown(at, len, own);
}

/// [`Phys::remapped`] of the title's memory.
pub(crate) fn remapped(at: u64, len: u64) {
    phys().remapped(at, len);
}

/// [`Phys::home`] of the title's memory.
pub(crate) fn home(address: u64, len: u64) -> Option<u64> {
    phys().home(address, len)
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = 64 << 10;

    #[test]
    fn a_range_is_at_home_when_one_run_of_pages_shows_all_of_it() {
        let mut phys = Phys::default();
        // Two addresses of the same four pages, and one of other pages.
        phys.shown(0x10_0000, 4 * P, 0x7000_0000);
        phys.shown(0x90_0000, 4 * P, 0x7000_0000);
        phys.shown(0x50_0000, 2 * P, 0x7100_0000);
        assert_eq!(phys.home(0x10_0000, 4 * P), Some(0x7000_0000));
        assert_eq!(phys.home(0x90_0000 + P + 16, 256), Some(0x7000_0000 + P + 16), "the same pages");
        assert_eq!(phys.home(0x50_0000, P), Some(0x7100_0000));
        assert_eq!(phys.home(0x10_0000, 5 * P), None, "past the run");
        assert_eq!(phys.home(0x10_0000 - 1, 2), None, "from before it");
        assert_eq!(phys.home(0x40_0000, 1), None, "nothing shown there");
        assert_eq!(phys.home(u64::MAX, 2), None);
    }

    /// The runtime maps a page at a time: runs that continue each other
    /// at both ends are one, and those that do not stay apart.
    #[test]
    fn pages_mapped_one_by_one_join_when_they_follow_each_other() {
        let mut phys = Phys::default();
        phys.shown(0x10_0000 + P, P, 0x7000_0000 + P);
        phys.shown(0x10_0000 + 3 * P, P, 0x7000_0000 + 3 * P);
        phys.shown(0x10_0000 + 2 * P, P, 0x7000_0000 + 2 * P);
        phys.shown(0x10_0000, P, 0x7000_0000);
        assert_eq!(phys.runs.len(), 1);
        assert_eq!(phys.home(0x10_0000, 4 * P), Some(0x7000_0000));
        // The next address shows a page from elsewhere: scattered.
        phys.shown(0x10_0000 + 4 * P, P, 0x7000_0000 + 9 * P);
        assert_eq!(phys.runs.len(), 2);
        assert_eq!(phys.home(0x10_0000, 5 * P), None);
        assert_eq!(phys.home(0x10_0000 + 4 * P, P), Some(0x7000_0000 + 9 * P));
    }

    #[test]
    fn a_remap_cuts_the_runs_it_touches() {
        let mut phys = Phys::default();
        phys.shown(0x10_0000, 8 * P, 0x7000_0000);
        // The middle shows other pages now.
        phys.shown(0x10_0000 + 2 * P, 2 * P, 0x7200_0000);
        assert_eq!(phys.home(0x10_0000, 2 * P), Some(0x7000_0000));
        assert_eq!(phys.home(0x10_0000 + 2 * P, 2 * P), Some(0x7200_0000));
        assert_eq!(phys.home(0x10_0000 + 4 * P, 4 * P), Some(0x7000_0000 + 4 * P), "the rest keeps its pages");
        assert_eq!(phys.home(0x10_0000, 8 * P), None, "no longer one run");
        // Unmapped across two runs.
        phys.remapped(0x10_0000 + P, 2 * P);
        assert_eq!(phys.home(0x10_0000, P), Some(0x7000_0000));
        assert_eq!(phys.home(0x10_0000 + P, 1), None);
        assert_eq!(phys.home(0x10_0000 + 2 * P, 1), None);
        assert_eq!(phys.home(0x10_0000 + 3 * P, P), Some(0x7200_0000 + P));
        phys.remapped(0, u64::MAX);
        assert!(phys.runs.is_empty());
    }
}
