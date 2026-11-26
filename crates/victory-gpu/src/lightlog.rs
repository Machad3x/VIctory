//! Diagnostics for a whole run losing its indirect light: the lives of
//! render target, depth and unordered access host textures, and which
//! submissions with programs using title memory missed it, in order. Each
//! kind logs at most [`PER_SECOND`] lines a second (a few MB in 5 minutes)
//! and counts the lines it leaves out.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use crate::resource::MemoryKey;

const PER_SECOND: u32 = 40;
/// The world's ray traced pass. Other ray walking programs are found in the
/// pipeline sets logged.
pub(crate) const RAY: &str = "f607780aa8e502d0";

/// Second, lines logged in it, lines left out since the last one logged.
#[derive(Default)]
struct Rate(u64, u32, u64);

impl Rate {
    /// The lines left out before this one, if it may be logged.
    fn take(&mut self) -> Option<u64> {
        static START: LazyLock<Instant> = LazyLock::new(Instant::now);
        let second = START.elapsed().as_secs();
        if self.0 != second {
            (self.0, self.1) = (second, 0);
        }
        if self.1 >= PER_SECOND {
            self.2 += 1;
            return None;
        }
        self.1 += 1;
        Some(std::mem::take(&mut self.2))
    }
}

#[derive(Default)]
struct Log {
    textures: Rate,
    submissions: Rate,
    /// Addresses that have held a render target, depth or unordered access
    /// host texture.
    held: HashSet<u64>,
    /// Pipeline sets already logged, by number.
    sets: HashMap<BTreeSet<String>, usize>,
    order: u64,
}

static LOG: LazyLock<Mutex<Log>> = LazyLock::new(Mutex::default);

/// A host texture was made for the memory `key` at `address`, with `uses`
/// (1 render target, 2 depth, 4 unordered access); `replaced` if the
/// memory's live one did not take the title texture.
pub(crate) fn created(address: u64, key: &MemoryKey, uses: u32, replaced: bool) {
    let mut log = LOG.lock().unwrap_or_else(|e| e.into_inner());
    let again = log.held.contains(&address);
    if uses != 0 {
        log.held.insert(address);
    } else if !again {
        return;
    }
    if let Some(left_out) = log.textures.take() {
        let frame = crate::queue::presented();
        tracing::info!(frame, uses, again, replaced, left_out, ?key, "lightlog texture made");
    }
}

/// The last holder of a render target, depth or unordered access host
/// texture let it go.
pub(crate) fn dropped(key: &MemoryKey, uses: u32) {
    let mut log = LOG.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(left_out) = log.textures.take() {
        let frame = crate::queue::presented();
        tracing::info!(frame, uses, left_out, ?key, "lightlog texture dropped");
    }
}

/// A submission ran. Per list, in order: whether it has programs using
/// title memory, whether it missed some, and the pipelines it binds.
/// Submissions with neither such programs nor [`RAY`] only advance the
/// order.
pub(crate) fn submission(withholding: bool, lists: impl Iterator<Item = (bool, bool, BTreeSet<String>)>) {
    let mut log = LOG.lock().unwrap_or_else(|e| e.into_inner());
    log.order += 1;
    let mut text = Vec::new();
    for (i, (memory, missed, pipelines)) in lists.enumerate() {
        let ray = pipelines.iter().any(|p| p.contains(RAY));
        if !memory && !ray {
            continue;
        }
        let next = log.sets.len();
        let set = match log.sets.get(&pipelines) {
            Some(&set) => set,
            None => {
                // Not under the rate: later lines name the set by number.
                // shortcut: the first 2000 sets are spelled out.
                if next < 2000 {
                    tracing::info!(set = next, ?pipelines, "lightlog pipeline set");
                }
                log.sets.insert(pipelines, next);
                next
            }
        };
        let (m, r) = (if missed { " missed" } else { "" }, if ray { " ray" } else { "" });
        text.push(format!("l{i} set{set}{m}{r}"));
    }
    if text.is_empty() {
        return;
    }
    if let Some(left_out) = log.submissions.take() {
        let (n, frame) = (log.order, crate::queue::presented());
        tracing::info!(n, frame, withholding, left_out, lists = %text.join(", "), "lightlog submission");
    }
}
