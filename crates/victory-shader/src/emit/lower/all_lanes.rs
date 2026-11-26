//! Which blocks every lane of a host wave reaches.

use super::*;

/// Blocks reached by every host lane: a fetch-inactive shuffle cannot
/// recover registers of invocations absent from a divergent host branch.
pub(super) fn all_lane_blocks(p: &Program, st: &Structured) -> Option<Vec<bool>> {
    let exit_block = st.exits.len();
    let n = exit_block + 1;
    let mut successors = vec![Vec::new(); n];
    let mut reaches_end = vec![false; n];
    reaches_end[exit_block] = true;
    let mut rpo = vec![usize::MAX; exit_block];
    for (index, block) in crate::ir::reverse_postorder(&p.blocks)
        .into_iter()
        .enumerate()
    {
        rpo[block] = index;
    }
    let mut aborts = vec![false; exit_block];
    for (b, exit) in st.exits.iter().enumerate() {
        match *exit {
            LaneExit::Jump(t) => {
                successors[b].push(t);
                if rpo[b] != usize::MAX && rpo[t] <= rpo[b] {
                    // `Lower::exit` adds a per-lane return at the iteration
                    // limit of an unconditional back edge. It bypasses the
                    // loop's normal exit and every later reconvergence point.
                    successors[b].push(exit_block);
                    aborts[b] = true;
                }
            }
            LaneExit::Branch {
                taken, not_taken, ..
            } => {
                successors[b].extend([taken, not_taken]);
            }
            LaneExit::End => successors[b].push(exit_block),
            LaneExit::Indirect => return None,
        }
    }
    loop {
        let mut changed = false;
        for b in (0..n).rev() {
            if !reaches_end[b] && successors[b].iter().any(|&q| reaches_end[q]) {
                reaches_end[b] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // Post-dominance without any path to the exit would be vacuous.
    if reaches_end.iter().any(|&reaches| !reaches) {
        return None;
    }
    let words = n.div_ceil(64);
    let mut post = vec![vec![u64::MAX; words]; n];
    for (b, next) in successors.iter().enumerate() {
        if next.is_empty() {
            post[b].fill(0);
            post[b][b / 64] |= 1 << (b % 64);
        }
    }
    loop {
        let mut changed = false;
        for b in (0..n).rev() {
            if successors[b].is_empty() {
                continue;
            }
            let mut row = vec![u64::MAX; words];
            for &q in &successors[b] {
                for (word, &other) in row.iter_mut().zip(&post[q]) {
                    *word &= other;
                }
            }
            row[b / 64] |= 1 << (b % 64);
            if row != post[b] {
                post[b] = row;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut all = vec![true; n];
    for (branch, exit) in st.exits.iter().enumerate() {
        let divergent = match *exit {
            LaneExit::Branch { when, .. } => !st
                .bools
                .support(when)
                .iter()
                .all(|a| matches!(a, Atom::Uniform(_))),
            _ => false,
        };
        if !divergent && !aborts[branch] {
            continue;
        }
        let mut seen = vec![false; n];
        let mut pending = successors[branch].clone();
        while let Some(b) = pending.pop() {
            if seen[b] || post[branch][b / 64] >> (b % 64) & 1 != 0 {
                continue;
            }
            seen[b] = true;
            all[b] = false;
            pending.extend(successors[b].iter().copied());
        }
    }
    all.truncate(exit_block);
    Some(all)
}
