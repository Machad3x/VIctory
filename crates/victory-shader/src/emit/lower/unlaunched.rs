use std::collections::{HashMap, HashSet};

use super::{Bin, EXEC, Fields, Inst, Lower, NULL, Operand, Reg, V, Value, Why};

/// A lane mask's bit in a lane the console never launched (a partial final
/// group's, kept for its barriers), next to the host's, which has the lane
/// in EXEC on entry ([`Lower::launched`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Unlaunched {
    /// The host's: nothing it derives from differs (constants, user data,
    /// compares of data).
    Same,
    /// Both have this bit (a mask of no lanes or of every lane).
    Both(bool),
    /// The console has this bit, whatever the host's (EXEC on entry: 0).
    Console(bool),
}

impl Unlaunched {
    fn not(self) -> Self {
        match self {
            Self::Same => Self::Same,
            Self::Both(b) => Self::Both(!b),
            Self::Console(b) => Self::Console(!b),
        }
    }

    fn bit(self) -> Option<bool> {
        match self {
            Self::Both(b) | Self::Console(b) => Some(b),
            Self::Same => None,
        }
    }

    /// `s_<f>_b64 d, a, c` (`Structured`'s lane-mask meaning of `f`).
    fn combine(f: &str, a: Self, c: Self) -> Option<Self> {
        let (base, a, c, invert) = match f {
            "and" | "or" | "xor" => (f, a, c, false),
            "andn2" => ("and", a, c.not(), false),
            "orn2" => ("or", a, c.not(), false),
            "andn1" => ("and", a.not(), c, false),
            "orn1" => ("or", a.not(), c, false),
            "nand" => ("and", a, c, true),
            "nor" => ("or", a, c, true),
            "xnor" => ("xor", a, c, true),
            _ => return None,
        };
        let g = |x: bool, y: bool| match base {
            "and" => x & y,
            "or" => x | y,
            _ => x ^ y,
        };
        let r = match (a, c) {
            (Self::Both(x), Self::Both(y)) => Self::Both(g(x, y)),
            (Self::Same | Self::Both(_), Self::Same | Self::Both(_)) => Self::Same,
            _ => match (a.bit(), c.bit()) {
                (Some(x), Some(y)) => Self::Console(g(x, y)),
                (Some(false), _) | (_, Some(false)) if base == "and" => Self::Console(false),
                (Some(true), _) | (_, Some(true)) if base == "or" => Self::Console(true),
                _ => return None,
            },
        };
        Some(if invert { r.not() } else { r })
    }
}

impl Lower<'_> {
    /// The value of the high dword of the wave64 register pair whose low
    /// dword holds `lo` (none in wave32).
    pub(super) fn pair_high(&self, lo: Value) -> Option<Value> {
        if !self.x.native.wave64 {
            return None;
        }
        let p = self.x.p;
        let pick = |reg: Reg| match reg {
            Reg::S(c) => Some(Reg::S(c + 1)),
            _ => None,
        };
        if let Some(&(reg, _)) = p.inputs.iter().find(|i| i.1 == lo) {
            let h = pick(reg)?;
            return p.inputs.iter().find(|i| i.0 == h).map(|i| i.1);
        }
        if let Some(&(b, i)) = self.phi_at.get(&lo) {
            let h = pick(p.blocks[b].phis[i].reg)?;
            return p.blocks[b]
                .phis
                .iter()
                .find(|q| q.reg == h)
                .map(|q| q.value);
        }
        // Written with it by one instruction; else (two instructions, one
        // for each dword) the dword the mask's reader takes with it.
        let together = || {
            let b = *self.def_block.get(&lo)?;
            let inst = p.blocks[b]
                .insts
                .iter()
                .find(|i| i.writes.iter().any(|w| w.1 == lo))?;
            let &(reg, _) = inst.writes.iter().find(|w| w.1 == lo)?;
            inst.write(pick(reg)?)
        };
        together().or_else(|| self.x.st.pairs.get(&lo).copied())
    }

    /// [`Self::guard`] of a memory access: not by the threads a partial final
    /// group only keeps for its barriers ([`Self::partial_active`]).
    pub(super) fn memory_guard(&mut self, b: usize, i: usize) -> Result<Option<V>, Why> {
        let g = self.guard(b, i)?;
        Ok(self.launched_guard(g))
    }

    /// `guard` without the threads a partial final group keeps for its
    /// barriers ([`Self::partial_active`]).
    pub(super) fn launched_guard(&mut self, guard: Option<V>) -> Option<V> {
        match (guard, self.partial_active) {
            (Some(g), Some(a)) => Some(self.s.m.bin(self.cur, Bin::And, g, a)),
            (g, a) => g.or(a),
        }
    }

    /// `on` (the lanes of mask `mask`, at `at`) without the threads a partial
    /// final group keeps for its barriers, when the mask holds only EXEC's
    /// lanes (`Structured::active`): the console never launched them.
    // shortcut: a loop that flips EXEC (e.g. s_not_b64 exec) gets no verdict and takes the AND, so unlaunched lanes stay off where the console turns them on; give phis a fixed-point verdict if a title needs it.
    pub(super) fn launched(&mut self, at: usize, on: V, mask: Value) -> V {
        let Some(a) = self.partial_active else {
            return on;
        };
        if !self.x.st.active.contains(&mask) {
            return on;
        }
        match self.unlaunched_mask(mask) {
            // Set apart from EXEC on entry (`s_mov_b64 exec, -1`): the
            // host's bit is the console's.
            Some(Unlaunched::Same | Unlaunched::Both(true)) => on,
            // Its complement (`s_not_b64 exec, exec`): 1 there.
            Some(Unlaunched::Console(true)) => {
                let t = self.s.m.bool(true);
                let out = self.s.m.bin(at, Bin::Xor, a, t);
                self.s.m.bin(at, Bin::Or, on, out)
            }
            // Taken from EXEC (0 there), or not known: a mask of EXEC's lanes
            // has only lanes the host's EXEC has, so 0 is the console's bit
            // unless shown otherwise. Loops (a join whose edges come back
            // through itself, `busy`) are not known and land here.
            _ => self.s.m.bin(at, Bin::And, on, a),
        }
    }

    /// [`Self::unlaunched`] of the lane mask whose low dword is `lo`, with
    /// its wave64 high dword's; `None` where the dwords differ.
    fn unlaunched_mask(&self, lo: Value) -> Option<Unlaunched> {
        let (mut known, mut busy) = (HashMap::new(), HashSet::new());
        let u = self.unlaunched(lo, &mut known, &mut busy)?;
        match self.pair_high(lo) {
            Some(hi) => (self.unlaunched(hi, &mut known, &mut busy)? == u).then_some(u),
            None if self.x.p.wave64 => None,
            None => Some(u),
        }
    }

    /// [`Self::launched_guard`] of a shuffle's source lanes. Where EXEC at
    /// `inst` is set apart from EXEC on entry (a whole-wave section after
    /// `s_or_saveexec_b64 s, -1`), the console has those lanes active.
    pub(super) fn shuffle_guard(&mut self, inst: &Inst, guard: Option<V>) -> Option<V> {
        let Some(a) = self.partial_active else {
            return guard;
        };
        let exec = inst.read(Reg::S(EXEC)).and_then(|e| self.unlaunched_mask(e));
        match exec {
            Some(Unlaunched::Same | Unlaunched::Both(true)) => guard,
            Some(Unlaunched::Console(true)) => guard.map(|g| {
                let t = self.s.m.bool(true);
                let out = self.s.m.bin(self.cur, Bin::Xor, a, t);
                self.s.m.bin(self.cur, Bin::Or, g, out)
            }),
            _ => self.launched_guard(guard),
        }
    }

    /// A lane mask's bit, in a lane the console never launched, next to
    /// the host's (whose EXEC on entry has the lane); `None` where that is
    /// not known. Any value reached through a value in `busy` (a loop's
    /// back edge), a join included, is `None`.
    fn unlaunched(
        &self,
        v: Value,
        known: &mut HashMap<Value, Option<Unlaunched>>,
        busy: &mut HashSet<Value>,
    ) -> Option<Unlaunched> {
        if let Some(&u) = known.get(&v) {
            return u;
        }
        if !busy.insert(v) {
            return None;
        }
        let u = self.unlaunched_def(v, known, busy);
        busy.remove(&v);
        // A join cut off at a busy edge has no verdict here, not in general.
        let cut = self.def_block.get(&v).is_some_and(|&b| {
            self.x.p.blocks[b]
                .phis
                .iter()
                .any(|q| q.value == v && q.args.iter().any(|a| busy.contains(a)))
        });
        if !cut {
            known.insert(v, u);
        }
        u
    }

    fn unlaunched_def(
        &self,
        v: Value,
        known: &mut HashMap<Value, Option<Unlaunched>>,
        busy: &mut HashSet<Value>,
    ) -> Option<Unlaunched> {
        use Unlaunched::{Both, Console, Same};
        let p = self.x.p;
        if let Some(&r) = self.x.st.restored.get(&v) {
            return self.unlaunched(r, known, busy);
        }
        let Some(&b) = self.def_block.get(&v) else {
            // EXEC on entry has none of those lanes; user and system values
            // are the same.
            let &(reg, _) = p.inputs.iter().find(|i| i.1 == v)?;
            return Some(if reg == Reg::S(EXEC) || reg == Reg::S(EXEC + 1) {
                Console(false)
            } else {
                Same
            });
        };
        if let Some(phi) = p.blocks[b].phis.iter().find(|q| q.value == v) {
            let mut all = None;
            for &arg in &phi.args {
                if busy.contains(&arg) {
                    return None;
                }
                let u = self.unlaunched(arg, known, busy)?;
                if all.is_some_and(|a| a != u) {
                    return None;
                }
                all = Some(u);
            }
            return all;
        }
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == v))?;
        let name = inst.name();
        // A wave64 mask's high dword (odd register of its pair) reads the
        // high dwords of its sources.
        let hi = match inst.writes.iter().find(|w| w.1 == v)?.0 {
            Reg::S(r) => u8::from(p.wave64 && r & 1 == 1),
            _ => 0,
        };
        let exec = inst.read(Reg::S(EXEC + hi));
        match &inst.fields {
            // A compare's or carry's mask has EXEC's lanes only.
            Fields::Valu(x) if name.starts_with("v_cmp") || x.carry.is_some() => {
                let e = self.unlaunched(exec?, known, busy)?;
                Unlaunched::combine("and", Same, e)
            }
            Fields::Sop { src, .. } => {
                let wave64 = p.wave64;
                let width = if wave64 { "_b64" } else { "_b32" };
                let op = name.strip_prefix("s_")?.strip_suffix(width)?;
                let constant = |x: u64| {
                    let x = if wave64 { x } else { x & 0xFFFF_FFFF };
                    match x {
                        0 => Both(false),
                        _ if x == u64::MAX >> (32 * u8::from(!wave64)) => Both(true),
                        _ => Same,
                    }
                };
                let mut arg = |n: usize| -> Option<Unlaunched> {
                    match *src.get(n)? {
                        Operand::Scalar { code, .. } if code != NULL => {
                            self.unlaunched(inst.read(Reg::S(code + hi))?, known, busy)
                        }
                        Operand::Const(c @ 128..=208) => Some(constant(if c <= 192 {
                            u64::from(c) - 128
                        } else {
                            (192 - i64::from(c)) as u64
                        })),
                        Operand::Literal(x) => Some(constant(u64::from(x))),
                        _ => None,
                    }
                };
                let save = match (op.strip_suffix("_saveexec"), op.strip_suffix("_wrexec")) {
                    (Some(f), _) => Some((true, f)),
                    (_, Some(f)) => Some((false, f)),
                    _ => None,
                };
                if let Some((saveexec, f)) = save {
                    // `_saveexec`'s scalar destination gets the old EXEC.
                    let new = !saveexec || inst.write(Reg::S(EXEC + hi)) == Some(v);
                    let s = if new { Some(arg(0)?) } else { None };
                    let old = self.unlaunched(exec?, known, busy)?;
                    return match s {
                        Some(s) => Unlaunched::combine(f, s, old),
                        None => Some(old),
                    };
                }
                match op {
                    "mov" => arg(0),
                    "not" => Some(arg(0)?.not()),
                    "cselect" => {
                        let (a, c) = (arg(0)?, arg(1)?);
                        (a == c).then_some(a)
                    }
                    _ => {
                        let (a, c) = (arg(0)?, arg(1)?);
                        Unlaunched::combine(op, a, c)
                    }
                }
            }
            _ => None,
        }
    }
}
