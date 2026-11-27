use super::flags;
use super::{Attr, Bin, EXEC, LANE_MASK, Lower, Reg, Term, V, Val, Value, Why, pred};

impl Lower<'_> {
    /// Console flow: whether block `b`'s `s_cbranch_*` is taken, the same
    /// in both halves (wave-wide mask tests through [`Self::any_lane`]).
    pub(super) fn console_branch(&mut self, b: usize) -> Result<V, Why> {
        let inst = self.x.p.blocks[b].insts.last().ok_or("a branch")?;
        let name = inst.name();
        let c = self.cur;
        let (any, set) = match name {
            "s_cbranch_scc0" | "s_cbranch_scc1" => {
                let v = inst.read(Reg::Scc).ok_or("a branch's SCC")?;
                let scc = self.scc_bit(v)?;
                (scc, name == "s_cbranch_scc1")
            }
            "s_cbranch_vccz" | "s_cbranch_vccnz" | "s_cbranch_execz" | "s_cbranch_execnz" => {
                let reg = if name.contains("vcc") {
                    crate::ir::VCC
                } else {
                    EXEC
                };
                let lo = inst.read(Reg::S(reg)).ok_or("a branch's mask")?;
                if !self.x.st.masks.contains_key(&lo) {
                    return Err("a branch on an untracked mask");
                }
                (self.any_lane(lo)?, name.ends_with("nz"))
            }
            _ => return Err("a branch the console flow does not model"),
        };
        Ok(if set {
            any
        } else {
            let t = self.s.m.bool(true);
            self.s.m.bin(c, Bin::Xor, any, t)
        })
    }

    /// Console flow: an SCC value as a bit. A mask operation's SCC (result
    /// not zero) is whether some lane of the console wave has its bit.
    pub(super) fn scc_bit(&mut self, v: Value) -> Result<V, Why> {
        match self.vals.get(&v) {
            Some(&Val::Bit(x)) => return Ok(x),
            Some(&Val::Missing(why)) if why != LANE_MASK => return Err(why),
            _ => {}
        }
        let b = *self
            .def_block
            .get(&v)
            .ok_or("an SCC from the program's start")?;
        let inst = self.x.p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == v))
            .ok_or("an SCC without its instruction")?;
        // SCC tests the new EXEC. `s_*_saveexec` also writes a scalar, the
        // EXEC it replaces, and that write comes first (`s_*_wrexec` writes
        // the new EXEC to both).
        let masks = &self.x.st.masks;
        let mask = inst
            .write(Reg::S(EXEC))
            .filter(|w| masks.contains_key(w))
            .or_else(|| {
                inst.writes
                    .iter()
                    .filter(|w| matches!(w.0, Reg::S(_)))
                    .map(|w| w.1)
                    .find(|w| masks.contains_key(w))
            })
            .ok_or("an SCC the emitter did not compute")?;
        self.any_lane(mask)
    }

    /// Console flow: whether mask `lo` holds in some lane of the console wave.
    pub(super) fn any_lane(&mut self, lo: Value) -> Result<V, Why> {
        let c = self.cur;
        let m = self.x.st.masks[&lo];
        let bit = self.boolean(m, c)?;
        let bit = self.launched(c, bit, lo);
        let (lo, hi) = self.console_ballot(c, bit);
        let or = self.s.m.bin(c, Bin::Or, lo, hi);
        let z = self.s.m.i32(0);
        Ok(self.s.m.cmp(c, pred::NE, or, z))
    }

    /// Console flow: the console wave's ballot of `bit`, low and high word.
    pub(super) fn console_ballot(&mut self, at: usize, bit: V) -> (V, V) {
        if self.split {
            let word = self.ballot_word(at, bit);
            return self.exchange(at, word);
        }
        // One host wave is the console wave (the high word is 0 in waves
        // of 32).
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        (self.s.m.extract(at, r, 0), self.s.m.extract(at, r, 1))
    }

    /// The host wave's ballot of `bit` (32 lanes: the first word).
    pub(super) fn ballot_word(&mut self, at: usize, bit: V) -> V {
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        self.s.m.extract(at, r, 0)
    }

    /// The console thread's flat id in the group (see [`Self::thread_id`]).
    pub(super) fn console_flat(&mut self) -> V {
        if let Some(f) = self.console_flat {
            return f;
        }
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let wave = self.s.wave_size.expect("set with console lanes");
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        let lane = self.s.op(
            e,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        );
        let f = self.s.m.bin(e, Bin::Add, base, lane);
        self.s.flags |= flags::WAVE_OPS;
        self.console_flat = Some(f);
        f
    }

    /// This lane's console lane (0..63, 0..31 in waves of 32) and its
    /// half of a wave64 (0, 1).
    pub(super) fn console_lane(&mut self) -> (V, V) {
        let f = self.console_flat();
        let e = self.s.entry;
        let (mask, k5) = (self.s.m.i32(self.wave() - 1), self.s.m.i32(5));
        let lane = self.s.m.bin(e, Bin::And, f, mask);
        let half = self.s.m.bin(e, Bin::LShr, lane, k5);
        (lane, half)
    }

    /// Console flow: each half gives `x` (the same in all its lanes) and
    /// both get (the lower half's, the upper half's): through two group-
    /// shared dwords between group barriers, which every thread of the
    /// group reaches (see [`CONSOLE_FLOW`]). A group of 32 threads or
    /// fewer has no upper half: its lanes count as inactive there (0).
    pub(super) fn exchange(&mut self, at: usize, x: V) -> (V, V) {
        let threads: u32 = self.s.threads.expect("compute").iter().product();
        let z = self.s.m.i32(0);
        if threads <= 32 {
            return (x, z);
        }
        if self.multi {
            // Each half its own slot; the halves of a console wave reach
            // their exchanges at the same group barrier count.
            let f = self.console_flat();
            let k5 = self.s.m.i32(5);
            let h = self.s.m.bin(at, Bin::LShr, f, k5);
            let slots = match self.xchg {
                Some(g) => g,
                None => {
                    let arr = self.s.m.array(threads / 32, self.s.t.i32);
                    let g = self.s.m.global("xchg", arr, 3);
                    self.xchg = Some(g);
                    g
                }
            };
            let mine = self.s.m.element(at, slots, h);
            self.s.m.store(at, mine, x);
            self.hw_barrier(at);
            let (one, not1) = (self.s.m.i32(1), self.s.m.i32(!1));
            let (l, u) = (
                self.s.m.bin(at, Bin::And, h, not1),
                self.s.m.bin(at, Bin::Or, h, one),
            );
            let (pl, pu) = (self.s.m.element(at, slots, l), self.s.m.element(at, slots, u));
            let (lo, hi) = (self.s.m.load(at, pl), self.s.m.load(at, pu));
            self.hw_barrier(at);
            return (lo, hi);
        }
        let (_, half) = self.console_lane();
        let slots = match self.xchg {
            Some(g) => g,
            None => {
                let arr = self.s.m.array(2, self.s.t.i32);
                let g = self.s.m.global("xchg", arr, 3);
                self.xchg = Some(g);
                g
            }
        };
        let void = self.s.t.void;
        // Thread-group sync with a group-shared memory fence.
        let mode = self.s.m.i32(1 | 8);
        let mine = self.s.m.element(at, slots, half);
        self.s.m.store(at, mine, x);
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        let one = self.s.m.i32(1);
        let p0 = self.s.m.element(at, slots, z);
        let p1 = self.s.m.element(at, slots, one);
        let lo = self.s.m.load(at, p0);
        let hi = self.s.m.load(at, p1);
        // Both read before the next exchange writes.
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        (lo, hi)
    }

    /// Multi-wave console flow: the group-shared state (`phase`: group
    /// barriers each thread has passed; `arrived`: per half, the count of
    /// the program's barriers it reached (mod 4, top 2 bits) and the phase
    /// stamp it reached the last one at; `finished`: per half, the phase
    /// stamp it ended at, 0 while running). Made and cleared at the entry.
    ///
    /// Why: the console waves of a group branch differently, so they reach
    /// different group barriers (their exchanges); barriers pair up by
    /// count, not place. What must agree is decided from stamps: a stamp
    /// `s` written before barrier `s` counts once a thread has passed `s`,
    /// so every thread deciding after the same barrier sees the same thing.
    pub(super) fn multi_state(&mut self) -> (V, V, V) {
        if let Some(s) = self.multi_sync {
            return s;
        }
        let threads: u32 = self.s.threads.expect("compute").iter().product();
        let i32t = self.s.t.i32;
        let (per_thread, per_half) = (
            self.s.m.array(threads, i32t),
            self.s.m.array(threads / 32, i32t),
        );
        let ph = self.s.m.global("phase", per_thread, 3);
        let arr = self.s.m.global("arrived", per_half, 3);
        let fin = self.s.m.global("finished", per_half, 3);
        self.multi_sync = Some((ph, arr, fin));
        let e = self.s.entry;
        let f = self.console_flat();
        let (k5, z) = (self.s.m.i32(5), self.s.m.i32(0));
        let h = self.s.m.bin(e, Bin::LShr, f, k5);
        for (g, i) in [(ph, f), (arr, h), (fin, h)] {
            let p = self.s.m.element(e, g, i);
            self.s.m.store(e, p, z);
        }
        self.hw_barrier(e);
        (ph, arr, fin)
    }

    /// A group barrier (LDS fence, UAV fence when the program writes UAVs)
    /// counted in this thread's phase; the new phase.
    pub(super) fn hw_barrier(&mut self, at: usize) -> V {
        let (ph, _, _) = self.multi_state();
        let void = self.s.t.void;
        let mode = self.s.m.i32(1 | 8 | if self.writes_uavs { 2 } else { 0 });
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        let f = self.console_flat();
        let p = self.s.m.element(at, ph, f);
        let one = self.s.m.i32(1);
        let old = self.s.m.load(at, p);
        let new = self.s.m.bin(at, Bin::Add, old, one);
        self.s.m.store(at, p, new);
        new
    }

    /// This thread's phase and half.
    pub(super) fn phase_and_half(&mut self, at: usize) -> (V, V) {
        let (ph, _, _) = self.multi_state();
        let f = self.console_flat();
        let p = self.s.m.element(at, ph, f);
        let q = self.s.m.load(at, p);
        let k5 = self.s.m.i32(5);
        (q, self.s.m.bin(at, Bin::LShr, f, k5))
    }

    /// Loops from `self.cur` over group barriers until `ok(phase, half)`
    /// holds for every half of the group (host lane `i` tests half `i`);
    /// `self.cur` is then the block after.
    pub(super) fn wait_for_halves(&mut self, ok: impl Fn(&mut Self, usize, V, V) -> V) {
        let threads: u32 = self.s.threads.expect("compute").iter().product();
        let (wait, after) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(self.cur, Term::Br(wait));
        let q = self.hw_barrier(wait);
        let i32t = self.s.t.i32;
        let lane = self.s.op(wait, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let (n, z) = (self.s.m.i32(threads / 32), self.s.m.i32(0));
        let inside = self.s.m.cmp(wait, pred::IULT, lane, n);
        let i = self.s.m.select(wait, inside, lane, z);
        let good = ok(self, wait, q, i);
        let outside = self.s.m.cmp(wait, pred::IUGE, lane, n);
        let fine = self.s.m.bin(wait, Bin::Or, outside, good);
        let i1 = self.s.t.i1;
        let all = self.s.op(wait, "waveAllTrue", Attr::NoUnwind, i1, 114, &[fine]);
        self.s.flags |= flags::WAVE_OPS;
        self.s.m.terminate(wait, Term::CondBr(all, after, wait));
        self.cur = after;
    }

    /// Whether half `i` has ended by phase `q`.
    pub(super) fn ended(&mut self, at: usize, q: V, i: V) -> V {
        let (_, _, fin) = self.multi_state();
        let p = self.s.m.element(at, fin, i);
        let s = self.s.m.load(at, p);
        let z = self.s.m.i32(0);
        let set = self.s.m.cmp(at, pred::NE, s, z);
        let before = self.s.m.cmp(at, pred::IULE, s, q);
        self.s.m.bin(at, Bin::And, set, before)
    }

    /// The program's `s_barrier` with several console waves in the group:
    /// this half arrives (count + 1, stamped), then waits until every half
    /// has arrived at this count (or the next) or ended.
    pub(super) fn soft_barrier(&mut self) {
        let (_, arr, _) = self.multi_state();
        let c = self.cur;
        let (q, h) = self.phase_and_half(c);
        let p = self.s.m.element(c, arr, h);
        let e = self.s.m.load(c, p);
        let (k30, one, three, mask) = (
            self.s.m.i32(30),
            self.s.m.i32(1),
            self.s.m.i32(3),
            self.s.m.i32(0x3fff_ffff),
        );
        let count = self.s.m.bin(c, Bin::LShr, e, k30);
        let count = self.s.m.bin(c, Bin::Add, count, one);
        let count = self.s.m.bin(c, Bin::And, count, three);
        let stamp = self.s.m.bin(c, Bin::Add, q, one);
        let top = self.s.m.bin(c, Bin::Shl, count, k30);
        let v = self.s.m.bin(c, Bin::Or, top, stamp);
        self.s.m.store(c, p, v);
        let next = self.s.m.bin(c, Bin::Add, count, one);
        let next = self.s.m.bin(c, Bin::And, next, three);
        self.wait_for_halves(move |l, at, q, i| {
            let p = l.s.m.element(at, arr, i);
            let e = l.s.m.load(at, p);
            let ci = l.s.m.bin(at, Bin::LShr, e, k30);
            let si = l.s.m.bin(at, Bin::And, e, mask);
            let same = l.s.m.cmp(at, pred::EQ, ci, count);
            let seen = l.s.m.cmp(at, pred::IULE, si, q);
            let here = l.s.m.bin(at, Bin::And, same, seen);
            let past = l.s.m.cmp(at, pred::EQ, ci, next);
            let ended = l.ended(at, q, i);
            let a = l.s.m.bin(at, Bin::Or, here, past);
            l.s.m.bin(at, Bin::Or, a, ended)
        });
    }

    /// The end of a thread with several console waves in the group: this
    /// half is stamped ended and keeps passing group barriers until every
    /// half has ended (the others' exchanges and barriers need it there).
    pub(super) fn drain(&mut self) {
        let (_, _, fin) = self.multi_state();
        let c = self.cur;
        let (q, h) = self.phase_and_half(c);
        let one = self.s.m.i32(1);
        let stamp = self.s.m.bin(c, Bin::Add, q, one);
        let p = self.s.m.element(c, fin, h);
        self.s.m.store(c, p, stamp);
        self.wait_for_halves(|l, at, q, i| l.ended(at, q, i));
    }
}
