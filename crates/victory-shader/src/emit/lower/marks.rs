//! Which lanes of a wave set a write or read mark (module notes of `emit`,
//! "Write marks", "Read marks").

use super::super::bitcode::{Attr, Bin, V, pred};
use super::super::shader::flags;
use super::Lower;

impl Lower<'_> {
    /// Whether this lane sets mark bit `n` (its load and atomic go to the
    /// bitmap in system memory): one lane per distinct bit in the wave, the
    /// lowest of the lanes `WaveMatch` groups with it. Most lanes of a wave
    /// mark the same page, so one load crosses the bus instead of 32. One
    /// wave op decides it, so lanes cannot see different active sets. In
    /// pixel shaders helpers match only each other (bit 31 of the key; bits
    /// stay below it): a helper's writes are dropped, so a non-helper lane
    /// leads every bit the shader really marks.
    pub(super) fn mark_lead(&mut self, n: V) -> Option<V> {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let (i1, i32t) = (self.s.t.i1, self.s.t.i32);
        let lane = self
            .s
            .op(c, "waveGetLaneIndex", Attr::ReadNone, i32t, 111, &[]);
        let key = if self.s.pixel {
            let helper = self
                .s
                .op(c, "isHelperLane.i1", Attr::ReadOnly, i1, 221, &[]);
            let (top, zero) = (self.s.m.i32(1 << 31), self.s.m.i32(0));
            let tag = self.s.m.select(c, helper, top, zero);
            self.s.m.bin(c, Bin::Or, n, tag)
        } else {
            n
        };
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let m = self
            .s
            .op(c, "waveMatch.i32", Attr::NoUnwind, four, 165, &[key]);
        let (lo, hi) = (self.s.m.extract(c, m, 0), self.s.m.extract(c, m, 1));
        // The lowest lane of the match: in `lo` (lanes 0-31) or `hi` + 32.
        let flo = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[lo]);
        let fhi = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[hi]);
        let (zero, k32) = (self.s.m.i32(0), self.s.m.i32(32));
        let fhi = self.s.m.bin(c, Bin::Add, fhi, k32);
        let some_lo = self.s.m.cmp(c, pred::NE, lo, zero);
        let first = self.s.m.select(c, some_lo, flo, fhi);
        Some(self.s.m.cmp(c, pred::EQ, first, lane))
    }
}
