//! Parameter interpolation instructions.

use super::*;

impl<'a> Lower<'a> {
    pub(super) fn interp(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        // A first half is lowered as the interpolation it is half of: the
        // completing half recomputes the whole value anyway, so a title that
        // reads the half early sees the finished one rather than nothing.
        let Some(&interp) = self.interps.get(&site).or_else(|| self.firsts.get(&site)) else {
            return Err("the first half of an interpolation");
        };
        let (attribute, mode) = match interp {
            Interp::Interpolated { unless, .. } if !self.avoids(site, unless) => {
                return Err("an interpolation with computed barycentrics");
            }
            Interp::Interpolated {
                attribute,
                barycentric,
                ..
            } => (
                attribute,
                match barycentric {
                    Barycentric::PerspectiveCenter => 2,
                    Barycentric::PerspectiveCentroid => 3,
                    Barycentric::PerspectiveSample => 6,
                    Barycentric::LinearCenter => 4,
                    Barycentric::LinearCentroid => 5,
                    Barycentric::LinearSample => 7,
                    Barycentric::PerspectivePullModel => return Err("pull-model interpolation"),
                },
            ),
            // P0, the provoking vertex: a flat (nointerpolation) input.
            Interp::Vertex {
                attribute,
                vertex: 2,
            } => (attribute, 1),
            Interp::Vertex { .. } => return Err("a non-provoking vertex value"),
        };
        let dst = match &inst.fields {
            Fields::Vintrp { dst, .. } => *dst,
            Fields::Valu(v) => v.dst.ok_or("an interpolation")?,
            _ => unreachable!(),
        };
        let Operand::Vector { reg, .. } = dst else {
            return Err("an interpolation");
        };
        // A channel of an input element, interpolated as `mode` says.
        let channel = |l: &mut Self, row: u8, ch: u8| -> Result<V, Why> {
            let (id, col) = l
                .find_input(row, ch)
                .ok_or("an input the signature lacks")?;
            let e = &mut l.s.inputs[id];
            if e.interp != mode && e.usage != 0 {
                return Err("one input interpolated two ways");
            }
            e.interp = mode;
            match l.load_any_input(id, col) {
                Val::I32(x) => Ok(x),
                _ => Err("an input"),
            }
        };
        let packing = match &self.x.native.io {
            StageIo::Ps { inputs, .. } => inputs
                .iter()
                .find(|i| i.row == attribute.row && i.fp16)
                .map(|i| i.pack2),
            _ => None,
        };
        let f16 = inst.name().ends_with("_f16");
        // VINTRP/VINTERP obey EXEC: inactive lanes keep the destination.
        // shortcut: p1 writes the whole interpolation, so a lane whose EXEC
        // differs between p1 and p2 does not get the console's partial value.
        // An EXEC the pass cannot give leaves the write unguarded, not the
        // program untranslated.
        let mut guard = self.guard(site.0, site.1).ok().flatten();
        // A packed first half has no host value, so a destination holding one
        // has nothing correct to keep.
        // shortcut: inactive lanes take the result there; guard it once first
        // halves pass the old value through (lower.rs).
        if let Interp::Interpolated { first, .. } = interp
            && f16
            && first.iter().any(|&(b, i)| {
                self.x.p.blocks[b].insts[i].write(Reg::V(reg)).is_some()
            })
        {
            guard = None;
        }
        let x = match packing {
            None if f16 || attribute.high => return Err("a 16-bit read of a 32-bit input"),
            None => channel(self, attribute.row, attribute.channel)?,
            Some(pack2) => {
                // 16-bit values: one in the low half of each dword, or two
                // per dword (`packed_half`); the host's are 32-bit floats.
                let half = |l: &mut Self, h: u8| -> Result<Option<V>, Why> {
                    let src = if pack2 {
                        packed_half(&l.s.inputs, attribute.row, h)
                    } else {
                        h.is_multiple_of(2).then_some((attribute.row, h / 2))
                    };
                    let Some((row, ch)) = src else {
                        return Ok(None);
                    };
                    let x = channel(l, row, ch)?;
                    let f = l.as_float(x);
                    Ok(Some(l.narrow16(f)))
                };
                let h = 2 * attribute.channel;
                let c = self.cur;
                if f16 {
                    let lo = half(self, h + u8::from(attribute.high))?
                        .ok_or("a packed input half no element holds")?;
                    // omod and clamp are part of the f16 instruction; op_sel[3]
                    // picks the destination half.
                    let lo = self.half_modifiers(inst, lo);
                    return self.write_half(inst, lo, guard);
                } else if mode == 1 {
                    // The whole dword of a flat input: both halves.
                    let z = self.s.m.i32(0);
                    let lo = half(self, h)?.unwrap_or(z);
                    let hi = half(self, h + 1)?.unwrap_or(z);
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                } else {
                    return Err("32-bit interpolation of a 16-bit input");
                }
            }
        };
        self.write(inst, Reg::V(reg), x, guard);
        Ok(())
    }
}
