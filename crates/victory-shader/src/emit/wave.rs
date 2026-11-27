//! Whole-wave sections (decision 0003): code between `s_or_saveexec -1`
//! (or `s_nand_saveexec 0`) and the restore of the saved mask runs in every lane of the console's
//! wave, the inactive ones included, which is how the compiler expands
//! HLSL wave reductions and prefix sums (an identity put into the inactive
//! lanes with `v_cndmask`, then DPP, `v_permlanex16` and `v_readlane`
//! steps). A host lane that is elsewhere cannot run that code, so each
//! section is simulated on the console's lanes with symbolic values, and a
//! value it leaves is lowered by what it turned out to be: the reduction or
//! prefix of one operation over the lanes' (masked) inputs, which DXIL's
//! wave operations compute over the host's active lanes. No instruction
//! sequence is matched; any step the simulation does not model leaves the
//! values it writes unknown.

use std::collections::HashMap;

use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::ir::{EXEC, Inst, Program, Reg, Value};

/// An associative operation and its identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
    AddU,
    AddF,
    MinU,
    MaxU,
    MinI,
    MaxI,
    MinF,
    MaxF,
    Or,
    And,
    Xor,
    /// `v_add_f16`: on the low halves (see [`Sym::Half`]).
    AddH,
}

impl Op {
    fn of(name: &str) -> Option<Op> {
        Some(match name {
            // The low dwords of a 64-bit sum (the carry is not followed).
            "v_add_nc_u32" | "v_add_co_u32" => Op::AddU,
            "v_add_f32" => Op::AddF,
            "v_min_u32" => Op::MinU,
            "v_max_u32" => Op::MaxU,
            "v_min_i32" => Op::MinI,
            "v_max_i32" => Op::MaxI,
            "v_min_f32" => Op::MinF,
            "v_max_f32" => Op::MaxF,
            "v_or_b32" => Op::Or,
            "v_and_b32" => Op::And,
            "v_xor_b32" => Op::Xor,
            _ => return None,
        })
    }

    /// Whether `k` leaves the other operand unchanged.
    fn identity(self, k: u32) -> bool {
        match self {
            Op::AddU | Op::Or | Op::Xor | Op::MaxU => k == 0,
            Op::AddF => k == 0 || k == 0x8000_0000,
            Op::MinU | Op::And => k == u32::MAX,
            Op::MinI => k == 0x7FFF_FFFF,
            Op::MaxI => k == 0x8000_0000,
            Op::MinF => k == 0x7F80_0000,
            Op::MaxF => k == 0xFF80_0000,
            Op::AddH => k & 0x7FFF == 0,
        }
    }

    /// Counting a lane twice changes the result.
    fn counts(self) -> bool {
        matches!(self, Op::AddU | Op::AddF | Op::Xor | Op::AddH)
    }
}

/// A lane's value inside a section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sym {
    Unknown,
    K(u32),
    /// The lane's own value of a VGPR value from before the section.
    Own(Value),
    /// A uniform value from before the section.
    Uni(Value),
    /// `op` over `lanes` of `mask ? src : ident` (`op` none: one lane).
    Red {
        op: Option<Op>,
        src: Value,
        mask: Value,
        ident: u32,
        lanes: u64,
    },
    /// A 16-bit operation's result: the low half is `op` over `lanes` of
    /// the low halves of `mask ? src : ident`; the high half is that of
    /// lane `hi`'s `mask ? src : ident` (16-bit results keep the
    /// register's high half).
    Half {
        op: Op,
        src: Value,
        mask: Value,
        ident: u32,
        lanes: u64,
        hi: u32,
    },
    /// One dword (`high`: the upper) of a 64-bit float sum over `lanes` of
    /// the doubles `mask ? hi:lo : 0.0`.
    Wide {
        lo: Value,
        hi: Value,
        mask: Value,
        lanes: u64,
        high: bool,
    },
}

/// What a value a section leaves holds in every lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shape {
    K(u32),
    Value(Value),
    /// The lane's own `mask ? src : ident`.
    Masked(Value, Value, u32),
    /// `op` over every lane.
    Active(Op, Value, Value, u32),
    /// `op` over the lanes below (`inclusive`: and including) each lane.
    Prefix(Op, Value, Value, u32, bool),
    /// [`Sym::Half`] over every lane: `op`, src, mask, identity, the lane
    /// giving the high half.
    Half(Op, Value, Value, u32, u32),
    /// [`Sym::Wide`] over every lane: low and high source dwords, mask, and
    /// whether this is the upper dword.
    Wide(Value, Value, Value, bool),
}

/// The section starting at instruction `start` of a block: its restore
/// index and the shapes of the values it writes.
pub(crate) struct Section {
    pub end: usize,
    pub shapes: HashMap<Value, Shape>,
    /// The instruction that ended the simulation before the section ended
    /// (a lane mask it does not follow), for the error a value read after
    /// it gets.
    pub ended: Option<&'static str>,
    /// The instruction that last wrote each value the simulation left
    /// without a shape, for the error a read of it gets: the name says what
    /// kind of step is not modelled (a compare, a lane value, a select).
    pub reasons: HashMap<Value, &'static str>,
}

/// The reason a value a whole-wave section cannot shape is refused with: the
/// step that stopped the simulation, with the category kept, so a report does
/// not read as an unimplemented instruction of the same name. One string per
/// instruction name, made once.
fn section_step(name: &'static str) -> &'static str {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static NAMES: OnceLock<Mutex<HashMap<&'static str, &'static str>>> = OnceLock::new();
    let mut names = NAMES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    names
        .entry(name)
        .or_insert_with(|| Box::leak(format!("a whole-wave section step: {name}").into_boxed_str()))
}

/// Every whole-wave section of `block`, by start index.
pub(crate) fn sections(p: &Program, block: usize) -> HashMap<usize, Section> {
    let insts = &p.blocks[block].insts;
    let wave: u32 = if p.wave64 { 64 } else { 32 };
    let mut out = HashMap::new();
    let mut i = 0;
    while i < insts.len() {
        let inst = &insts[i];
        // EXEC becomes every lane: `s_or_saveexec -1` or `s_nand_saveexec 0`.
        let all = match inst.name() {
            "s_or_saveexec_b64" | "s_or_saveexec_b32" => Some(Operand::Const(193)),
            "s_nand_saveexec_b64" | "s_nand_saveexec_b32" => Some(Operand::Const(128)),
            _ => None,
        };
        let start = all.is_some_and(
            |k| matches!(&inst.fields, Fields::Sop { src, .. } if src.first() == Some(&k)),
        );
        let saved = match &inst.fields {
            Fields::Sop {
                dst: Some(Operand::Scalar { code, .. }),
                ..
            } if start => inst.write(Reg::S(*code)),
            _ => None,
        };
        let Some(saved) = saved else {
            i += 1;
            continue;
        };
        let end = (i + 1..insts.len()).find(|&j| {
            insts[j].write(Reg::S(EXEC)).is_some()
                && insts[j].read(Reg::S(saved_code(inst))) == Some(saved)
        });
        let Some(end) = end else {
            i += 1;
            continue;
        };
        let (shapes, ended) = simulate(&insts[i + 1..end], wave, saved);
        // What wrote each value the simulation could not shape: the last
        // such instruction is the one a reader has to teach it.
        let mut reasons = HashMap::new();
        for inst in &insts[i + 1..end] {
            for &(_, w) in &inst.writes {
                if !shapes.contains_key(&w) {
                    reasons.insert(w, section_step(inst.name()));
                }
            }
        }
        out.insert(i, Section { end, shapes, ended, reasons });
        i = end + 1;
    }
    out
}

fn saved_code(inst: &Inst) -> u8 {
    match &inst.fields {
        Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            ..
        } => *code,
        _ => NULL,
    }
}

/// The lane `l` reads with DPP, whether it reads 0 instead, and whether it
/// writes; `None` for patterns not modelled.
fn dpp_lane(d: Dpp, l: u32) -> Option<(u32, bool, bool)> {
    match d {
        Dpp::Lanes { sel, .. } => Some(((l & !7) | (sel >> (3 * (l & 7)) & 7), false, true)),
        Dpp::Row {
            ctrl,
            row_mask,
            bank_mask,
            bound_ctrl,
            ..
        } => {
            let (row, pos, n) = (l & !15, l & 15, u32::from(ctrl & 15));
            let enabled = row_mask >> (l >> 4 & 3) & 1 != 0 && bank_mask >> (l >> 2 & 3) & 1 != 0;
            let (src, inside) = match ctrl {
                0..=0xFF => ((l & !3) | (u32::from(ctrl) >> (2 * (l & 3)) & 3), true),
                0x101..=0x10F => (l + n, pos + n < 16),
                0x111..=0x11F => (l.wrapping_sub(n), pos >= n),
                0x121..=0x12F => (row | (pos.wrapping_sub(n) & 15), true),
                0x140 => (l ^ 15, true),
                0x141 => (l ^ 7, true),
                0x150..=0x15F => (row | n, true),
                0x160..=0x16F => (l ^ n, true),
                _ => return None,
            };
            match (inside, bound_ctrl) {
                (true, _) => Some((src, false, enabled)),
                (false, true) => Some((l, true, enabled)),
                (false, false) => Some((l, false, false)),
            }
        }
    }
}

fn combine(op: Op, a: Sym, b: Sym) -> Sym {
    match (a, b) {
        (Sym::K(k), x) | (x, Sym::K(k)) if op.identity(k) => x,
        (
            Sym::Red {
                op: oa,
                src,
                mask,
                ident,
                lanes: la,
            },
            Sym::Red {
                op: ob,
                src: sb,
                mask: mb,
                ident: ib,
                lanes: lb,
            },
        ) if (src, mask, ident) == (sb, mb, ib)
            && op.identity(ident)
            && oa.is_none_or(|o| o == op)
            && ob.is_none_or(|o| o == op)
            && (!op.counts() || la & lb == 0) =>
        {
            Sym::Red {
                op: Some(op),
                src,
                mask,
                ident,
                lanes: la | lb,
            }
        }
        _ => Sym::Unknown,
    }
}

/// A lane of 16-bit `op` on the low halves of `a` and `b`, the high half
/// kept from the destination's `old` value.
fn half(op: Op, a: Sym, b: Option<Sym>, old: Sym) -> Sym {
    // A low half: the identity, or `op` over lanes of one masked source.
    let low = |s: Sym| match s {
        Sym::K(k) if op.identity(k) => Some(None),
        Sym::Red {
            op: None,
            src,
            mask,
            ident,
            lanes,
        } if op.identity(ident) => Some(Some((src, mask, ident, lanes))),
        Sym::Half {
            op: o,
            src,
            mask,
            ident,
            lanes,
            ..
        } if o == op => Some(Some((src, mask, ident, lanes))),
        _ => None,
    };
    let (src, mask, ident, hi) = match old {
        Sym::Red {
            op: None,
            src,
            mask,
            ident,
            lanes,
        } if lanes.is_power_of_two() => (src, mask, ident, lanes.trailing_zeros()),
        Sym::Half {
            src,
            mask,
            ident,
            hi,
            ..
        } => (src, mask, ident, hi),
        _ => return Sym::Unknown,
    };
    let (Some(x), Some(y)) = (low(a), b.and_then(low)) else {
        return Sym::Unknown;
    };
    let mut lanes = 0;
    for (s, m, i, l) in [x, y].into_iter().flatten() {
        if (s, m, i) != (src, mask, ident) || lanes & l != 0 {
            return Sym::Unknown;
        }
        lanes |= l;
    }
    if lanes == 0 {
        return Sym::Unknown;
    }
    Sym::Half {
        op,
        src,
        mask,
        ident,
        lanes,
        hi,
    }
}

/// A lane's `v_add_f64` of the dword pairs `a` and `b`.
fn add_f64(a: (Sym, Sym), b: (Sym, Sym)) -> (Sym, Sym) {
    // A pair's term: nothing (+0.0), or (low source, high source, mask, lanes).
    let term = |p: (Sym, Sym)| match p {
        (Sym::K(0), Sym::K(0)) => Some(None),
        (
            Sym::Red {
                op: None,
                src: lo,
                mask,
                ident: 0,
                lanes,
            },
            Sym::Red {
                op: None,
                src: hi,
                mask: m,
                ident: 0,
                lanes: l,
            },
        ) if (mask, lanes) == (m, l) => Some(Some((lo, hi, mask, lanes))),
        (
            Sym::Wide {
                lo,
                hi,
                mask,
                lanes,
                high: false,
            },
            upper,
        ) if upper
            == (Sym::Wide {
                lo,
                hi,
                mask,
                lanes,
                high: true,
            }) =>
        {
            Some(Some((lo, hi, mask, lanes)))
        }
        _ => None,
    };
    let unknown = (Sym::Unknown, Sym::Unknown);
    let (Some(x), Some(y)) = (term(a), term(b)) else {
        return unknown;
    };
    let (lo, hi, mask, lanes) = match (x, y) {
        (None, None) => return (Sym::K(0), Sym::K(0)),
        // A sum with +0.0 is not followed: it would turn a -0.0 into +0.0.
        (Some(x), Some(y)) if (x.0, x.1, x.2) == (y.0, y.1, y.2) && x.3 & y.3 == 0 => {
            (x.0, x.1, x.2, x.3 | y.3)
        }
        _ => return unknown,
    };
    let half = |high| Sym::Wide {
        lo,
        hi,
        mask,
        lanes,
        high,
    };
    (half(false), half(true))
}

fn simulate(
    insts: &[Inst],
    wave: u32,
    saved: Value,
) -> (HashMap<Value, Shape>, Option<&'static str>) {
    let all = if wave == 64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    let mut exec = all;
    let mut vec: HashMap<Value, Vec<Sym>> = HashMap::new();
    let mut uni: HashMap<Value, Sym> = HashMap::new();
    // Masks the section derives from the saved EXEC (`s_and` with it).
    let mut masked = vec![saved];
    // Scalars the section set to a constant, 32 bits each (a 64-bit value
    // is two of them), which `EXEC` may be set from.
    let mut sconst: HashMap<Value, u32> = HashMap::new();
    for inst in insts {
        let name = inst.name();
        if inst.write(Reg::S(EXEC)).is_some() {
            // Constant lane sets (prefix sums); anything else ends what
            // the simulation can follow.
            let lo = name
                .starts_with("s_mov_b")
                .then(|| constant(inst, &sconst))
                .flatten();
            let Some(k) = lo else {
                return (shapes(vec, uni, wave), Some(section_step(name)));
            };
            exec = (if name == "s_mov_b64" {
                k
            } else {
                (exec & !0xFFFF_FFFF) | (k & 0xFFFF_FFFF)
            }) & all;
            continue;
        }
        if inst.write(Reg::S(EXEC + 1)).is_some() {
            let Some(k) = name
                .starts_with("s_mov_b")
                .then(|| constant(inst, &sconst))
                .flatten()
            else {
                return (shapes(vec, uni, wave), Some(section_step(name)));
            };
            exec = ((exec & 0xFFFF_FFFF) | ((k & 0xFFFF_FFFF) << 32)) & all;
            continue;
        }
        if let Fields::Sop { src, dst, .. } = &inst.fields {
            if name.starts_with("s_and_b")
                && src.iter().any(|o| matches!(*o, Operand::Scalar { code, .. } if inst.read(Reg::S(code)).is_some_and(|v| masked.contains(&v))))
                && let Some(&(_, w)) = inst.writes.iter().find(|w| matches!(w.0, Reg::S(_)))
            {
                masked.push(w);
            }
            let base = match dst {
                Some(Operand::Scalar { code, .. }) => *code,
                _ => NULL,
            };
            // `s_bcnt1` of EXEC counts the lanes the section has on: a
            // constant, as is SCC (whether any is).
            let count = (matches!(name, "s_bcnt1_i32_b64" | "s_bcnt1_i32_b32")
                && matches!(src.first(), Some(Operand::Scalar { code: EXEC, .. })))
            .then(|| {
                let on = if name == "s_bcnt1_i32_b64" {
                    exec
                } else {
                    exec & 0xFFFF_FFFF
                };
                on.count_ones()
            });
            for &(r, w) in &inst.writes {
                if let Some(n) = count {
                    let k = if r == Reg::Scc { u32::from(n != 0) } else { n };
                    sconst.insert(w, k);
                    uni.insert(w, Sym::K(k));
                    continue;
                }
                // A copy of a constant keeps it (an `s_mov_b*` from an
                // inline constant or from a scalar the section itself set
                // to one); anything else drops what the scalar held. A
                // 64-bit move gives each scalar its own half.
                let known = name
                    .starts_with("s_mov_b")
                    .then(|| constant(inst, &sconst))
                    .flatten();
                match known {
                    Some(k) => {
                        let part = match r {
                            Reg::S(code) => scalar_part(name, base, code, k),
                            _ => k as u32,
                        };
                        sconst.insert(w, part);
                    }
                    None => {
                        sconst.remove(&w);
                    }
                }
                // A copy of a scalar the section wrote keeps what it holds.
                let copied = match (src.first(), r) {
                    (Some(&Operand::Scalar { code: from, .. }), Reg::S(code))
                        if name.starts_with("s_mov_b") && from != NULL =>
                    {
                        inst.read(Reg::S(from + (code - base)))
                            .and_then(|v| uni.get(&v).copied())
                    }
                    _ => None,
                };
                uni.entry(w).or_insert(copied.unwrap_or(Sym::Unknown));
            }
            continue;
        }
        let Fields::Valu(v) = &inst.fields else {
            for &(_, w) in &inst.writes {
                vec.insert(w, vec![Sym::Unknown; wave as usize]);
            }
            continue;
        };
        let f16_inline = name == "v_add_f16";
        let lanes = |op: &Operand,
                     vec: &HashMap<Value, Vec<Sym>>,
                     uni: &HashMap<Value, Sym>|
         -> Vec<Sym> {
            let one = |s: Sym| vec![s; wave as usize];
            match *op {
                Operand::Vector { reg, .. } => match inst.read(Reg::V(reg)) {
                    Some(x) => vec
                        .get(&x)
                        .cloned()
                        .unwrap_or_else(|| (0..wave).map(|_| Sym::Own(x)).collect()),
                    None => one(Sym::Unknown),
                },
                Operand::Scalar { code, .. } if code != NULL => match inst.read(Reg::S(code)) {
                    Some(x) => one(uni.get(&x).copied().unwrap_or(Sym::Uni(x))),
                    None => one(Sym::Unknown),
                },
                Operand::Const(c @ 240..=248) if f16_inline => {
                    let h = [
                        0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
                    ][usize::from(c - 240)];
                    one(Sym::K(h))
                }
                Operand::Const(c) => one(crate::resource::inline(c).map_or(Sym::Unknown, Sym::K)),
                Operand::Literal(x) => one(Sym::K(x)),
                _ => one(Sym::Unknown),
            }
        };
        let modifiers = v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 || v.sdwa.is_some();
        if name == "v_readlane_b32" {
            let (Some(Operand::Scalar { code, .. }), Some(src), Some(lane)) =
                (v.dst, v.src.first(), v.src.get(1))
            else {
                continue;
            };
            let l = match *lane {
                Operand::Const(c) => crate::resource::inline(c),
                Operand::Literal(x) => Some(x),
                _ => None,
            };
            let x = lanes(src, &vec, &uni);
            if let Some(w) = inst.write(Reg::S(code)) {
                // `Own` is each lane's own value, not lane `l`'s.
                let s = l
                    .and_then(|l| x.get(l as usize).copied())
                    .filter(|s| !matches!(s, Sym::Own(_)))
                    .unwrap_or(Sym::Unknown);
                uni.insert(w, s);
            }
            continue;
        }
        if name == "v_add_f64"
            && let Some(Operand::Vector { reg: d, .. }) = v.dst
            && let (Some(low), Some(high)) = (inst.write(Reg::V(d)), inst.write(Reg::V(d + 1)))
        {
            // Dword `i` of operand `k` in every lane; operands other than
            // register pairs are not followed.
            let dword = |k: usize, i: u8| {
                let op = match v.src.get(k) {
                    Some(&Operand::Vector { reg, dwords }) => Operand::Vector {
                        reg: reg + i,
                        dwords,
                    },
                    Some(&Operand::Scalar { code, dwords }) if code != NULL => Operand::Scalar {
                        code: code + i,
                        dwords,
                    },
                    _ => return vec![Sym::Unknown; wave as usize],
                };
                lanes(&op, &vec, &uni)
            };
            let a = (dword(0, 0), dword(0, 1));
            let b = (dword(1, 0), dword(1, 1));
            let old = |reg| lanes(&Operand::Vector { reg, dwords: 1 }, &vec, &uni);
            let (mut lo, mut hi) = (old(d), old(d + 1));
            for l in 0..wave as usize {
                if exec >> l & 1 != 0 {
                    (lo[l], hi[l]) = if modifiers || v.dpp.is_some() {
                        (Sym::Unknown, Sym::Unknown)
                    } else {
                        add_f64((a.0[l], a.1[l]), (b.0[l], b.1[l]))
                    };
                }
            }
            vec.insert(low, lo);
            vec.insert(high, hi);
            continue;
        }
        let Some(Operand::Vector { reg: dreg, .. }) = v.dst else {
            for &(_, w) in &inst.writes {
                uni.insert(w, Sym::Unknown);
            }
            continue;
        };
        let Some(dst) = inst.write(Reg::V(dreg)) else {
            continue;
        };
        let old = lanes(
            &Operand::Vector {
                reg: dreg,
                dwords: 1,
            },
            &vec,
            &uni,
        );
        // src0 as each lane fetches it (DPP), with the lanes that write.
        let src0 = v.src.first().map(|o| lanes(o, &vec, &uni));
        let mut writes = exec;
        let fetched: Option<Vec<Sym>> = match (v.dpp, &src0) {
            (Some(d), Some(s)) => {
                if exec != all {
                    vec.insert(dst, vec![Sym::Unknown; wave as usize]);
                    continue;
                }
                let mut f = Vec::with_capacity(wave as usize);
                let mut ok = true;
                for l in 0..wave {
                    match dpp_lane(d, l) {
                        Some((src, zero, on)) if src < wave => {
                            f.push(if zero { Sym::K(0) } else { s[src as usize] });
                            if !on {
                                writes &= !(1u64 << l);
                            }
                        }
                        _ => ok = false,
                    }
                }
                ok.then_some(f)
            }
            (None, Some(s)) => Some(s.clone()),
            _ => None,
        };
        let src = |k: usize| v.src.get(k).map(|o| lanes(o, &vec, &uni));
        let result: Vec<Sym> = match (name, fetched) {
            _ if modifiers => vec![Sym::Unknown; wave as usize],
            ("v_mov_b32", Some(a)) => a,
            ("v_bfrev_b32", Some(a)) => a
                .into_iter()
                .map(|s| match s {
                    Sym::K(k) => Sym::K(k.reverse_bits()),
                    _ => Sym::Unknown,
                })
                .collect(),
            ("v_cndmask_b32", Some(a)) => {
                let mask = match v.src.get(2) {
                    Some(&Operand::Scalar { code, .. }) => inst.read(Reg::S(code)),
                    _ => None,
                };
                let b = src(1).unwrap_or_default();
                (0..wave as usize)
                    .map(|l| match (a[l], b.get(l).copied(), mask) {
                        (Sym::K(ident), Some(Sym::Own(x)), Some(m)) if masked.contains(&m) => {
                            Sym::Red {
                                op: None,
                                src: x,
                                mask: m,
                                ident,
                                lanes: 1 << l,
                            }
                        }
                        _ => Sym::Unknown,
                    })
                    .collect()
            }
            ("v_permlanex16_b32", Some(a)) => {
                // Lanes of each row read the other row of their half; lane
                // i's pick is nibble i of {src2, src1}.
                let sel = |k: usize| match v.src.get(k) {
                    Some(&Operand::Const(c)) => crate::resource::inline(c),
                    Some(&Operand::Literal(x)) => Some(x),
                    _ => None,
                };
                match (sel(1), sel(2)) {
                    (Some(s1), Some(s2)) if exec == all || v.opsel & 1 != 0 => {
                        let s = u64::from(s1) | u64::from(s2) << 32;
                        (0..wave)
                            .map(|l| {
                                let other = (l & !31) | ((l & 16) ^ 16);
                                let pick = (s >> (4 * (l & 15)) & 15) as u32;
                                a[(other | pick) as usize]
                            })
                            .collect()
                    }
                    _ => vec![Sym::Unknown; wave as usize],
                }
            }
            ("v_sub_nc_u32", Some(a)) => {
                // A prefix sum less the lane's own term: the exclusive one.
                let b = src(1).unwrap_or_default();
                (0..wave as usize)
                    .map(|l| match (a[l], b.get(l).copied()) {
                        (
                            Sym::Red {
                                op: None | Some(Op::AddU),
                                src,
                                mask,
                                ident: 0,
                                lanes,
                            },
                            Some(Sym::Red {
                                op: None | Some(Op::AddU),
                                src: s2,
                                mask: m2,
                                ident: 0,
                                lanes: l2,
                            }),
                        ) if (src, mask) == (s2, m2) && lanes & l2 == l2 => {
                            let rest = lanes & !l2;
                            if rest == 0 {
                                Sym::K(0)
                            } else {
                                Sym::Red {
                                    op: Some(Op::AddU),
                                    src,
                                    mask,
                                    ident: 0,
                                    lanes: rest,
                                }
                            }
                        }
                        _ => Sym::Unknown,
                    })
                    .collect()
            }
            ("v_add_f16", Some(a)) if v.opsel == 0 => {
                let b = src(1).unwrap_or_default();
                (0..wave as usize)
                    .map(|l| half(Op::AddH, a[l], b.get(l).copied(), old[l]))
                    .collect()
            }
            (n, Some(a)) => match (Op::of(n), src(1)) {
                (Some(op), Some(b)) => (0..wave as usize)
                    .map(|l| combine(op, a[l], b[l]))
                    .collect(),
                _ => vec![Sym::Unknown; wave as usize],
            },
            _ => vec![Sym::Unknown; wave as usize],
        };
        let merged = (0..wave as usize)
            .map(|l| {
                if writes >> l & 1 != 0 {
                    result[l]
                } else {
                    old[l]
                }
            })
            .collect();
        vec.insert(dst, merged);
    }
    (shapes(vec, uni, wave), None)
}

/// What scalar `reg`, written by an `s_mov_b32`/`s_mov_b64` of constant `k`
/// whose destination starts at `base`, holds: a 32-bit move its low half, a
/// 64-bit move each scalar its own half.
fn scalar_part(name: &str, base: u8, reg: u8, k: u64) -> u32 {
    if name == "s_mov_b64" && reg == base + 1 {
        (k >> 32) as u32
    } else {
        k as u32
    }
}

fn constant(inst: &Inst, sconst: &HashMap<Value, u32>) -> Option<u64> {
    let Fields::Sop { src, .. } = &inst.fields else {
        return None;
    };
    let wide = inst.name() == "s_mov_b64";
    let half = |code: u8| -> Option<u32> {
        inst.read(Reg::S(code)).and_then(|v| sconst.get(&v).copied())
    };
    match *src.first()? {
        // The scalars the section itself set to constants carry them on.
        Operand::Scalar { code, .. } if code != NULL => {
            let low = half(code)?;
            Some(if wide {
                u64::from(low) | (u64::from(half(code + 1)?) << 32)
            } else {
                u64::from(low)
            })
        }
        Operand::Scalar { code: NULL, .. } => Some(0),
        Operand::Const(c @ 128..=208) => {
            let x = crate::resource::inline(c)?;
            Some(if wide {
                i64::from(x as i32) as u64
            } else {
                u64::from(x)
            })
        }
        Operand::Const(c) if !wide => crate::resource::inline(c).map(u64::from),
        Operand::Literal(x) => Some(u64::from(x)),
        _ => None,
    }
}

/// The shape of each value, where every lane agrees on one.
fn shapes(
    vec: HashMap<Value, Vec<Sym>>,
    uni: HashMap<Value, Sym>,
    wave: u32,
) -> HashMap<Value, Shape> {
    let all = if wave == 64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    let mut out = HashMap::new();
    let single = |s: Sym| match s {
        Sym::K(k) => Some(Shape::K(k)),
        Sym::Own(x) | Sym::Uni(x) => Some(Shape::Value(x)),
        Sym::Red {
            op: Some(op),
            src,
            mask,
            ident,
            lanes,
        } if lanes == all => Some(Shape::Active(op, src, mask, ident)),
        Sym::Half {
            op,
            src,
            mask,
            ident,
            lanes,
            hi,
        } if lanes == all => Some(Shape::Half(op, src, mask, ident, hi)),
        Sym::Wide {
            lo,
            hi,
            mask,
            lanes,
            high,
        } if lanes == all => Some(Shape::Wide(lo, hi, mask, high)),
        _ => None,
    };
    for (v, s) in uni {
        if let Some(sh) = single(s) {
            out.insert(v, sh);
        }
    }
    for (v, lanes) in vec {
        if let Some(sh) = per_lane(&lanes, all) {
            out.insert(v, sh);
        }
    }
    out
}

fn per_lane(lanes: &[Sym], all: u64) -> Option<Shape> {
    let first = lanes[0];
    let own = |l: usize, s: Sym| match s {
        Sym::Red {
            op: None,
            src,
            mask,
            ident,
            lanes,
        } if lanes == 1 << l => Some((src, mask, ident)),
        _ => None,
    };
    if let Some(m) = own(0, first)
        && lanes.iter().enumerate().all(|(l, &s)| own(l, s) == Some(m))
    {
        return Some(Shape::Masked(m.0, m.1, m.2));
    }
    if lanes.iter().all(|&s| s == first) {
        return match first {
            Sym::K(k) => Some(Shape::K(k)),
            Sym::Uni(x) | Sym::Own(x) => Some(Shape::Value(x)),
            Sym::Red {
                op: Some(op),
                src,
                mask,
                ident,
                lanes,
            } if lanes == all => Some(Shape::Active(op, src, mask, ident)),
            Sym::Wide {
                lo,
                hi,
                mask,
                lanes,
                high,
            } if lanes == all => Some(Shape::Wide(lo, hi, mask, high)),
            _ => None,
        };
    }
    // Prefixes: lane l holds lanes 0..l (exclusive) or 0..=l (inclusive).
    let mut meta: Option<(Value, Value, u32, bool)> = None;
    let mut op_found: Option<Op> = None;
    for (l, &s) in lanes.iter().enumerate() {
        let below = (1u64 << l) - 1;
        let (op, src, mask, ident, set) = match s {
            Sym::Red {
                op,
                src,
                mask,
                ident,
                lanes,
            } => (op, src, mask, ident, lanes),
            // Lane 0 of an exclusive prefix holds the identity.
            Sym::K(_) if l == 0 => continue,
            _ => return None,
        };
        let inclusive = set == below | (1u64 << l);
        if !inclusive && set != below {
            return None;
        }
        if l == 0 && !inclusive {
            return None;
        }
        if op.is_none() && set.count_ones() != 1 {
            return None;
        }
        let lane_meta = (src, mask, ident, inclusive);
        match meta {
            None => meta = Some(lane_meta),
            Some(prev) if prev == lane_meta => {}
            _ => return None,
        }
        if let Some(o) = op {
            match op_found {
                None => op_found = Some(o),
                Some(prev) if prev == o => {}
                _ => return None,
            }
        }
    }
    let (src, mask, ident, inclusive) = meta?;
    let op = op_found?;
    if !op.identity(ident) {
        return None;
    }
    match lanes[0] {
        Sym::K(k) if !inclusive && k == ident => {}
        Sym::Red { .. } if inclusive => {}
        _ => return None,
    }
    Some(Shape::Prefix(op, src, mask, ident, inclusive))
}

/// The compiler's sum of doubles over a console wave64: v10 (low) and v11
/// (high) of the lanes in `vcc`, left in s12 (low) and s13 (high).
#[cfg(test)]
#[rustfmt::skip]
pub(super) const F64_SUM: &[u32] = &[
    0xBE80_25C1,              // s_or_saveexec_b64 s[0:1], -1
    0x7E04_0280,              // v_mov_b32 v2, 0
    0x7E06_0280,              // v_mov_b32 v3, 0
    0x8790_006A,              // s_and_b64 s[16:17], vcc, s[0:1]
    0xD501_0002, 0x0042_1502, // v_cndmask_b32 v2, v2, v10, s[16:17]
    0xD501_0003, 0x0042_1703, // v_cndmask_b32 v3, v3, v11, s[16:17]
    0x7E08_02E9, 0xDE54_C102, // v_mov_b32 v4, v2 dpp8:[1,0,3,2,5,4,7,6]
    0x7E0A_02E9, 0xDE54_C103, // v_mov_b32 v5, v3 (the same)
    0xD564_0002, 0x0002_0504, // v_add_f64 v[2:3], v[4:5], v[2:3]
    0x7E08_02E9, 0x9360_1202, // v_mov_b32 v4, v2 dpp8:[2,2,0,0,6,6,4,4]
    0x7E0A_02E9, 0x9360_1203,
    0xD564_0002, 0x0002_0504,
    0x7E08_02E9, 0x0009_2402, // v_mov_b32 v4, v2 dpp8:[4,4,4,4,0,0,0,0]
    0x7E0A_02E9, 0x0009_2403,
    0xD564_0002, 0x0002_0504,
    0x7E08_02FA, 0xFF01_4002, // v_mov_b32 v4, v2 row_mirror
    0x7E0A_02FA, 0xFF01_4003,
    0xD564_0002, 0x0002_0504,
    0xD778_0004, 0x0201_0102, // v_permlanex16_b32 v4, v2, 0, 0
    0xD778_0005, 0x0201_0103, // v_permlanex16_b32 v5, v3, 0, 0
    0xD564_0002, 0x0002_0504,
    0xD760_000E, 0x0001_4102, // v_readlane_b32 s14, v2, 32
    0xD760_000F, 0x0001_4103, // v_readlane_b32 s15, v3, 32
    0xD564_0002, 0x0002_040E, // v_add_f64 v[2:3], s[14:15], v[2:3]
    0xD760_000E, 0x0001_0102, // v_readlane_b32 s14, v2, 0
    0xD760_000F, 0x0001_0103, // v_readlane_b32 s15, v3, 0
    0xBE8C_030E,              // s_mov_b32 s12, s14
    0xBE8D_030F,              // s_mov_b32 s13, s15
    0xBEFE_0400,              // s_mov_b64 exec, s[0:1]
];

#[cfg(test)]
mod tests {
    use super::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    const fn sop1(sdst: u8, op: u8, ssrc0: u8) -> u32 {
        0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | ssrc0 as u32
    }

    const fn vop1(vdst: u8, op: u8, src0: u16) -> u32 {
        (0x3F << 25) | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
    }

    const fn vop2(vdst: u8, op: u8, src0: u16, vsrc1: u8) -> u32 {
        (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
    }

    const fn dpp16(vsrc0: u8, ctrl: u16, row_mask: u8, bank_mask: u8, bound_ctrl: bool) -> u32 {
        (row_mask as u32) << 28
            | (bank_mask as u32) << 24
            | (bound_ctrl as u32) << 19
            | (ctrl as u32) << 8
            | vsrc0 as u32
    }

    const fn vop3(vdst: u8, op: u16, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
        [
            0xD400_0000 | (op as u32) << 16 | vdst as u32,
            (src0 as u32 & 0x1FF) | ((src1 as u32 & 0x1FF) << 9) | ((src2 as u32 & 0x1FF) << 18),
        ]
    }

    fn build(words: &[u32], wave64: bool) -> Program {
        let code: Vec<u8> = words
            .iter()
            .copied()
            .chain(std::iter::once(S_ENDPGM))
            .flat_map(u32::to_le_bytes)
            .collect();
        crate::ir::build(&code, wave64, 256).unwrap()
    }

    #[test]
    fn dpp_lanes_follow_the_isa() {
        // row_shr:1 with bound_ctrl: lane 16 reads 0, lane 17 lane 16.
        let shr = Dpp::Row {
            ctrl: 0x111,
            row_mask: 0xF,
            bank_mask: 0xF,
            bound_ctrl: true,
            fi: false,
        };
        assert_eq!(dpp_lane(shr, 16), Some((16, true, true)));
        assert_eq!(dpp_lane(shr, 17), Some((16, false, true)));
        // row_mirror: lane 3 reads lane 12.
        let mirror = Dpp::Row {
            ctrl: 0x140,
            row_mask: 0xF,
            bank_mask: 0xF,
            bound_ctrl: true,
            fi: false,
        };
        assert_eq!(dpp_lane(mirror, 3), Some((12, false, true)));
        // dpp8 [1,0,3,2,5,4,7,6] swaps neighbours.
        let swap = Dpp::Lanes {
            sel: 1 | 3 << 6 | 2 << 9 | 5 << 12 | 4 << 15 | 7 << 18 | 6 << 21,
            fi: false,
        };
        assert_eq!(dpp_lane(swap, 8), Some((9, false, true)));
    }

    /// A wave64 sum of doubles leaves its two dwords in scalar copies; the
    /// lanes the sum was not gathered in (the upper half) keep no shape.
    #[test]
    fn a_wave64_f64_sum_leaves_its_two_dwords() {
        let p = build(F64_SUM, true);
        let insts = &p.blocks[0].insts;
        let of = |reg: Reg| insts.iter().find_map(|i| i.read(reg)).unwrap();
        let last = |reg: Reg| {
            insts
                .iter()
                .filter_map(|i| i.write(reg))
                .next_back()
                .unwrap()
        };
        let (lo, hi) = (of(Reg::V(10)), of(Reg::V(11)));
        let mask = insts[3].write(Reg::S(16)).unwrap();
        let sec = &sections(&p, 0)[&0];
        assert_eq!(sec.ended, None);
        for (reg, high) in [(12, false), (13, true)] {
            assert_eq!(
                sec.shapes.get(&last(Reg::S(reg))),
                Some(&Shape::Wide(lo, hi, mask, high)),
                "s{reg}"
            );
        }
        assert_eq!(sec.shapes.get(&last(Reg::V(2))), None);

        // Without the upper half's term the sum is not the wave's.
        let mut w = F64_SUM.to_vec();
        let add = w
            .windows(2)
            .position(|x| x == [0xD564_0002, 0x0002_040E])
            .unwrap();
        w.drain(add..add + 2);
        let p = build(&w, true);
        let s12 = p.blocks[0]
            .insts
            .iter()
            .filter_map(|i| i.write(Reg::S(12)))
            .next_back()
            .unwrap();
        assert_eq!(sections(&p, 0)[&0].shapes.get(&s12), None);
    }

    /// Lane 32 of a vector the section did not write is not each lane's own.
    #[test]
    fn a_lane_read_of_an_unwritten_vector_has_no_shape() {
        let words = [
            0xBE80_25C1,              // s_or_saveexec_b64 s[0:1], -1
            0xD760_000E, 0x0001_410A, // v_readlane_b32 s14, v10, 32
            0xBE8C_030E,              // s_mov_b32 s12, s14
            0xBEFE_0400,              // s_mov_b64 exec, s[0:1]
        ];
        let p = build(&words, true);
        let s12 = p.blocks[0]
            .insts
            .iter()
            .find_map(|i| i.write(Reg::S(12)))
            .unwrap();
        assert_eq!(sections(&p, 0)[&0].shapes.get(&s12), None);
    }

    #[test]
    fn prefixes_are_recognised() {
        let (x, m) = (Value(1), Value(2));
        let red = |lanes: u64| Sym::Red {
            op: Some(Op::AddU),
            src: x,
            mask: m,
            ident: 0,
            lanes,
        };
        let inclusive: Vec<Sym> = (0..64).map(|l| red(u64::MAX >> (63 - l))).collect();
        assert_eq!(
            per_lane(&inclusive, u64::MAX),
            Some(Shape::Prefix(Op::AddU, x, m, 0, true))
        );
        let mut exclusive: Vec<Sym> = (0..64).map(|l| red((1u64 << l) - 1)).collect();
        exclusive[0] = Sym::K(0);
        assert_eq!(
            per_lane(&exclusive, u64::MAX),
            Some(Shape::Prefix(Op::AddU, x, m, 0, false))
        );
        assert_eq!(
            per_lane(&vec![red(u64::MAX); 64], u64::MAX),
            Some(Shape::Active(Op::AddU, x, m, 0))
        );
    }

    #[test]
    fn constant_decodes_width_and_inline_kinds() {
        let b64 = 0x04;
        let b32 = 0x03;
        let check = |words: &[u32], wave64: bool| -> Option<u64> {
            let p = build(words, wave64);
            constant(&p.blocks[0].insts[0], &HashMap::new())
        };

        assert_eq!(
            check(&[sop1(EXEC, b64, 194)], true),
            Some(0xFFFF_FFFF_FFFF_FFFE)
        );
        assert_eq!(
            check(&[sop1(EXEC, b64, 196)], true),
            Some(0xFFFF_FFFF_FFFF_FFFC)
        );
        assert_eq!(
            check(&[sop1(EXEC, b64, 208)], true),
            Some(0xFFFF_FFFF_FFFF_FFF0)
        );
        assert_eq!(check(&[sop1(EXEC, b64, 193)], true), Some(u64::MAX));
        assert_eq!(check(&[sop1(EXEC, b64, NULL)], true), Some(0));
        assert_eq!(check(&[sop1(EXEC, b64, 243)], true), None);
        assert_eq!(
            check(&[sop1(EXEC, b64, 255), 0x1234_5678], true),
            Some(0x1234_5678)
        );

        assert_eq!(check(&[sop1(EXEC, b32, 194)], false), Some(0xFFFF_FFFE));
        assert_eq!(check(&[sop1(EXEC, b32, 243)], false), Some(0xBF80_0000));
        assert_eq!(check(&[sop1(EXEC, b32, NULL)], false), Some(0));
        assert_eq!(
            check(&[sop1(EXEC, b32, 255), 0x1234_5678], false),
            Some(0x1234_5678)
        );
        assert_eq!(check(&[sop1(EXEC, b32, 0)], false), None);
    }

    /// A 64-bit `s_mov_b64` keeps its halves in the two scalars; a 32-bit
    /// read of the high one gets the high half, and a 64-bit read refuses
    /// unless both are known (#2382 review).
    #[test]
    fn a_64_bit_scalar_keeps_its_halves_apart() {
        let (b64, b32) = (0x04, 0x03);
        // s_mov_b32 s8, s5: what the section made s5 hold.
        let p = build(&[sop1(8, b32, 5)], false);
        let inst = &p.blocks[0].insts[0];
        let mut map = HashMap::new();
        map.insert(inst.read(Reg::S(5)).unwrap(), 0x89AB_CDEF);
        assert_eq!(constant(inst, &map), Some(0x89AB_CDEF));
        // s_mov_b64 s[8:9], s[4:5]: both halves, or none.
        let p = build(&[sop1(8, b64, 4)], true);
        let inst = &p.blocks[0].insts[0];
        let mut map = HashMap::new();
        map.insert(inst.read(Reg::S(4)).unwrap(), 0x89AB_CDEF);
        map.insert(inst.read(Reg::S(5)).unwrap(), 0x0123_4567);
        assert_eq!(constant(inst, &map), Some(0x0123_4567_89AB_CDEF));
        map.remove(&inst.read(Reg::S(5)).unwrap());
        assert_eq!(constant(inst, &map), None, "either half unknown: refuse");
        // The store: a 64-bit move gives each scalar its own half.
        let k = 0x0123_4567_89AB_CDEF;
        assert_eq!(scalar_part("s_mov_b64", 8, 8, k), 0x89AB_CDEF);
        assert_eq!(scalar_part("s_mov_b64", 8, 9, k), 0x0123_4567);
        assert_eq!(scalar_part("s_mov_b32", 8, 8, k), 0x89AB_CDEF);
    }

    /// A 64-bit `s_mov_b64 exec, s[4:5]` takes both halves the section set
    /// with separate 32-bit moves: `EXEC` is every lane, and work after it
    /// keeps a shape instead of ending as unknown (#2382 review).
    #[test]
    fn a_64_bit_exec_move_takes_both_scalar_halves() {
        let w = [
            sop1(0, 0x25, 193),    // s_or_saveexec_b64 s0, -1 (every lane)
            sop1(4, 0x03, 193),    // s_mov_b32 s4, -1
            sop1(5, 0x03, 193),    // s_mov_b32 s5, -1
            sop1(EXEC, 0x04, 4),   // s_mov_b64 exec, s[4:5]
            vop1(1, 0x01, 0xFA),   // v_mov_b32 v1, 7
            dpp16(1, 0x111, 0xF, 0xF, false),
            sop1(EXEC, 0x04, 0),   // s_mov_b64 exec, s[0:1]: the section ends
        ];
        let p = build(&w, true);
        let insts = &p.blocks[0].insts;
        let value = insts
            .iter()
            .filter_map(|i| i.write(Reg::V(1)))
            .next_back()
            .expect("the DPP writes v1");
        let sec = &sections(&p, 0)[&0];
        assert!(
            sec.shapes.contains_key(&value),
            "EXEC is every lane: the DPP keeps a shape, not unknown"
        );
    }

    #[test]
    fn exec_minus_two_masks_lane_zero_and_enforces_dpp_guard() {
        for wave64 in [false, true] {
            let mut w = Vec::new();
            let save_op = if wave64 { 0x25 } else { 0x3D };
            let mov_exec = if wave64 { 0x04 } else { 0x03 };
            let push_exec = |w: &mut Vec<u32>, k: u8| w.push(sop1(EXEC, mov_exec, k));
            let push_exec_lit = |w: &mut Vec<u32>, lit: u32| {
                w.push(sop1(EXEC, mov_exec, 255));
                w.push(lit);
            };

            w.push(sop1(0, save_op, 193));
            w.extend(vop3(1, 0x101, 128, 256, 0));
            for shift in [1u16, 2, 4, 8] {
                w.push(vop2(1, 0x25, 0xFA, 1));
                w.push(dpp16(1, 0x110 + shift, 0xF, 0xF, false));
            }
            w.extend(vop3(2, 0x360, 257, 128 + 15, 0));
            push_exec_lit(&mut w, 0xFFFF_0000);
            w.push(vop2(1, 0x25, 2, 1));

            if wave64 {
                w.extend(vop3(2, 0x360, 257, 128 + 31, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 255));
                w.push(0x0000_FFFF);
                w.push(vop2(1, 0x25, 2, 1));

                w.extend(vop3(2, 0x360, 257, 128 + 47, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 255));
                w.push(0xFFFF_0000);
                w.push(vop2(1, 0x25, 2, 1));
            }

            push_exec(&mut w, 193);
            w.push(vop1(3, 0x01, 257));
            w.push(vop1(3, 0x01, 0xFA));
            w.push(dpp16(1, 0x111, 0xF, 0xF, false));

            w.extend(vop3(2, 0x360, 257, 128 + 15, 0));
            push_exec_lit(&mut w, 0x0001_0000);
            w.push(vop1(3, 0x01, 2));

            if wave64 {
                w.extend(vop3(2, 0x360, 257, 128 + 31, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 129));
                w.push(vop1(3, 0x01, 2));

                w.extend(vop3(2, 0x360, 257, 128 + 47, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 255));
                w.push(0x0001_0000);
                w.push(vop1(3, 0x01, 2));
            }

            push_exec(&mut w, 193);
            w.push(vop1(2, 0x01, 128));
            w.push(vop1(5, 0x01, 135));
            push_exec(&mut w, 194);
            w.push(vop1(2, 0x01, 259));
            w.push(vop1(4, 0x01, 0xFA));
            w.push(dpp16(5, 0xE4, 0xF, 0xF, false));
            w.push(sop1(EXEC, mov_exec, 0));

            let p = build(&w, wave64);
            let insts = &p.blocks[0].insts;
            let saved = insts[0].write(Reg::S(0)).unwrap();
            let v0_in = p
                .inputs
                .iter()
                .find_map(|&(r, v)| (r == Reg::V(0)).then_some(v))
                .unwrap();
            let v2_val = insts[insts.len() - 4].write(Reg::V(2)).unwrap();
            let v4_val = insts[insts.len() - 3].write(Reg::V(4)).unwrap();
            let sec = &sections(&p, 0)[&0];
            assert_eq!(
                sec.shapes.get(&v2_val),
                Some(&Shape::Prefix(Op::AddU, v0_in, saved, 0, false))
            );
            assert_eq!(sec.shapes.get(&v4_val), None);
        }
    }

    #[test]
    fn wave32_mov_b64_all_lanes_keeps_exec_masked_for_dpp() {
        let w = [
            sop1(0, 0x3D, 193),
            vop1(1, 0x01, 135),
            sop1(EXEC, 0x04, 193),
            vop1(2, 0x01, 0xFA),
            dpp16(1, 0xE4, 0xF, 0xF, false),
            sop1(EXEC + 1, 0x03, 193),
            vop1(3, 0x01, 0xFA),
            dpp16(1, 0xE4, 0xF, 0xF, false),
            sop1(EXEC, 0x03, 0),
        ];
        let p = build(&w, false);
        let insts = &p.blocks[0].insts;
        let v2_val = insts[3].write(Reg::V(2)).unwrap();
        let v3_val = insts[5].write(Reg::V(3)).unwrap();
        let sec = &sections(&p, 0)[&0];
        assert_eq!(sec.shapes.get(&v2_val), Some(&Shape::K(7)));
        assert_eq!(sec.shapes.get(&v3_val), Some(&Shape::K(7)));
    }

    /// A value a whole-wave section cannot shape is refused with the step it
    /// stopped at *and* the category (#2365 review): the report must not read
    /// as an unimplemented `s_not_b32`.
    #[test]
    fn a_section_step_keeps_its_category() {
        let w = [
            sop1(0, 0x3D, 193),    // s_or_saveexec_b64 s0, -1
            vop1(1, 0x01, 135),    // v_mov_b32 v1, 7
            sop1(EXEC, 0x07, 193), // s_not_b32 exec, -1: not a constant set
            sop1(EXEC, 0x03, 0),   // s_mov_b32 exec, s0: the section ends
        ];
        let p = build(&w, false);
        let sec = &sections(&p, 0)[&0];
        assert_eq!(sec.ended, Some("a whole-wave section step: s_not_b32"));
    }

    /// A save whose restore is not in the block does not hide the sections
    /// after it.
    #[test]
    fn an_unmatched_saveexec_keeps_later_sections() {
        let w = [
            sop1(0, 0x3D, 193),  // s_or_saveexec_b32 s0, -1 (never restored)
            sop1(2, 0x3D, 193),  // s_or_saveexec_b32 s2, -1
            vop1(1, 0x01, 135),  // v_mov_b32 v1, 7
            sop1(EXEC, 0x03, 2), // s_mov_b32 exec_lo, s2 (the section ends)
        ];
        let p = build(&w, false);
        let found = sections(&p, 0);
        assert!(!found.contains_key(&0));
        assert_eq!(found.get(&1).map(|s| s.end), Some(3));
    }

    #[test]
    fn per_lane_prefix_soundness() {
        let (v0, v_other, m0, m_other) = (Value(1), Value(2), Value(3), Value(4));
        let red = |op: Option<Op>, src: Value, mask: Value, ident: u32, lanes: u64| Sym::Red {
            op,
            src,
            mask,
            ident,
            lanes,
        };

        for (wave, all) in [(32usize, u64::from(u32::MAX)), (64usize, u64::MAX)] {
            let below_mask = |l: usize| if l == 0 { 0 } else { (1u64 << l) - 1 };

            let mut valid_inc: Vec<Sym> = (0..wave)
                .map(|l| red(Some(Op::AddU), v0, m0, 0, below_mask(l) | (1u64 << l)))
                .collect();
            valid_inc[0] = red(None, v0, m0, 0, 1);
            assert_eq!(
                per_lane(&valid_inc, all),
                Some(Shape::Prefix(Op::AddU, v0, m0, 0, true))
            );

            let mut valid_exc: Vec<Sym> = (0..wave)
                .map(|l| red(Some(Op::AddU), v0, m0, 0, below_mask(l)))
                .collect();
            valid_exc[0] = Sym::K(0);
            valid_exc[1] = red(None, v0, m0, 0, 1);
            assert_eq!(
                per_lane(&valid_exc, all),
                Some(Shape::Prefix(Op::AddU, v0, m0, 0, false))
            );

            let mut mixed_other_src = valid_exc.clone();
            mixed_other_src[0] = red(None, v_other, m0, 0, 1);
            assert_eq!(per_lane(&mixed_other_src, all), None);

            let mut mixed_same_src = valid_exc.clone();
            mixed_same_src[0] = red(None, v0, m0, 0, 1);
            assert_eq!(per_lane(&mixed_same_src, all), None);

            let mut exc_bad_lane1_src = valid_exc.clone();
            exc_bad_lane1_src[1] = red(None, v_other, m0, 0, 1);
            assert_eq!(per_lane(&exc_bad_lane1_src, all), None);

            let mut exc_bad_lane1_set = valid_exc.clone();
            exc_bad_lane1_set[1] = red(None, v0, m0, 0, 0b11);
            assert_eq!(per_lane(&exc_bad_lane1_set, all), None);

            for bad0 in [
                red(None, v0, m0, 1, 1),
                red(None, v_other, m0, 0, 1),
                red(None, v0, m_other, 0, 1),
                red(None, v0, m0, 0, 0),
                Sym::K(0),
            ] {
                let mut bad_inc = valid_inc.clone();
                bad_inc[0] = bad0;
                assert_eq!(per_lane(&bad_inc, all), None);
            }

            let mut bad_ident: Vec<Sym> = (0..wave)
                .map(|l| red(Some(Op::AddU), v0, m0, 1, below_mask(l) | (1u64 << l)))
                .collect();
            bad_ident[0] = red(None, v0, m0, 1, 1);
            assert_eq!(per_lane(&bad_ident, all), None);
        }
    }

    /// `EXEC` set from a scalar the section itself set to a constant: the
    /// simulation follows it instead of stopping there, which is what a
    /// section that masks itself with a computed all-ones needs.
    #[test]
    fn a_section_follows_exec_set_from_a_scalar_constant() {
        let w = [
            sop1(0, 0x3D, 193),  // s_or_saveexec_b32 s0, -1 (every lane)
            sop1(5, 0x03, 193),  // s_mov_b32 s5, -1
            sop1(EXEC, 0x03, 5), // s_mov_b32 exec_lo, s5
            vop1(2, 0x01, 135),  // v_mov_b32 v2, 7
            sop1(EXEC, 0x03, 0), // s_mov_b32 exec_lo, s0 (the section ends)
        ];
        let p = build(&w, false);
        let insts = &p.blocks[0].insts;
        let v2_val = insts[3].write(Reg::V(2)).unwrap();
        let sec = &sections(&p, 0)[&0];
        assert_eq!(sec.shapes.get(&v2_val), Some(&Shape::K(7)));
    }

    #[test]
    fn v_add_f16_inline_float_constants_are_not_zero_identity() {
        // Wave32 f16 sum reduction with an extra `v_add_f16 v1, <const>, v1`
        // step: 0 (code 128) is a true additive identity, whereas inline float
        // constants (240..=248: 0.5, -0.5, 1.0, ..., 1/(2*pi)) are non-zero
        // f16 values on GFX10 and must not be decoded as f32 bits whose low 15
        // bits happen to be zero.
        let run = |src_const: u16| -> (Option<Shape>, Value, Value) {
            let mut w = vec![sop1(0, 0x3D, 193)]; // s_or_saveexec_b32 s0, -1
            w.extend(vop3(1, 0x101, 128, 256, 0)); // v_cndmask_b32 v1, 0, v0, s0
            w.push(vop2(1, 0x32, src_const, 1)); // v_add_f16 v1, <const>, v1
            for ctrl in [0x00B1u16, 0x004E, 0x0164, 0x0168] {
                w.push(vop2(1, 0x32, 0xFA, 1));
                w.push(dpp16(1, ctrl, 0xF, 0xF, false));
            }
            w.extend(vop3(2, 0x378, 257, 128, 128)); // v_permlanex16_b32 v2, v1, 0, 0
            w.push(vop2(1, 0x32, 258, 1)); // v_add_f16 v1, v2, v1
            w.extend(vop3(1, 0x360, 257, 128 + 31, 0)); // v_readlane_b32 s1, v1, 31
            w.push(sop1(EXEC, 0x03, 0)); // s_mov_b32 exec_lo, s0

            let p = build(&w, false);
            let insts = &p.blocks[0].insts;
            let saved = insts[0].write(Reg::S(0)).unwrap();
            let v0_in = insts[1].read(Reg::V(0)).unwrap();
            let s1 = insts[insts.len() - 3].write(Reg::S(1)).unwrap();
            let sec = &sections(&p, 0)[&0];
            (sec.shapes.get(&s1).copied(), v0_in, saved)
        };

        let (zero_shape, v0_in, saved) = run(128);
        assert_eq!(
            zero_shape,
            Some(Shape::Half(Op::AddH, v0_in, saved, 0, 31))
        );
        for c in 240..=248u16 {
            let (shape, _, _) = run(c);
            assert_eq!(shape, None, "inline float const {c} was treated as identity");
        }
    }
}
