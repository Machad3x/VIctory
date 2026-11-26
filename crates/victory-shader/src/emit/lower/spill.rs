//! Which lane accesses of SGPR spills and restores need no lane operation.

use std::collections::{HashMap, HashSet};

use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Inst, Program, Reg, Value};
use crate::resource::Site;
use crate::structure::Structured;

/// Instructions that read a lane mask (not EXEC implicitly, not a lane's
/// bit) as a number.
pub(super) fn reads_mask_as_data(st: &Structured, i: &Inst) -> bool {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let ops: Vec<&Operand> = match &i.fields {
        // A mask operation (its result a mask) is no data use.
        Fields::Sop { .. }
            if i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(w)) =>
        {
            return false;
        }
        Fields::Sop { src, .. } => src.iter().collect(),
        Fields::Valu(v)
            if !matches!(i.name(), "v_cndmask_b32") && !i.name().contains("_co_ci_") =>
        {
            v.src.iter().collect()
        }
        _ => return false,
    };
    ops.into_iter().any(|op| match *op {
        Operand::Scalar { code, .. } if code != NULL => i.read(Reg::S(code)).is_some_and(is_mask),
        _ => false,
    })
}

/// The per-lane lowering can replace a proven restore with its original SGPR.
pub(super) fn restored_lane_read(st: &Structured, inst: &Inst) -> bool {
    inst.writes
        .first()
        .is_some_and(|&(_, result)| st.restored.contains_key(&result))
}

/// Constant-lane SGPR writes whose VGPR is used only by other such writes
/// and by proven restores. A VGPR use elsewhere, even through a phi, needs
/// the original lane write to be carried out.
pub(super) fn spill_writes(
    p: &Program,
    st: &Structured,
    phi_at: &HashMap<Value, Site>,
) -> HashMap<Value, Option<Value>> {
    let insts = || p.blocks.iter().flat_map(|b| &b.insts);
    let mut spills = HashMap::new();
    for inst in insts().filter(|i| i.name() == "v_writelane_b32") {
        if let Fields::Valu(w) = &inst.fields
            && let (Some(Operand::Vector { reg, .. }), [Operand::Scalar { code, .. }, lane]) =
                (w.dst, w.src.as_slice())
            && *code != NULL
            && crate::resource::lane_of(*lane).is_some()
            && let Some(result) = inst.write(Reg::V(reg))
        {
            spills.insert(result, inst.read(Reg::V(reg)));
        }
    }
    if spills.is_empty() {
        return spills;
    }

    let mut used: Vec<_> = insts()
        .filter(|inst| !is_sgpr_spill(st, inst, &spills))
        .flat_map(|inst| &inst.reads)
        .filter_map(|&(reg, value)| matches!(reg, Reg::V(_)).then_some(value))
        .collect();
    let mut seen_phis = HashSet::new();
    while let Some(value) = used.pop() {
        if let Some(previous) = spills.remove(&value) {
            used.extend(previous);
        } else if let Some(&(b, i)) = phi_at.get(&value)
            && seen_phis.insert(value)
        {
            used.extend(&p.blocks[b].phis[i].args);
        }
    }
    spills
}

pub(super) fn is_named_lane(inst: &Inst) -> bool {
    matches!(inst.name(), "v_readlane_b32" | "v_writelane_b32")
}

pub(super) fn is_sgpr_spill(
    st: &Structured,
    inst: &Inst,
    spill_writes: &HashMap<Value, Option<Value>>,
) -> bool {
    match inst.name() {
        "v_readlane_b32" => restored_lane_read(st, inst),
        "v_writelane_b32" => inst
            .writes
            .first()
            .is_some_and(|&(_, result)| spill_writes.contains_key(&result)),
        _ => false,
    }
}
