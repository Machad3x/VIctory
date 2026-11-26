//! Instructions whose effects reach past their registers.

use super::{Fields, Inst};

pub(super) fn has_side_effect(inst: &Inst) -> bool {
    let name = inst.name();
    matches!(inst.fields, Fields::Exp { .. })
        || matches!(inst.fields, Fields::Flat { .. }) && !name.starts_with("global_load_")
        || name.contains("store")
        || name.contains("atomic")
        || name == "s_barrier"
        || matches!(inst.fields, Fields::Ds { .. })
            && !(name.starts_with("ds_read")
                || name.starts_with("ds_swizzle")
                || name.contains("permute"))
}
