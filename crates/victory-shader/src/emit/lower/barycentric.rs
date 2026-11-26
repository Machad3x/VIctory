//! Barycentrics a pixel program reads directly, not through `v_interp_*`.
//! After avatarmanz/VIctory-7900XTX (from fxceless/victory-fork 578d459c).

use super::*;

impl<'a> Lower<'a> {
    /// The kind and index of `v`, if it is a barycentric input VGPR.
    pub(super) fn barycentric_input(&self, v: Value) -> Option<(Barycentric, u8)> {
        self.x.io.inputs.iter().find_map(|&(w, i)| match i {
            Input::Barycentric { kind, index } if w == v => Some((kind, index)),
            _ => None,
        })
    }

    /// Loads `v` if it is a barycentric input VGPR not loaded yet (they are
    /// loaded on first use: interpolation alone declares none): `None` if it
    /// is not one.
    pub(super) fn lazy_barycentric(&mut self, v: Value) -> Option<Result<(), Why>> {
        let (kind, index) = self.barycentric_input(v)?;
        let val = self.barycentric(kind, index);
        self.vals.insert(v, val);
        Some(match val {
            Val::Missing(why) => Err(why),
            _ => Ok(()),
        })
    }

    /// Barycentric `index` (0 i, 1 j) of `kind`: a channel of
    /// `SV_Barycentrics` declared with the same interpolation mode.
    ///
    /// GFX10 interpolates P0 + i (P1 - P0) + j (P2 - P0), so i and j are
    /// the weights of vertices 1 and 2. The host's weights (x, y, z) are
    /// those of vertices 0, 1 and 2, vertex 0 being the provoking vertex,
    /// which is the console's P0 (`v_interp_mov` P0 is lowered to a flat
    /// input): i is y and j is z.
    /// shortcut: triangles only; on lines and points the console's i and j
    /// are not checked against the host's weights.
    pub(super) fn barycentric(&mut self, kind: Barycentric, index: u8) -> Val {
        let mode = match kind {
            Barycentric::PerspectiveCenter => 2,
            Barycentric::PerspectiveCentroid => 3,
            Barycentric::PerspectiveSample => 6,
            Barycentric::LinearCenter => 4,
            Barycentric::LinearCentroid => 5,
            Barycentric::LinearSample => 7,
            Barycentric::PerspectivePullModel => {
                return Val::Missing("pull-model barycentrics");
            }
        };
        if index > 1 {
            return Val::Missing("pull-model barycentrics");
        }
        // One element per interpolation mode (the console's own, if its
        // signature declares one): DXIL allows two.
        let bary = |e: &SigElement| e.kind == 28;
        let found = self
            .s
            .inputs
            .iter()
            .position(|e| bary(e) && (e.interp == mode || e.usage == 0));
        let id = match found {
            Some(id) => id,
            None => {
                let index = self.s.inputs.iter().filter(|e| bary(e)).count() as u32;
                if index >= 2 {
                    return Val::Missing("barycentrics of a third interpolation mode");
                }
                // A system value the validator wants unpacked (packing
                // location -1), like other non-packed system values.
                self.s.inputs.push(SigElement {
                    name: "SV_Barycentrics".into(),
                    index,
                    kind: 28,
                    comp: 9,
                    interp: mode,
                    row: None,
                    col: 0,
                    cols: 3,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        let e = &mut self.s.inputs[id];
        if e.col != 0 || e.cols < 3 {
            return Val::Missing("a console barycentrics element without i and j");
        }
        e.row = None;
        e.interp = mode;
        self.s.flags |= flags::BARYCENTRICS;
        self.load_any_input(id, index + 1)
    }
}
