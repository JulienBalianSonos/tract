use super::*;

pub(super) type Layout = TVec<TVec<usize>>;

/// Tracks logical head, group, query, key and feature coordinates through views.
/// Unit dimensions carry no coordinate; reshapes must preserve factor order.
pub(super) struct Coordinates {
    pub dims: [TDim; 5],
}

impl Coordinates {
    /// Assign coordinate factors to axes, omitting statically unit dimensions.
    pub fn layout(&self, axes: &[&[usize]]) -> Layout {
        axes.iter()
            .map(|axis| axis.iter().copied().filter(|&r| self.dims[r] != 1.to_dim()).collect())
            .collect()
    }

    /// Prove coordinate flow from the supplied anchors through f16 views and
    /// einsums. Only the requested contraction coordinate may be reduced.
    pub fn trace(
        &self,
        model: &TypedModel,
        outlet: OutletId,
        anchors: &[(OutletId, Layout)],
        reduction: usize,
        depth: usize,
    ) -> Option<Layout> {
        if depth > 24 || outlet.slot != 0 {
            return None;
        }
        if let Some((_, layout)) = anchors.iter().find(|(id, _)| *id == outlet) {
            let fact = model.outlet_fact(outlet).ok()?;
            if fact.datum_type != DatumType::F16 || fact.rank() != layout.len() {
                return None;
            }
            for (roles, dim) in layout.iter().zip(fact.shape.iter()) {
                let product = roles.iter().fold(1.to_dim(), |p, &r| p * self.dims[r].clone());
                if product != *dim {
                    return None;
                }
            }
            return Some(layout.clone());
        }
        let node = model.node(outlet.node);
        if node.outputs[0].fact.datum_type != DatumType::F16 {
            return None;
        }
        let trace = |input| self.trace(model, input, anchors, reduction, depth + 1);
        if let Some(op) = node.op_as::<EinSum>() {
            if node.inputs.len() != 2 || op.q_params.is_some() || op.operating_dt != DatumType::F16
            {
                return None;
            }
            let inputs = [trace(node.inputs[0])?, trace(node.inputs[1])?];
            let mut output = tvec![tvec![]; node.outputs[0].fact.rank()];
            let mut seen = [false; 5];
            for axis in op.axes.iter_all_axes() {
                if axis.outputs[0].len() > 1
                    || axis.inputs.iter().any(|positions| positions.len() > 1)
                {
                    return None;
                }
                let mut roles = tvec![];
                for (ix, positions) in axis.inputs.iter().enumerate() {
                    if let Some(&position) = positions.first() {
                        let candidate = inputs[ix].get(position)?;
                        if !candidate.is_empty() {
                            if !roles.is_empty() && roles != *candidate {
                                return None;
                            }
                            roles = candidate.clone();
                        }
                    }
                }
                for &role in &roles {
                    if seen[role] {
                        return None;
                    }
                    seen[role] = true;
                }
                if let Some(&position) = axis.outputs[0].first() {
                    output[position] = roles;
                } else if !roles.is_empty() && roles != tvec![reduction] {
                    return None;
                }
            }
            return Some(output);
        }
        if node.inputs.len() != 1 {
            return None;
        }
        let mut layout = trace(node.inputs[0])?;
        if let Some(op) = node.op_as::<AxisOp>() {
            match op {
                AxisOp::Add(axis) => layout.insert(*axis, tvec![]),
                AxisOp::Rm(axis) => {
                    if !layout.get(*axis)?.is_empty() {
                        return None;
                    }
                    layout.remove(*axis);
                }
                AxisOp::Move(from, to) => {
                    let roles = layout.remove(*from);
                    layout.insert(*to, roles);
                }
                AxisOp::Reshape(start, from, to) => {
                    let roles: TVec<usize> =
                        layout.get(*start..start + from.len())?.iter().flatten().copied().collect();
                    let mut index = 0;
                    let mut replacement = tvec![];
                    for dim in to {
                        let mut product = 1.to_dim();
                        let mut axis = tvec![];
                        while product != *dim {
                            let role = *roles.get(index)?;
                            product *= self.dims[role].clone();
                            axis.push(role);
                            index += 1;
                        }
                        replacement.push(axis);
                    }
                    if index != roles.len() {
                        return None;
                    }
                    layout = layout[..*start]
                        .iter()
                        .cloned()
                        .chain(replacement)
                        .chain(layout[start + from.len()..].iter().cloned())
                        .collect();
                }
            }
        } else if node.op_is::<MultiBroadcastTo>() {
            let input = model.outlet_fact(node.inputs[0]).ok()?;
            let output = &node.outputs[0].fact;
            if input.rank() != output.rank() {
                return None;
            }
            for (axis, (a, b)) in input.shape.iter().zip(output.shape.iter()).enumerate() {
                if a != b {
                    if *a != 1.to_dim()
                        || *b != self.dims[1]
                        || layout.iter().flatten().any(|&r| r == 1)
                    {
                        return None;
                    }
                    layout[axis] = tvec![1];
                }
            }
        } else if !node.op_is::<Cast>()
            || model.outlet_fact(node.inputs[0]).ok()?.datum_type != DatumType::F16
        {
            return None;
        }
        Some(layout)
    }
}
