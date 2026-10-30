use cubecl_core::ir::{
    dialect::{
        bitwise::{CountOnesOp, LeadingZerosBitsOp, ReverseBitsOp, TrailingZerosBitsOp},
        general::{CastOp, ReinterpretCastOp},
        math::FAbsOp,
    },
    interfaces::TypedExt,
    prelude::*,
    types::VectorType,
};

use crate::{
    opencl::{
        call, opencl_op_with_out,
        ty::{native_vector, scalar_name},
    },
    shared::ty::TypeExtCPP,
};

// C casts, but float to integer (undefined in C out of range) saturates, toward zero and NaN to 0, as
// PTX and WGSL convert.
opencl_op_with_out!(CastOp, |op, ctx| {
    let input = op.input(ctx);
    let result = op.get_result(ctx);
    let scalar = |value: Value| {
        value
            .try_get_scalar_ty(ctx)
            .and_then(|ty| scalar_name(ctx, ty))
    };
    let float = |name: &str| matches!(name, "half" | "float" | "double");
    let ty = result.get_type(ctx);
    let target = match ty.deref(ctx).downcast_ref::<VectorType>() {
        Some(vector) => native_vector(ctx, vector),
        None => scalar(result).map(String::from),
    };
    match (scalar(input), scalar(result), target) {
        (Some(from), Some(to), Some(target)) if float(from) && !float(to) => {
            format!("convert_{target}_sat({})", input.name(ctx))
        }
        _ => format!("({})({})", ty.to_cpp(ctx), input.name(ctx)),
    }
});

// A value reinterpreted through a union (well defined in C); pointers by a cast.
opencl_op_with_out!(ReinterpretCastOp, |op, ctx| {
    let input = op.input(ctx);
    let ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    if input.is_ptr(ctx) {
        format!("({ty})({})", input.name(ctx))
    } else {
        let in_ty = input.get_type(ctx).to_cpp(ctx);
        format!(
            "((union {{ {in_ty} from; {ty} to; }}){{ .from = {} }}).to",
            input.name(ctx)
        )
    }
});

call!(FAbsOp, "fabs", input);
call!(CountOnesOp, "popcount", input);
call!(LeadingZerosBitsOp, "clz", input);
call!(TrailingZerosBitsOp, "ctz", input);

// `reverse_bits32` / `64`: `bit_reverse` where the device has it (the module's header).
opencl_op_with_out!(ReverseBitsOp, |op, ctx| {
    let input = op.input(ctx);
    let ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    let (bits, unsigned) = match input.size(ctx) {
        8 => (64, "ulong"),
        _ => (32, "uint"),
    };
    format!(
        "({ty})reverse_bits{bits}(({unsigned})({}))",
        input.name(ctx)
    )
});
