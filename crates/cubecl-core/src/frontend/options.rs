use cubecl_ir::EnumSet;
use cubecl_ir::{FastMath, Scope};

pub fn fast_math_expand<R>(
    scope: &Scope,
    value: EnumSet<FastMath>,
    body: impl FnOnce(&Scope) -> R,
) -> R {
    let prev = scope.state().modes.fp_math_mode;
    scope.state_mut().modes.fp_math_mode = value;
    scope.state_mut().kernel_fp_math |= value;
    let res = body(scope);
    scope.state_mut().modes.fp_math_mode = prev;

    res
}
