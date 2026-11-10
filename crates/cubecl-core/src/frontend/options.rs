use cubecl_ir::EnumSet;
use cubecl_ir::{FastMath, Scope};

/// Expands `body` under the fast-math modes `value` (`fast_math` on a function). A program-wide
/// mode may only narrow: what the kernel allows ([`kernel_fast_math_expand`]) and every region
/// inside it allows.
pub fn fast_math_expand<R>(
    scope: &Scope,
    value: EnumSet<FastMath>,
    body: impl FnOnce(&Scope) -> R,
) -> R {
    let prev = scope.state().modes.fp_math_mode;
    scope.state_mut().modes.fp_math_mode = value;
    let program = scope.state().kernel_fp_math;
    scope.state_mut().kernel_fp_math = program.intersection(value);
    let res = body(scope);
    scope.state_mut().modes.fp_math_mode = prev;

    res
}

/// Expands a kernel's `body` under its `fast_math` modes `value`: also the modes the whole program
/// may assume (`GlobalState::kernel_fp_math`), which the regions inside it can only narrow.
pub fn kernel_fast_math_expand<R>(
    scope: &Scope,
    value: EnumSet<FastMath>,
    body: impl FnOnce(&Scope) -> R,
) -> R {
    scope.state_mut().kernel_fp_math = value;
    fast_math_expand(scope, value, body)
}
