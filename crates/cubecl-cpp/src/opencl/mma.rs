//! Cooperative matrices on Intel's DPAS (`cl_intel_subgroup_matrix_multiply_accumulate`): an 8-row
//! tile as wide as the subgroup (`S`, 8 or 16 lanes), K = 16 for 16-bit inputs and 32 for 8-bit
//! ones, f32 (or i32) accumulation. Lane `l` of a fragment holds, per row `m` (vector component):
//!
//! - A (8 x K): S = 8, an `int` of the K / 8 inputs at columns `l * K / 8 ..`; S = 16, a `short`
//!   of the K / 16 at `l * K / 16 ..` (VNNI along K);
//! - B (K x S): column `l`, its K inputs packed as `int`s of 32 / bits consecutive rows (VNNI);
//! - C / D (8 x S): column `l`, one value per row; 16-bit accumulators are kept as `float`.
//!
//! Loads follow CUDA's `load_matrix_sync` contract (pointer and stride aligned to 16 bytes), so
//! rows are read with subgroup block reads.

use cubecl_core::{
    cmma::{MatrixIdent, MatrixLayout, MatrixShape, MatrixType},
    ir::{
        AddressSpace,
        dialect::matrix::{CastOp, ElementwiseOp, FillOp, LoadOp, MultiplyAccumulateOp, StoreOp},
        interfaces::{TypeExt, TypedExt},
        prelude::*,
        types::PointerType,
    },
};
use itertools::Itertools;

use crate::{
    error::{CompileError, EmissionErrors},
    opencl::{opencl_op, ty::opencl_ty},
    shared::{
        CompilationOptions, CppValue, DeclareMatrixOp,
        ty::{TypeExtCPP, TypeToCPP},
        wmma_api_base::{self, matrix_ty},
    },
    target::OpenCl,
};

const ROWS: usize = 8;

/// Records that the kernel needs a fragment DPAS lacks (the compilation fails once emission ends); the
/// emission goes on with `fallback`.
fn unsupported<T>(ctx: &Context, what: String, fallback: T) -> T {
    ctx.aux_ty::<EmissionErrors>()
        .record(CompileError::UnsupportedOp(what));
    fallback
}

fn lanes(ctx: &Context) -> usize {
    ctx.aux_ty::<CompilationOptions>()
        .opencl
        .subgroup_size
        .unwrap_or(8) as usize
}

/// The element kind: bits and signedness (f = float).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Elem {
    F16,
    F32,
    I8,
    U8,
    I32,
}

fn elem(ctx: &Context, ty: &MatrixType) -> Elem {
    let elem = ty.elem_ty;
    if elem.is_float16(ctx) {
        Elem::F16
    } else if elem.is_float32(ctx) {
        Elem::F32
    } else if elem.size(ctx) == 1 {
        if elem.is_signed_int(ctx) {
            Elem::I8
        } else {
            Elem::U8
        }
    } else if elem.size(ctx) == 4 && elem.is_int(ctx) {
        Elem::I32
    } else {
        unsupported(
            ctx,
            format!("DPAS fragment of {}", elem.to_cpp(ctx)),
            Elem::F32,
        )
    }
}

/// The `OpenCL` vector a lane keeps its part of a fragment in.
fn fragment(ctx: &Context, ty: &MatrixType) -> &'static str {
    match (ty.ident, elem(ctx, ty), lanes(ctx)) {
        (MatrixIdent::Accumulator, Elem::I32, _) => "int8",
        (MatrixIdent::Accumulator, _, _) => "float8",
        // unsigned 8-bit operands are unsigned vectors (`intel_sub_group_u8_*`)
        (MatrixIdent::A, Elem::U8, 16) => "ushort8",
        (MatrixIdent::A, _, 16) => "short8",
        (_, Elem::U8, _) => "uint8",
        _ => "int8",
    }
}

/// K of a tile of `kind` inputs: 32 bytes along K.
fn k_native(kind: Elem) -> usize {
    match kind {
        Elem::I8 | Elem::U8 => 32,
        _ => 16,
    }
}

/// The scalar of a fragment's vector (`int8` -> `int`).
fn fragment_scalar(ctx: &Context, ty: &MatrixType) -> &'static str {
    fragment(ctx, ty).trim_end_matches('8')
}

opencl_ty!(MatrixType, |ty, ctx| {
    let MatrixShape { m, n, k } = ty.shape;
    let s = lanes(ctx);
    let k_native = k_native(elem(ctx, ty));
    if !ctx.aux_ty::<CompilationOptions>().opencl.dpas {
        return unsupported(
            ctx,
            "cooperative matrices without DPAS".into(),
            String::new(),
        );
    }
    // Every fragment is declared with its type: the one place to refuse what DPAS lacks.
    let supported = match ty.ident {
        MatrixIdent::Accumulator => matches!(elem(ctx, ty), Elem::F32 | Elem::F16 | Elem::I32),
        _ => matches!(elem(ctx, ty), Elem::F16 | Elem::I8 | Elem::U8),
    };
    if !supported {
        let what = format!("{:?} fragment of {}", ty.ident, ty.elem_ty.to_cpp(ctx));
        return unsupported(ctx, what, String::new());
    }
    if m != ROWS || n != s || (ty.ident != MatrixIdent::Accumulator && k != k_native) {
        let what = format!("{m}x{n}x{k} fragments (DPAS tiles are {ROWS}x{s}x{k_native} here)");
        return unsupported(ctx, what, String::new());
    }
    fragment(ctx, ty).into()
});

opencl_op!(DeclareMatrixOp, |op, ctx| {
    wmma_api_base::compile_matrix_declaration(
        ctx,
        op.get_result(ctx),
        op.value_ty(ctx).get_type(ctx),
    ) + "\n"
});

opencl_op!(FillOp, |op, ctx| {
    let mat = op.matrix(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let ty = matrix_ty(ctx, op.matrix(ctx));
    let frag = fragment(ctx, &ty);
    let fill = match (ty.ident, elem(ctx, &ty), lanes(ctx)) {
        (MatrixIdent::Accumulator, Elem::I32, _) => format!("(int8)({value})"),
        (MatrixIdent::Accumulator, _, _) => format!("(float8)((float)({value}))"),
        (MatrixIdent::A, Elem::F16, 16) => format!("(short8)(as_short({value}))"),
        (MatrixIdent::A, Elem::I8 | Elem::U8, 16) => {
            let scalar = fragment_scalar(ctx, &ty);
            format!("({frag})(as_{scalar}((char2)({value}, {value})))")
        }
        (_, Elem::F16, _) => format!("(int8)(as_int((half2)({value}, {value})))"),
        (_, Elem::I8 | Elem::U8, _) => {
            let scalar = fragment_scalar(ctx, &ty);
            format!("({frag})(as_{scalar}((char4)({value}, {value}, {value}, {value})))")
        }
        _ => unsupported(ctx, format!("{frag} fill of this type"), String::new()),
    };
    format!("*{mat} = {fill};\n")
});

/// The pointer `value` as a pointer to its scalars, with its address space.
fn scalar_ptr(ctx: &Context, value: Value, scalar: &str) -> (String, AddressSpace) {
    let PointerType { address_space, .. } = value.get_type(ctx).as_ptr(ctx);
    let space = crate::opencl::ty::ptr_space(address_space);
    (
        format!("(({space} {scalar}*)({}))", value.name(ctx)),
        address_space,
    )
}

fn components(f: impl Fn(usize) -> String) -> String {
    (0..ROWS).map(f).join(", ")
}

/// The value of `value` when a constant defines it.
fn constant(ctx: &Context, value: Value) -> Option<u64> {
    let op = value.defining_op()?;
    let constant = op.as_op::<pliron::builtin::ops::ConstantOp>(ctx)?;
    let attr = constant.get_attr_builtin_constant_value(ctx)?;
    let attr = pliron::attribute::attr_cast::<dyn crate::shared::CppConstantAttr>(&**attr)?;
    Some(attr.as_f64(ctx) as u64)
}

const LANE: &str = "get_sub_group_local_id()";

/// Where a fragment loads from: its scalars' pointer and address space, and the row stride (in
/// scalars), a constant when `packed` is known.
struct Source {
    p: String,
    space: AddressSpace,
    stride: String,
    packed: Option<u64>,
}

impl Source {
    /// A block read of `kind` (`uint`, `ushort` or `uchar`) at row `row`.
    fn row(&self, kind: &str, row: usize) -> String {
        let suffix = match kind {
            "uint" => "",
            "ushort" => "_us",
            _ => "_uc",
        };
        let space = crate::opencl::ty::ptr_space(self.space);
        let (p, stride) = (&self.p, &self.stride);
        format!(
            "intel_sub_group_block_read{suffix}(({space} const {kind}*)({p} + {row} * {stride}))"
        )
    }

    /// A block read of the whole tile, rows packed: eight `kind`s per lane.
    fn tile(&self, kind: &str) -> String {
        let (space, p) = (crate::opencl::ty::ptr_space(self.space), &self.p);
        let suffix = if kind == "uint" { "8" } else { "_us8" };
        format!("intel_sub_group_block_read{suffix}(({space} const {kind}*)({p}))")
    }
}

opencl_op!(LoadOp, |op, ctx| {
    let ty = matrix_ty(ctx, op.matrix(ctx));
    let kind = elem(ctx, &ty);
    let scalar = match kind {
        Elem::F16 => "ushort",
        Elem::F32 => "float",
        Elem::I8 | Elem::U8 => "uchar",
        Elem::I32 => "int",
    };
    let (p, space) = scalar_ptr(ctx, op.source(ctx), scalar);
    if space == AddressSpace::Local {
        // Block reads take global or local memory.
        return unsupported(
            ctx,
            "fragment load from private memory".into(),
            String::new(),
        );
    }
    let source = Source {
        p,
        space,
        stride: op.stride(ctx).name(ctx).to_string(),
        packed: constant(ctx, op.stride(ctx)),
    };
    let value = match ty.ident {
        MatrixIdent::A => load_a(ctx, &ty, kind, &source),
        MatrixIdent::B => load_b(ctx, &ty, kind, &source),
        MatrixIdent::Accumulator => load_accumulator(ctx, &ty, kind, &source, op.layout(ctx).0),
    };
    format!("*{} = {value};\n", op.matrix(ctx).name(ctx))
});

fn load_a(ctx: &Context, ty: &MatrixType, kind: Elem, source: &Source) -> String {
    let (frag, scalar) = (fragment(ctx, ty), fragment_scalar(ctx, ty));
    let wide = lanes(ctx) == 8;
    let Source { p, stride, .. } = source;
    if !matches!(ty.layout, MatrixLayout::ColMajor) {
        // Rows packed (stride = K): the whole tile in one block read.
        if source.packed == Some(k_native(kind) as u64) {
            return format!(
                "as_{frag}({})",
                source.tile(if wide { "uint" } else { "ushort" })
            );
        }
        // Row m: its K inputs, one block read (S = 8: an int per lane; S = 16: a short).
        let read = if wide { "uint" } else { "ushort" };
        let rows = components(|m| format!("as_{scalar}({})", source.row(read, m)));
        return format!("({frag})({rows})");
    }
    // Column-major: lane l gathers its columns, eight rows each.
    let columns = |count: usize, vector: &str| {
        (0..count)
            .map(|c| {
                format!("{vector} c{c} = vload8(0, {p} + ({count} * {LANE} + {c}) * {stride});")
            })
            .join(" ")
    };
    match (kind, wide) {
        (Elem::F16, true) => format!(
            "({{ {} (int8)({}); }})",
            columns(2, "ushort8"),
            components(|m| format!("as_int((ushort2)(c0.s{m}, c1.s{m}))"))
        ),
        (Elem::F16, false) => format!("as_short8(vload8(0, {p} + {LANE} * {stride}))"),
        (_, true) => format!(
            "({{ {} ({frag})({}); }})",
            columns(4, "uchar8"),
            components(|m| format!("as_{scalar}((uchar4)(c0.s{m}, c1.s{m}, c2.s{m}, c3.s{m}))"))
        ),
        (_, false) => format!(
            "({{ {} ({frag})({}); }})",
            columns(2, "uchar8"),
            components(|m| format!("as_{scalar}((uchar2)(c0.s{m}, c1.s{m}))"))
        ),
    }
}

fn load_b(ctx: &Context, ty: &MatrixType, kind: Elem, source: &Source) -> String {
    let (frag, scalar) = (fragment(ctx, ty), fragment_scalar(ctx, ty));
    let Source {
        p, space, stride, ..
    } = source;
    if matches!(ty.layout, MatrixLayout::ColMajor) {
        // Column-major: lane l's column is contiguous, already VNNI.
        let space = crate::opencl::ty::ptr_space(*space);
        return format!("as_{frag}(vload8(0, (({space} const uint*)({p} + {LANE} * {stride}))))");
    }
    // Row-major: one block read per K row, packed in pairs (16-bit) or fours (8-bit).
    let (read, per_int) = match kind {
        Elem::F16 => ("ushort", 2),
        _ => ("uchar", 4),
    };
    let rows = (0..8 * per_int)
        .map(|k| format!("{read} r{k} = {};", source.row(read, k)))
        .join(" ");
    let packed = components(|j| {
        let parts = (0..per_int)
            .map(|i| format!("r{}", per_int * j + i))
            .join(", ");
        match per_int {
            2 => format!("as_int((ushort2)({parts}))"),
            _ => format!("as_{scalar}((uchar4)({parts}))"),
        }
    });
    let frag = if per_int == 2 { "int8" } else { frag };
    format!("({{ {rows} ({frag})({packed}); }})")
}

fn load_accumulator(
    ctx: &Context,
    ty: &MatrixType,
    kind: Elem,
    source: &Source,
    layout: MatrixLayout,
) -> String {
    let frag = fragment(ctx, ty);
    let Source {
        p, space, stride, ..
    } = source;
    if matches!(layout, MatrixLayout::ColMajor) {
        return match kind {
            Elem::F16 => {
                let space = crate::opencl::ty::ptr_space(*space);
                format!("vload_half8(0, (({space} const half*){p}) + {LANE} * {stride})")
            }
            _ => format!("vload8(0, {p} + {LANE} * {stride})"),
        };
    }
    // Rows packed (stride = the tile's width): the whole tile in one block read.
    if kind != Elem::F16 && source.packed == Some(lanes(ctx) as u64) {
        return format!("as_{frag}({})", source.tile("uint"));
    }
    let rows = components(|m| match kind {
        Elem::F16 => format!("convert_float(as_half({}))", source.row("ushort", m)),
        Elem::I32 => format!("as_int({})", source.row("uint", m)),
        _ => format!("as_float({})", source.row("uint", m)),
    });
    format!("({frag})({rows})")
}

opencl_op!(StoreOp, |op, ctx| {
    let mat = op.matrix(ctx).name(ctx);
    let stride = op.stride(ctx).name(ctx);
    let ty = matrix_ty(ctx, op.matrix(ctx));
    if ty.ident != MatrixIdent::Accumulator {
        return unsupported(ctx, "store of an A or B fragment".into(), String::new());
    }
    let kind = elem(ctx, &ty);
    // Accumulators are f32, f16 or i32 (checked where they are declared).
    let scalar = match kind {
        Elem::F16 => "half",
        Elem::I32 => "int",
        _ => "float",
    };
    let (p, space) = scalar_ptr(ctx, op.destination(ctx), scalar);
    let row_major = !matches!(op.layout(ctx).0, MatrixLayout::ColMajor);
    if row_major
        && space == AddressSpace::Shared
        && kind != Elem::F16
        && constant(ctx, op.stride(ctx)) == Some(lanes(ctx) as u64)
    {
        // Rows packed (stride = the tile's width): the whole tile in one block write.
        return format!("intel_sub_group_block_write8((__local uint*)({p}), as_uint8(*{mat}));\n");
    }
    if row_major {
        // Row m across the lanes: a block write in local memory, coalesced stores in global.
        (0..ROWS)
            .map(|m| {
                let value = match kind {
                    Elem::F16 => format!("convert_half_rte({mat}->s{m})"),
                    _ => format!("{mat}->s{m}"),
                };
                match (space, kind) {
                    (AddressSpace::Shared, Elem::F16) => format!(
                        "intel_sub_group_block_write_us((__local ushort*)({p} + {m} * {stride}), as_ushort({value}));\n"
                    ),
                    (AddressSpace::Shared, _) => format!(
                        "intel_sub_group_block_write((__local uint*)({p} + {m} * {stride}), as_uint({value}));\n"
                    ),
                    _ => format!("{p}[{m} * {stride} + {LANE}] = {value};\n"),
                }
            })
            .join("")
    } else {
        match kind {
            Elem::F16 => format!("vstore_half8_rte(*{mat}, 0, {p} + {LANE} * {stride});\n"),
            _ => format!("vstore8(*{mat}, 0, {p} + {LANE} * {stride});\n"),
        }
    }
});

// The closure on every value of an accumulator, in place: lane l holds column l, its rows the
// vector's components (16-bit accumulators kept as `float`).
opencl_op!(ElementwiseOp, |op, ctx| {
    let (input, output) = (op.matrix_in(ctx), op.matrix_out(ctx));
    let ty = matrix_ty(ctx, input);
    if ty.ident != MatrixIdent::Accumulator {
        return unsupported(
            ctx,
            "elementwise op on an A or B fragment".into(),
            String::new(),
        );
    }
    let (narrow, widen) = match elem(ctx, &ty) {
        Elem::F16 => ("convert_half", "convert_float"),
        _ => ("", ""),
    };
    let closure = op.closure(ctx);
    let captures = op
        .closure_captures(ctx)
        .iter()
        .map(|capture| format!(", {}", capture.name(ctx)))
        .join("");
    let (input, output) = (input.name(ctx), output.name(ctx));
    (0..ROWS)
        .map(|m| {
            format!(
                "{output}->s{m} = {widen}({closure}({m}, {LANE}, {narrow}({input}->s{m}){captures}));\n"
            )
        })
        .join("")
});

opencl_op!(MultiplyAccumulateOp, |op, ctx| {
    let a = op.mat_a(ctx).name(ctx);
    let b = op.mat_b(ctx).name(ctx);
    let c = op.mat_c(ctx).name(ctx);
    let d = op.mat_d(ctx).name(ctx);
    let a_ty = matrix_ty(ctx, op.mat_a(ctx));
    let b_ty = matrix_ty(ctx, op.mat_b(ctx));
    let d_ty = matrix_ty(ctx, op.mat_d(ctx));
    // Inputs are f16, i8 or u8 (checked where they are declared).
    let name = |e: Elem| match e {
        Elem::I8 => "i8",
        Elem::U8 => "u8",
        _ => "f16",
    };
    let (ea, eb) = (elem(ctx, &a_ty), elem(ctx, &b_ty));
    let k = k_native(ea);
    let mad = format!("intel_sub_group_{}_{}_matrix_mad_k{k}", name(ea), name(eb));
    let result = match elem(ctx, &d_ty) {
        // A 16-bit accumulator rounds as it would be stored.
        Elem::F16 => format!("convert_float8(convert_half8_rte({mad}(*{a}, *{b}, *{c})))"),
        _ => format!("{mad}(*{a}, *{b}, *{c})"),
    };
    format!("*{d} = {result};\n")
});

opencl_op!(CastOp, |op, ctx| {
    let input = op.input(ctx).name(ctx);
    let output = op.output(ctx).name(ctx);
    let (from, to) = (
        matrix_ty(ctx, op.input(ctx)),
        matrix_ty(ctx, op.output(ctx)),
    );
    if from.ident != MatrixIdent::Accumulator || to.ident != MatrixIdent::Accumulator {
        return unsupported(ctx, "cast of an A or B fragment".into(), String::new());
    }
    let value = match (elem(ctx, &from), elem(ctx, &to)) {
        (_, Elem::F16) => format!("convert_float8(convert_half8_rte(*{input}))"),
        (Elem::I32, Elem::F32) => format!("convert_float8(*{input})"),
        // Toward zero, as CUDA's `int(x)` per element.
        (Elem::F32 | Elem::F16, Elem::I32) => format!("convert_int8_sat(*{input})"),
        _ => format!("*{input}"),
    };
    format!("*{output} = {value};\n")
});
