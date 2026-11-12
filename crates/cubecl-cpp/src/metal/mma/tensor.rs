//! Cooperative matrices larger than the 8x8 simdgroup matrix, on Metal 4's tensor operations
//! (Metal Performance Primitives' `matmul2d`, which Apple10 GPUs run on their neural
//! accelerators).
//!
//! Each fragment is a `cooperative_tensor` of a `matmul2d` that one SIMD group runs: its left
//! input, right input or destination. The operation spreads the cells over the lanes in a way it
//! does not specify, so fragments are loaded and stored through tensor views of memory and
//! combined cell by cell only with fragments of the same operation. A fragment's type names that
//! operation, from the product it takes part in: the operand types, and whether each input is
//! column major, which the operation reads transposed (a view's innermost stride must be 1).

use core::fmt::Display;

use cubecl_core::{
    cmma::{MatrixIdent, MatrixLayout, MatrixShape, MatrixType},
    ir::{
        dialect::matrix::{CastOp, FillOp, LoadOp, MultiplyAccumulateOp, StoreOp},
        interfaces::TypeExt,
        prelude::*,
    },
};

use crate::{
    metal::ty::ptr_space,
    shared::{CppValue, DeclareMatrixOp, ty::TypeExtCPP, wmma_api_base::matrix_ty},
};

/// Whether the fragment shape is a tensor operation's rather than the simdgroup matrix's.
pub fn is_tensor_op(shape: &MatrixShape) -> bool {
    (shape.m, shape.n, shape.k) != (8, 8, 8)
}

/// The product a fragment takes part in: the element types of `A`, `B` and the accumulator, and
/// whether `A` and `B` are column major.
struct Product {
    types: [String; 3],
    transposed: [bool; 2],
}

impl Product {
    /// The product of `ty`'s fragment with no other information: a float accumulator's from
    /// `half` operands, an 8-bit integer operand's into `int32_t`, row major operands except a
    /// column major fragment itself.
    fn natural(ctx: &Context, ty: &MatrixType) -> Self {
        let elem = ty.elem_ty.to_cpp(ctx);
        let [ab, cd] = match (ty.ident, elem.as_str()) {
            (MatrixIdent::Accumulator, "int32_t") => ["int8_t".into(), elem],
            (MatrixIdent::Accumulator, "float") => ["half".into(), elem],
            (MatrixIdent::Accumulator, _) => [elem.clone(), elem],
            (_, "int8_t" | "uint8_t") => [elem, "int32_t".into()],
            (_, _) => [elem, "float".into()],
        };
        let column = ty.layout == MatrixLayout::ColMajor;
        let transposed = match ty.ident {
            MatrixIdent::A => [column, false],
            MatrixIdent::B => [false, column],
            MatrixIdent::Accumulator => [false, false],
        };
        Self {
            types: [ab.clone(), ab, cd],
            transposed,
        }
    }

    /// The product of the operation `matrix` is used in, or for a cast's output its input's
    /// operation, else the natural one.
    fn of(ctx: &Context, matrix: Value) -> Self {
        let ty = matrix_ty(ctx, matrix);
        for user in matrix.uses(ctx).iter().map(|r#use| r#use.user_op()) {
            if let Some(mma) = user.as_op::<MultiplyAccumulateOp>(ctx) {
                let operands = [mma.mat_a(ctx), mma.mat_b(ctx), mma.mat_c(ctx)]
                    .map(|operand| matrix_ty(ctx, operand));
                let column = |ty: &MatrixType| ty.layout == MatrixLayout::ColMajor;
                return Self {
                    types: operands.map(|ty| ty.elem_ty.to_cpp(ctx)),
                    transposed: [column(&operands[0]), column(&operands[1])],
                };
            }
            if let Some(cast) = user.as_op::<CastOp>(ctx)
                && cast.output(ctx) == matrix
            {
                return Self {
                    transposed: Self::of(ctx, cast.input(ctx)).transposed,
                    ..Self::natural(ctx, &ty)
                };
            }
        }
        Self::natural(ctx, &ty)
    }

    /// The `matmul2d` of one SIMD group that computes a `shape` fragment of this product.
    fn operation(&self, shape: &MatrixShape) -> String {
        let MatrixShape { m, n, k } = shape;
        let [left, right] = self.transposed;
        format!(
            "mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor({m}, {n}, {k}, \
             {left}, {right}, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), \
             metal::execution_simdgroup>"
        )
    }

    /// The cooperative tensor type of a `ty` fragment of this product.
    fn fragment(&self, ty: &MatrixType) -> String {
        let op = self.operation(&ty.shape);
        let [a, b, cd] = &self.types;
        let left = format!("{op}::cooperative_tensor_left_input_t<{a}, {b}, {cd}>");
        let right = format!("{op}::cooperative_tensor_right_input_t<{a}, {b}, {cd}>");
        match ty.ident {
            MatrixIdent::A => left,
            MatrixIdent::B => right,
            MatrixIdent::Accumulator => {
                format!("{op}::cooperative_tensor_destination_t<{left}, {right}, {cd}>")
            }
        }
    }
}

/// The fragment type of `ty` in its natural product (see [`Product::natural`]); a declaration
/// names the one of the product the fragment takes part in.
pub fn fragment_ty(ctx: &Context, ty: &MatrixType) -> String {
    Product::natural(ctx, ty).fragment(ty)
}

pub fn declare(ctx: &Context, op: &DeclareMatrixOp) -> String {
    let matrix = op.get_result(ctx);
    let ty = Product::of(ctx, matrix).fragment(&matrix_ty(ctx, matrix));
    let id = matrix.name(ctx);
    format!("{ty} {id}_store;\nthread {ty}* {id} = &{id}_store;\n")
}

pub fn fill(ctx: &Context, op: &FillOp) -> String {
    let elem = matrix_ty(ctx, op.matrix(ctx)).elem_ty.to_cpp(ctx);
    let value = format!("{elem}({})", op.value(ctx).name(ctx));
    cells(op.matrix(ctx).name(ctx), |cell| {
        format!("{cell} = {value};")
    })
}

/// `body` on every cell the lane holds of `matrix`, named by the expression it is given.
fn cells(matrix: impl Display, body: impl Fn(&str) -> String) -> String {
    let body = body(&format!("(*{matrix})[i]"));
    format!("for (uint16_t i = 0; i < {matrix}->get_capacity(); ++i) {{ {body} }}\n")
}

/// A fragment's cells in memory, at `ptr` with `stride` elements between the starts of its rows
/// (row major) or columns (column major).
enum Memory {
    /// A tensor view of them: an `A` or `B` fragment's (a column major one transposed), a row
    /// major accumulator's.
    View(String),
    /// The cell at the multidimensional index `at` of a column major accumulator, which no view
    /// lays out with an innermost stride of 1.
    Cell(String),
}

impl Memory {
    fn new(
        ctx: &Context,
        ty: &MatrixType,
        ptr: Value,
        stride: Value,
        layout: MatrixLayout,
    ) -> Self {
        let MatrixShape { m, n, k } = ty.shape;
        let (rows, columns) = match ty.ident {
            MatrixIdent::A => (m, k),
            MatrixIdent::B => (k, n),
            MatrixIdent::Accumulator => (m, n),
        };
        let space = ptr_space(ptr.get_type(ctx).as_ptr(ctx).address_space);
        let elem = ty.elem_ty.to_cpp(ctx);
        let stride = stride.name(ctx);
        let ptr = format!("(({space} {elem}*)({}))", ptr.name(ctx));
        // innermost extent first
        let extents = match (ty.ident, layout) {
            (MatrixIdent::Accumulator, MatrixLayout::ColMajor) => {
                return Self::Cell(format!("{ptr}[at[0] * int({stride}) + at[1]]"));
            }
            (_, MatrixLayout::ColMajor) => format!("metal::extents<int32_t, {rows}, {columns}>"),
            _ => format!("metal::extents<int32_t, {columns}, {rows}>"),
        };
        Self::View(format!(
            "metal::tensor<{space} {elem}, {extents}, metal::tensor_inline>({ptr}, {extents}(), \
             metal::array<int32_t, 2>{{1, int({stride})}})"
        ))
    }

    /// `view` with the view, else `cell` on every valid cell and its place in memory.
    fn access(
        self,
        matrix: impl Display,
        view: impl Fn(&str) -> String,
        cell: impl Fn(&str, &str) -> String,
    ) -> String {
        match self {
            Self::View(memory) => view(&memory),
            Self::Cell(memory) => cells(&matrix, |value| {
                format!(
                    "if ({matrix}->is_valid_element(i)) {{ auto at = \
                     {matrix}->get_multidimensional_index(i); {} }}",
                    cell(value, &memory)
                )
            }),
        }
    }
}

pub fn load(ctx: &Context, op: &LoadOp) -> String {
    let matrix = op.matrix(ctx);
    let ty = matrix_ty(ctx, matrix);
    let memory = Memory::new(ctx, &ty, op.source(ctx), op.stride(ctx), op.layout(ctx).0);
    let name = matrix.name(ctx);
    memory.access(
        &name,
        |view| format!("{name}->load({view});\n"),
        |value, cell| format!("{value} = {cell};"),
    )
}

pub fn store(ctx: &Context, op: &StoreOp) -> String {
    let matrix = op.matrix(ctx);
    let ty = matrix_ty(ctx, matrix);
    let memory = Memory::new(
        ctx,
        &ty,
        op.destination(ctx),
        op.stride(ctx),
        op.layout(ctx).0,
    );
    let name = matrix.name(ctx);
    memory.access(
        &name,
        |view| format!("{name}->store({view});\n"),
        |value, cell| format!("{cell} = {value};"),
    )
}

/// `d = a b + c`: the operation accumulates into its destination, so `c` is copied to `d` first
/// when they differ, cell by cell (the cooperative tensor's copy assignment does not compile).
pub fn execute(ctx: &Context, op: &MultiplyAccumulateOp) -> String {
    let [a, b, c, d] = [op.mat_a(ctx), op.mat_b(ctx), op.mat_c(ctx), op.mat_d(ctx)];
    let operation = Product::of(ctx, d).operation(&matrix_ty(ctx, d).shape);
    let copy = match c == d {
        true => String::new(),
        false => copy(c.name(ctx), d.name(ctx), ""),
    };
    let [a, b, d] = [a, b, d].map(|matrix| matrix.name(ctx));
    format!("{copy}{{ {operation} op; op.run(*{a}, *{b}, *{d}); }}\n")
}

/// Cell by cell: the output is a destination of its input's operation, which spreads cells over
/// lanes the same way whatever their type.
pub fn cast(ctx: &Context, op: &CastOp) -> String {
    let elem = matrix_ty(ctx, op.output(ctx)).elem_ty.to_cpp(ctx);
    copy(op.input(ctx).name(ctx), op.output(ctx).name(ctx), &elem)
}

/// Every cell of `from` into `to`, through `convert` (a type, or nothing).
fn copy(from: impl Display, to: impl Display, convert: &str) -> String {
    cells(to, |cell| format!("{cell} = {convert}((*{from})[i]);"))
}
