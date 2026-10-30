//! The half of `OpenCL`'s compilation any thread can do: from a kernel to the `OpenCL` C, and the
//! program and kernel the driver builds from it.

use crate::OpenClCompiler;
use crate::compute::cl;
use cubecl_core::server::LaunchError;
use cubecl_core::{ir::DeviceProperties, prelude::*};
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::compiler::{
    ArtifactCompiler, ArtifactId, CompilationError, CompilationRecording,
};
use cubecl_server::kernel::{BufferIOAttr, CompiledKernel, CubeKernel, DebugInformation};
use cubecl_server::logging::ServerLogger;
use cubecl_server::validation::{validate_cube_dim, validate_shared_memory, validate_units};
use opencl_sys::{
    CL_KERNEL_SPILL_MEM_SIZE_INTEL, CL_KERNEL_WORK_GROUP_SIZE, CL_PROGRAM_BUILD_LOG, cl_context,
    cl_device_id, cl_kernel, cl_kernel_work_group_info,
};
use std::ffi::CString;
use std::sync::Arc;

/// Spill (bytes per thread) beyond which a kernel without cooperative matrices is rebuilt with more
/// registers per thread: a little spill costs less than the threads the larger register file takes.
const SPILL_LIMIT: u64 = 4096;

/// The build options every kernel gets: `OpenCL` C 3.0, and unsuffixed literals (the shared emission
/// writes them as doubles) in single precision. That narrows `double` constants too, harmless while
/// the runtime registers no `f64`.
const OPTIONS: &str = "-cl-std=CL3.0 -cl-single-precision-constant";

/// A kernel object, released with the last clone.
#[derive(Debug)]
pub(crate) struct KernelHandle(cl_kernel);

impl KernelHandle {
    pub(crate) fn raw(&self) -> cl_kernel {
        self.0
    }
}

// SAFETY: `OpenCL` kernel objects are thread-safe handles; only the server thread sets their
// arguments and enqueues them, one launch at a time.
unsafe impl Send for KernelHandle {}
// SAFETY: as above.
unsafe impl Sync for KernelHandle {}

impl Drop for KernelHandle {
    fn drop(&mut self) {
        // The reference this handle owns.
        cl::api().clReleaseKernel(self.0);
    }
}

/// A kernel ready to launch.
#[derive(Debug, Clone)]
pub(crate) struct OpenClKernel {
    pub kernel: Arc<KernelHandle>,
    pub cube_dim: CubeDim,
    pub io: Option<Arc<[BufferIOAttr]>>,
    /// After its buffers, the kernel takes the dynamic metadata (a buffer) when it has one, then the
    /// info struct (by value) when it has one.
    pub has_dynamic_meta: bool,
    pub has_info: bool,
}

/// `OpenCL` C ready for the driver to build, with what launching it needs: what the compilation
/// store keeps for a kernel.
#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub(crate) struct OpenClCacheEntry {
    entrypoint_name: String,
    cube_dim: (u32, u32, u32),
    source: String,
    io: Option<Vec<BufferIOAttr>>,
    has_dynamic_meta: bool,
    has_info: bool,
    uses_mma: bool,
}

/// `OpenCL` C, and the kernel finalizing built from it.
#[derive(Debug)]
pub(crate) struct OpenClArtifact {
    /// What the store keeps.
    pub entry: OpenClCacheEntry,
    /// `None` for a source read from the store, which is built when it is loaded.
    pub kernel: Option<OpenClKernel>,
}

/// Compiles kernels to `OpenCL` C for one device, and has the driver build them.
///
/// Shared by every thread compiling at once: `OpenCL` contexts, programs and kernel creation are
/// thread-safe (setting a kernel's arguments is not, and only the server thread does).
#[derive(Debug)]
pub(crate) struct OpenClArtifactCompiler {
    context: cl_context,
    device: cl_device_id,
    properties: DeviceProperties,
    options: cubecl_cpp::shared::CompilationOptions,
    build_options: CString,
    /// Intel GPUs: the options of a kernel rebuilt where the default register file spills (256
    /// registers per thread instead of 128, half the threads per core).
    large_registers: Option<CString>,
}

// SAFETY: the context and device are thread-safe `OpenCL` handles, the context retained by this
// value.
unsafe impl Send for OpenClArtifactCompiler {}
// SAFETY: as above; `&self` only creates programs and kernels.
unsafe impl Sync for OpenClArtifactCompiler {}

impl Drop for OpenClArtifactCompiler {
    fn drop(&mut self) {
        cl::api().clReleaseContext(self.context);
    }
}

impl OpenClArtifactCompiler {
    pub(crate) fn new(
        context: cl_context,
        device: cl_device_id,
        properties: DeviceProperties,
        options: cubecl_cpp::shared::CompilationOptions,
        large_registers: bool,
    ) -> Self {
        // A reference of the compiler's own: it may outlive the server on a compiling thread.
        cl::api().clRetainContext(context);
        Self {
            context,
            device,
            properties,
            options,
            build_options: CString::new(OPTIONS).expect("no interior NUL"),
            large_registers: large_registers.then(|| {
                CString::new(format!("{OPTIONS} -cl-intel-256-GRF-per-thread"))
                    .expect("no interior NUL")
            }),
        }
    }

    /// Has the driver build `entry`'s program and kernel.
    pub(crate) fn load(&self, entry: &OpenClCacheEntry) -> Result<OpenClKernel, CompilationError> {
        let cube_dim = entry.cube_dim.into();
        Ok(OpenClKernel {
            kernel: self.build(
                &entry.source,
                &entry.entrypoint_name,
                cube_dim,
                entry.uses_mma,
            )?,
            cube_dim,
            io: entry.io.clone().map(Arc::from),
            has_dynamic_meta: entry.has_dynamic_meta,
            has_info: entry.has_info,
        })
    }
}

impl ArtifactCompiler for OpenClArtifactCompiler {
    type Variant = ();
    type Lowered = CompiledKernel<OpenClCompiler>;
    type Artifact = OpenClArtifact;

    fn lower(
        &self,
        kernel: &dyn CubeKernel,
        id: &ArtifactId<()>,
        recording: &mut CompilationRecording,
        logger: &ServerLogger,
    ) -> Result<Self::Lowered, LaunchError> {
        validate_cube_dim(&self.properties, &id.kernel)?;
        validate_units(&self.properties, &id.kernel)?;

        let mut lowered = CompiledKernel::lower(kernel, recording, |definition| {
            CompiledKernel::compile(
                kernel,
                definition,
                &mut OpenClCompiler::default(),
                &self.options,
            )
        })?;

        if logger.compilation_source_activated() {
            lowered.debug_info = Some(DebugInformation::new("opencl", id.kernel.clone()));
        }
        logger.log_compilation(&lowered);

        // Checked before the driver builds the program: it would refuse the kernel there anyway,
        // with a build error instead of a resource limit error.
        validate_shared_memory(
            &self.properties,
            lowered.repr.as_ref().map(|repr| repr.shared_memory_size),
        )?;

        Ok(lowered)
    }

    /// Has the driver build the program and kernel too: most of what an `OpenCL` kernel costs.
    fn finalize(
        &self,
        _id: &ArtifactId<()>,
        mut lowered: Self::Lowered,
    ) -> Result<OpenClArtifact, LaunchError> {
        let repr = lowered
            .repr
            .as_ref()
            .ok_or_else(|| CompilationError::Generic {
                reason: "the OpenCL compiler returned no kernel representation".into(),
                backtrace: BackTrace::capture(),
            })?;
        let (has_dynamic_meta, has_info, uses_mma) =
            (repr.has_dynamic_meta, repr.has_info, repr.uses_mma);
        let cube_dim = lowered.cube_dim;
        let entry = OpenClCacheEntry {
            io: lowered.io.take(),
            entrypoint_name: lowered.entrypoint_name,
            cube_dim: (cube_dim.x, cube_dim.y, cube_dim.z),
            source: lowered.source,
            has_dynamic_meta,
            has_info,
            uses_mma,
        };
        let kernel = self.load(&entry)?;
        Ok(OpenClArtifact {
            entry,
            kernel: Some(kernel),
        })
    }
}

impl OpenClArtifactCompiler {
    /// Builds `source` for this compiler's device and takes its kernel `entrypoint_name`: on Intel GPUs with
    /// 256 registers per thread where 128 spill (any spill for a kernel holding cooperative matrices, whose
    /// accumulators live in registers; past [`SPILL_LIMIT`] for the others) and the larger build still runs
    /// `cube_dim`'s units. A kernel that cannot run that many is refused at launch.
    fn build(
        &self,
        source: &str,
        entrypoint_name: &str,
        cube_dim: CubeDim,
        uses_mma: bool,
    ) -> Result<Arc<KernelHandle>, CompilationError> {
        let kernel = self.build_with(source, entrypoint_name, &self.build_options)?;
        let limit = if uses_mma { 0 } else { SPILL_LIMIT };
        let spill = self.spill_bytes(&kernel);
        let units = cube_dim.num_elems();
        let kernel = self
            .wider(source, entrypoint_name, spill, limit, units)
            .unwrap_or(kernel);
        Ok(Arc::new(kernel))
    }

    /// The kernel built with the larger register file (Intel) when `spill` bytes are past `limit`,
    /// if it spills less and still runs `units` units.
    fn wider(
        &self,
        source: &str,
        entrypoint_name: &str,
        spill: u64,
        limit: u64,
        units: u32,
    ) -> Option<KernelHandle> {
        let options = self.large_registers.as_ref().filter(|_| spill > limit)?;
        let wide = self.build_with(source, entrypoint_name, options).ok()?;
        (self.spill_bytes(&wide) < spill && self.max_units(&wide) >= units).then_some(wide)
    }

    fn work_group_info(
        &self,
        kernel: &KernelHandle,
        param: cl_kernel_work_group_info,
    ) -> Option<u64> {
        let bytes = cl3::kernel::get_kernel_work_group_data(kernel.0, self.device, param).ok()?;
        Some(crate::device::word(&bytes))
    }

    /// The register spill of `kernel` (Intel), 0 where the device does not say.
    fn spill_bytes(&self, kernel: &KernelHandle) -> u64 {
        self.work_group_info(
            kernel,
            CL_KERNEL_SPILL_MEM_SIZE_INTEL as cl_kernel_work_group_info,
        )
        .unwrap_or(0)
    }

    /// The most work-items a work-group of `kernel` may have on this device.
    fn max_units(&self, kernel: &KernelHandle) -> u32 {
        self.work_group_info(kernel, CL_KERNEL_WORK_GROUP_SIZE)
            .map_or(u32::MAX, |units| units.min(u32::MAX as u64) as u32)
    }

    fn build_with(
        &self,
        source: &str,
        entrypoint_name: &str,
        options: &CString,
    ) -> Result<KernelHandle, CompilationError> {
        let generic = |reason: String| CompilationError::Generic {
            reason,
            backtrace: BackTrace::capture(),
        };
        let program = cl3::program::create_program_with_source(self.context, &[source])
            .map_err(|code| generic(format!("clCreateProgramWithSource failed ({code})")))?;
        let built = cl3::program::build_program(
            program,
            &[self.device],
            options,
            None,
            std::ptr::null_mut(),
        );
        let result = match built {
            Err(code) => {
                let log = cl3::program::get_program_build_data(
                    program,
                    self.device,
                    CL_PROGRAM_BUILD_LOG,
                )
                .map(|bytes| {
                    String::from_utf8_lossy(&bytes)
                        .trim_end_matches('\0')
                        .to_string()
                })
                .unwrap_or_default();
                Err(generic(format!(
                    "OpenCL build of {entrypoint_name} failed ({code}):\n{log}\n{source}"
                )))
            }
            Ok(()) => CString::new(entrypoint_name)
                .map_err(|_| generic(format!("kernel name {entrypoint_name:?} holds a NUL")))
                .and_then(|name| {
                    cl3::kernel::create_kernel(program, &name)
                        .map(KernelHandle)
                        .map_err(|code| {
                            generic(format!("clCreateKernel({entrypoint_name}) failed ({code})"))
                        })
                }),
        };
        // The kernel holds its own reference to the program.
        crate::compute::cl::api().clReleaseProgram(program);
        result
    }
}
