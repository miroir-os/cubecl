//! The device's compiled kernels, behind the one type the server holds.

use crate::compute::artifact::{OpenClArtifactCompiler, OpenClKernel};
use crate::compute::programs::OpenClPrograms;
use cubecl_core::server::LaunchError;
use cubecl_server::compiler::{ArtifactId, KernelLoader};
use cubecl_server::kernel::CubeKernel;
use cubecl_server::logging::ServerLogger;

/// Compiles `CubeCL` IR to `OpenCL` C and on to kernels built for the device, caching both.
#[derive(Debug)]
pub(crate) struct OpenClContext {
    kernels: KernelLoader<OpenClPrograms>,
}

impl OpenClContext {
    /// `namespace` names what the stored sources were compiled for (see [`OpenClPrograms::new`]).
    pub(crate) fn new(compiler: OpenClArtifactCompiler, namespace: String) -> Self {
        Self {
            kernels: KernelLoader::new(OpenClPrograms::new(compiler, namespace)),
        }
    }

    /// The kernel for `kernel`, whose id is `id`, compiling it first when no store holds it.
    pub(crate) fn load(
        &mut self,
        kernel: &dyn CubeKernel,
        id: &ArtifactId<()>,
        logger: &ServerLogger,
    ) -> Result<OpenClKernel, LaunchError> {
        self.kernels.load(kernel, id, logger)
    }

    /// Compiles every queued kernel now, rather than inside the next launch.
    pub(crate) fn compile_queued(&mut self, logger: &ServerLogger) {
        self.kernels.compile_queued(logger);
    }

    /// Queues `kernel` to be compiled with others, by the next kernel loaded for a launch.
    pub(crate) fn queue(&mut self, kernel: Box<dyn CubeKernel>) {
        self.kernels.enqueue(kernel, ());
    }
}
