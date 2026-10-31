use crate::compute::artifact::OpenClKernel as CompiledKernel;
use crate::compute::cl::{self, ClError, Event};
use crate::compute::programs::OpenClPrograms;
use crate::compute::stream::{OpenClStream, OpenClStreamBackend};
use crate::memory::{OpenClResource, OpenClStorage};
use cubecl_common::bytes::Bytes;
use cubecl_common::profile::{ProfileDuration, ProfileTicks};
use cubecl_core::server::ServerStorage;
use cubecl_core::{
    prelude::*,
    server::{
        BufferBinding, CopyDescriptor, IoError, KernelArguments, KernelResource, LaunchError,
        ProfileError, ProfilingToken, ResourceLimitError, ServerCommunication, ServerError,
        ServerUtilities,
    },
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future::DynFut;
use cubecl_environment::stream::StreamId;
use cubecl_server::command::CopyLayout;
use cubecl_server::compiler::{ArtifactId, KernelLoader};
use cubecl_server::memory_management::{Cleanup, PageUpdate};
use cubecl_server::{
    execution::LaunchMode,
    kernel::{BufferIOAttr, CubeKernel},
    logging::ServerLogger,
    memory_management::{
        ManagedMemoryHandle,
        relocation::{RelocatingStreams, RelocationNeed, RelocationReason},
    },
    server::Server,
    storage::{ComputeStorage, ManagedResource},
    stream::{
        EventStreamBackend, ExecuteScope, FailureStore, MultiStream, ResolvedStreams, WriteScoped,
        failed_writing,
    },
    timestamp_profiler::TimestampProfiler,
};
use opencl_sys::{
    CL_INVALID_WORK_GROUP_SIZE, CL_MEM_COPY_HOST_PTR, CL_MEM_READ_ONLY, CL_PROFILING_COMMAND_END,
    CL_PROFILING_COMMAND_START, cl_context, cl_mem,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// `OpenCL` compute server.
#[derive(Debug)]
pub struct OpenClServer {
    /// Compiles `CubeCL` IR to `OpenCL` C and on to kernels built for the device, caching both.
    kernels: KernelLoader<OpenClPrograms>,
    streams: MultiStream<OpenClStreamBackend>,
    pub(crate) utilities: Arc<ServerUtilities>,
    timestamps: TimestampProfiler,
    /// Open profiling windows, by token.
    windows: HashMap<ProfilingToken, Window>,
    cl_context: cl_context,
}

// SAFETY: the context and the handles the fields hold are thread-safe OpenCL objects; only the server thread
// uses the server.
unsafe impl Send for OpenClServer {}

/// An open profiling window: the launches on its stream from its opening, timed by their events.
#[derive(Debug)]
struct Window {
    stream: StreamId,
    first: Option<Event>,
    last: Option<Event>,
    failure: Option<ServerError>,
}

impl OpenClServer {
    /// A server on `cl_context`, whose reference it takes over.
    pub(crate) fn new(
        cl_context: cl_context,
        kernels: KernelLoader<OpenClPrograms>,
        backend: OpenClStreamBackend,
        utilities: Arc<ServerUtilities>,
    ) -> Self {
        let logger = utilities.logger.clone();
        let max_streams = {
            use cubecl_server::config::RuntimeConfig;
            cubecl_server::config::CubeClRuntimeConfig::get()
                .streaming
                .max_streams
        };
        Self {
            kernels,
            streams: MultiStream::new(logger, backend, max_streams),
            utilities,
            timestamps: TimestampProfiler::default(),
            windows: HashMap::new(),
            cl_context,
        }
    }

    /// Fails the windows open on `stream`: what they measure includes the failed work. (The shared
    /// profiler's `failure` fails every open window, a failure on another stream too.)
    fn profile_failure(&mut self, stream: StreamId, error: &ServerError) {
        for window in self
            .windows
            .values_mut()
            .filter(|window| window.stream == stream)
        {
            window.failure.get_or_insert_with(|| error.clone());
        }
    }

    /// Adds a launch on `stream` to the windows open there: their first and last runs.
    fn timed_launch(&mut self, stream: StreamId, event: Event) {
        for window in self
            .windows
            .values_mut()
            .filter(|window| window.stream == stream)
        {
            window.first.get_or_insert_with(|| event.clone());
            window.last = Some(event.clone());
        }
    }

    /// Fails what `bindings` writes (every buffer, when the kernel's `io` is unknown).
    fn fail_outputs(
        &mut self,
        stream_id: StreamId,
        bindings: &KernelArguments,
        io: Option<&[BufferIOAttr]>,
        err: ServerError,
    ) {
        let mut written = self.write_set();
        written.extend(bindings.buffers_written(io).cloned());
        failed_writing(self, stream_id, written, err);
    }

    /// The grid of a dynamic count, read back from the device (`OpenCL` has no indirect dispatch).
    fn read_count(
        &mut self,
        binding: &BufferBinding,
        stream_id: StreamId,
    ) -> Result<[usize; 3], ServerError> {
        self.streams.ensure_written([binding].into_iter())?;
        let mut resolved = self.streams.resolve(stream_id, [binding].into_iter())?;
        let resource = resolve_origin_resource(&mut resolved, binding)?;
        if resource.size < 12 {
            return Err(IoError::Unknown {
                description: format!("a dynamic cube count of {} bytes", resource.size),
                backtrace: BackTrace::capture(),
            }
            .into());
        }
        let mut groups = [0u32; 3];
        let stream = resolved.current();
        // SAFETY: the binding holds three u32; the copy is blocking.
        unsafe {
            stream.memory.copy(
                stream.queue,
                true,
                groups.as_mut_ptr().cast(),
                resource.at(0).cast_const(),
                12,
            )
        }
        .map_err(ClError::into_sync)?;
        Ok(groups.map(|g| g as usize))
    }
}

impl Drop for OpenClServer {
    fn drop(&mut self) {
        // A kernel on one stream may use another stream's pages: every queue drains before any
        // stream frees its memory.
        let ids: Vec<_> = self.streams.stream_ids().collect();
        for id in ids {
            if let Some(stream) = self.streams.try_stream_mut(&id)
                && let Err(err) = stream.finish()
            {
                log::warn!("{err}");
            }
        }
        // The server's reference; the queues and programs hold their own.
        cl::api().clReleaseContext(self.cl_context);
    }
}

/// A binding's device memory, from its origin stream (each stream owns its memory).
fn resolve_origin_resource(
    resolved: &mut ResolvedStreams<'_, OpenClStreamBackend>,
    binding: &BufferBinding,
) -> Result<OpenClResource, IoError> {
    resolved
        .get(&binding.stream)
        .memory_management
        .get_resource(
            binding.memory.clone(),
            binding.offset_start,
            binding.offset_end,
        )
}

/// The contiguous runs of a pitched row-major tensor in its binding: (byte offset, bytes) of each,
/// checked to lie in the binding's `size` bytes.
fn runs(descriptor: &CopyDescriptor, size: u64) -> Result<Vec<(usize, usize)>, IoError> {
    let total = descriptor.shape.iter().product::<usize>() * descriptor.elem_size;
    if total == 0 {
        return Ok(Vec::new());
    }
    let layout = CopyLayout::of(&descriptor.shape, &descriptor.strides, descriptor.elem_size)?;
    let runs: Vec<_> = match layout.pitch {
        None => vec![(0, total)],
        Some(pitch) => (0..pitch.height)
            .map(|row| (row * pitch.stride_bytes, pitch.width_bytes))
            .collect(),
    };
    let end = runs.last().map_or(0, |(offset, len)| offset + len) as u64;
    if end > size {
        return Err(IoError::Unknown {
            description: format!("a copy of {end} bytes from a {size}-byte binding"),
            backtrace: BackTrace::capture(),
        });
    }
    Ok(runs)
}

/// Sets `compiled`'s arguments: a pointer per buffer, then the dynamic metadata as a buffer of its
/// own (returned, to be released once the launch is enqueued) and the static part by value.
fn set_arguments(
    compiled: &CompiledKernel,
    buffers: &[&BufferBinding],
    bindings: &KernelArguments,
    resolved: &mut ResolvedStreams<'_, OpenClStreamBackend>,
    context: cl_context,
) -> Result<Option<Metadata>, ServerError> {
    let kernel = compiled.kernel.raw();
    let memory = resolved.current().memory;
    let mut index = 0u32;
    for binding in buffers {
        let resource = resolve_origin_resource(resolved, binding)?;
        // SAFETY: the kernel is live and the pointer lies in this context's device memory.
        unsafe { memory.set_arg(kernel, index, resource.at(0).cast_const()) }?;
        index += 1;
    }
    let info = &bindings.info;
    let static_len = info.dynamic_metadata_offset.min(info.data.len());
    let metadata = match compiled.has_dynamic_meta {
        true => {
            let metadata = Metadata::new(context, bytemuck::cast_slice(&info.data[static_len..]))?;
            // SAFETY: the kernel and the buffer are live, in the same context.
            unsafe { cl::set_arg(kernel, index, &metadata.0) }?;
            index += 1;
            Some(metadata)
        }
        false => None,
    };
    if compiled.has_info {
        let bytes: &[u8] = bytemuck::cast_slice(&info.data[..static_len]);
        // SAFETY: the kernel's info struct, by value: the static part of the info is its layout.
        unsafe { cl::set_arg(kernel, index, bytes) }?;
    }
    Ok(metadata)
}

/// A refused enqueue as a launch error: a work-group size the build took but the device refuses
/// at enqueue (`PoCL`) is too many units.
fn launch_error(err: ClError, requested: u32, max: u32) -> ServerError {
    ServerError::Launch(match err.code {
        Some(CL_INVALID_WORK_GROUP_SIZE) => {
            LaunchError::TooManyResources(ResourceLimitError::Units {
                requested,
                max,
                backtrace: BackTrace::capture(),
            })
        }
        _ => err.into(),
    })
}

/// Enqueues the copies on `stream` of `runs` (offset in the binding, bytes) between `resource` and
/// the host bytes at `host`, where the runs lie one after the other.
///
/// # Safety
///
/// The runs lie in `resource` ([`runs`]), and `host` is valid for their total and outlives the
/// copies.
unsafe fn copy_runs(
    stream: &OpenClStream,
    resource: &OpenClResource,
    runs: &[(usize, usize)],
    host: *mut u8,
    to_device: bool,
) -> Result<(), ClError> {
    let mut at = 0;
    for &(offset, len) in runs {
        // SAFETY: as the function's.
        let (device, host) = (resource.at(offset), unsafe { host.add(at) }.cast());
        let (dst, src) = if to_device {
            (device, host)
        } else {
            (host, device)
        };
        // SAFETY: as the function's.
        unsafe {
            stream
                .memory
                .copy(stream.queue, false, dst, src.cast_const(), len)
        }?;
        at += len;
    }
    Ok(())
}

/// The dynamic metadata of one launch: a read-only buffer holding a copy of it, released once the launch is
/// enqueued (the kernel keeps it alive until it has run).
struct Metadata(cl_mem);

impl Metadata {
    fn new(context: cl_context, data: &[u8]) -> Result<Self, ClError> {
        // An empty tail (a kernel whose tensors all have rank 0) still binds a buffer.
        let padded = [0u8; 4];
        let data = if data.is_empty() { &padded[..] } else { data };
        // SAFETY: `data` is valid for its length; the buffer copies it at creation.
        unsafe {
            cl3::memory::create_buffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                data.len(),
                data.as_ptr().cast_mut().cast(),
            )
        }
        .map(Self)
        .map_err(|code| ClError {
            call: "clCreateBuffer",
            code: Some(code),
        })
    }
}

impl Drop for Metadata {
    fn drop(&mut self) {
        // The reference this value owns; the runtime keeps the buffer while a kernel uses it.
        cl::api().clReleaseMemObject(self.0);
    }
}

impl ServerCommunication for OpenClServer {}

impl WriteScoped for OpenClServer {
    type Streams = MultiStream<OpenClStreamBackend>;

    fn write_streams(&mut self) -> &mut Self::Streams {
        &mut self.streams
    }

    fn on_failure(&mut self, stream: StreamId, error: &ServerError) {
        self.profile_failure(stream, error);
    }
}

impl Server for OpenClServer {
    fn logger(&self) -> Arc<ServerLogger> {
        self.utilities.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities> {
        self.utilities.clone()
    }

    fn initialize_memory(
        &mut self,
        memory: ManagedMemoryHandle,
        size: u64,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        let mut resolved = self.streams.resolve(stream_id, std::iter::empty())?;
        let cursor = resolved.cursor;
        Relocating(&mut resolved).relocate_when_wanted();
        let (stream, failures) = resolved.current_and_failures();
        let allocated = stream
            .memory_management
            .reserve(size, PageUpdate::Allow, failures)
            .and_then(|reserved| {
                stream
                    .memory_management
                    .bind(reserved, memory.clone(), cursor, failures)
            });
        if let Err(err) = allocated {
            let err = ServerError::from(err);
            failures.fail_unallocated(&memory, err.clone());
            return Err(err);
        }
        Ok(())
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        if let Err(err) = self
            .streams
            .ensure_written(descriptors.iter().map(|d| &d.handle))
        {
            return Box::pin(async move { Err(err) });
        }
        let result = (|| {
            let mut resolved = self
                .streams
                .resolve(stream_id, descriptors.iter().map(|d| &d.handle))?;
            // Every copy checked before any is enqueued.
            let mut copies = Vec::with_capacity(descriptors.len());
            for descriptor in &descriptors {
                let resource = resolve_origin_resource(&mut resolved, &descriptor.handle)?;
                let runs = runs(descriptor, resource.size)?;
                let bytes = vec![0u8; runs.iter().map(|(_, len)| len).sum()];
                copies.push((resource, runs, bytes));
            }
            let stream = resolved.current();
            let enqueued = copies.iter_mut().try_for_each(|(resource, runs, bytes)| {
                // SAFETY: `bytes` holds the runs and outlives the finish below.
                unsafe { copy_runs(stream, resource, runs, bytes.as_mut_ptr(), false) }
            });
            match (enqueued, stream.finish()) {
                (Ok(()), Ok(())) => Ok(copies
                    .into_iter()
                    .map(|(_, _, bytes)| Bytes::from_bytes_vec(bytes))
                    .collect()),
                (Err(err), Ok(())) => Err(err.into()),
                (_, Err(err)) => {
                    // The queue never drained: the copies may still land, so the buffers stay alive.
                    core::mem::forget(copies);
                    Err(err.into_sync())
                }
            }
        })();
        Box::pin(async move { result })
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        for (descriptor, data) in descriptors {
            let mut written = self.write_set();
            written.push(descriptor.handle.clone());
            ExecuteScope::over(self, stream_id, written).execute(|server| {
                let mut resolved = server
                    .streams
                    .resolve(stream_id, [&descriptor.handle].into_iter())?;
                let resource = resolve_origin_resource(&mut resolved, &descriptor.handle)?;
                let runs = runs(&descriptor, resource.size)?;
                let total: usize = runs.iter().map(|(_, len)| len).sum();
                if total != data.len() {
                    return Err(IoError::Unknown {
                        description: format!(
                            "{} bytes written to a {total}-byte tensor",
                            data.len()
                        ),
                        backtrace: BackTrace::capture(),
                    }
                    .into());
                }
                let stream = resolved.current();
                // SAFETY: `data` holds the runs, and the stream holds `data` until the copies have run.
                let enqueued =
                    unsafe { copy_runs(stream, &resource, &runs, data.as_ptr().cast_mut(), true) }
                        .and_then(|()| stream.enqueued());
                Ok(stream.hold(data, enqueued)?)
            });
        }
    }

    unsafe fn launch(
        &mut self,
        kernel: Box<dyn CubeKernel>,
        count: CubeCount,
        bindings: KernelArguments,
        stream_id: StreamId,
        launch_mode: LaunchMode,
    ) {
        if launch_mode == LaunchMode::Queue {
            self.kernels.enqueue(kernel, ());
            return;
        }
        let id = ArtifactId {
            kernel: kernel.id(),
            variant: (),
        };
        let compiled = self.kernels.load(&*kernel, &id, &self.utilities.logger);
        let compiled = match compiled {
            Ok(compiled) => compiled,
            Err(err) => {
                if !launch_mode.discards_launch() {
                    self.fail_outputs(stream_id, &bindings, None, ServerError::Launch(err));
                } else {
                    self.profile_failure(stream_id, &ServerError::Launch(err));
                }
                return;
            }
        };
        if launch_mode.discards_launch() {
            return;
        }
        let kernel_id = id.kernel;
        let io = compiled.io.clone();

        // The grid first: an empty one runs nothing and writes nothing, so its outputs keep what they held,
        // a failure included.
        let groups = match &count {
            CubeCount::Static(x, y, z) => [*x as usize, *y as usize, *z as usize],
            CubeCount::Dynamic(binding) => match self.read_count(binding, stream_id) {
                Ok(groups) => groups,
                Err(err) => return self.fail_outputs(stream_id, &bindings, io.as_deref(), err),
            },
        };
        if groups.contains(&0) {
            return;
        }

        let mut written = self.write_set();
        written.extend(bindings.buffers_written(io.as_deref()).cloned());
        ExecuteScope::launching(
            self,
            kernel_id,
            stream_id,
            bindings.buffers_read(io.as_deref()),
            written,
        )
        .execute(|server| {
            let buffers = bindings
                .resources
                .iter()
                .map(|resource| match resource {
                    KernelResource::Buffer(binding) => Ok(binding),
                    KernelResource::TensorMap(_) => Err(IoError::UnsupportedIoOperation {
                        backtrace: BackTrace::capture(),
                    }),
                    KernelResource::Peer(_) => {
                        unreachable!(
                            "OpenCL shares no memory between devices, so no binding is a peer"
                        )
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let timed = server
                .windows
                .values()
                .any(|window| window.stream == stream_id);
            let mut resolved = server.streams.resolve(stream_id, buffers.iter().copied())?;
            let metadata = set_arguments(
                &compiled,
                &buffers,
                &bindings,
                &mut resolved,
                server.cl_context,
            )?;
            let stream = resolved.current();
            let dim = compiled.cube_dim;
            let local = [dim.x as usize, dim.y as usize, dim.z as usize];
            let max_units = server.utilities.properties.hardware.max_units_per_cube;
            // SAFETY: every argument is set, to memory that outlives the run.
            let event =
                unsafe { cl::launch(stream.queue, compiled.kernel.raw(), groups, local, timed) }
                    .map_err(|err| launch_error(err, dim.num_elems(), max_units))?;
            drop(metadata);
            stream.enqueued()?;
            drop(resolved);
            if let Some(event) = event {
                server.timed_launch(stream_id, event);
            }
            Ok(())
        });
    }

    fn sync(
        &mut self,
        handles: Vec<BufferBinding>,
        stream_id: StreamId,
    ) -> DynFut<Result<(), ServerError>> {
        if let Err(err) = self.streams.ensure_written(handles.iter()) {
            return Box::pin(async move { Err(err) });
        }
        let result = (|| {
            let mut resolved = self.streams.resolve(stream_id, std::iter::empty())?;
            let (stream, failures) = resolved.current_and_failures();
            stream.synced();
            let event = OpenClStreamBackend::flush(stream, failures);
            OpenClStreamBackend::wait_event_sync(event)
        })();
        Box::pin(async move { result })
    }

    fn check(
        &mut self,
        handles: Vec<BufferBinding>,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        self.streams.ensure_written(handles.iter())
    }

    fn compile_queued(&mut self) {
        self.kernels.compile_queued(&self.utilities.logger);
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut resolved = self.streams.resolve(stream_id, std::iter::empty())?;
        let stream = resolved.current();
        // SAFETY: the stream's own queue.
        Ok(unsafe { cl::flush(stream.queue) }?)
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        let token = self.timestamps.start();
        self.windows.insert(
            token,
            Window {
                stream: stream_id,
                first: None,
                last: None,
                failure: None,
            },
        );
        Ok(token)
    }

    fn end_profile(
        &mut self,
        _stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        let Some(Window {
            stream: origin,
            first,
            last,
            failure,
        }) = self.windows.remove(&token)
        else {
            return self.timestamps.stop(token);
        };
        // The token is the profiler's to clear either way; its own failures are never recorded (they are the
        // window's, above).
        let cleared = self.timestamps.stop(token);
        if let Some(failure) = failure {
            return Err(ProfileError::from(&failure));
        }
        cleared?;
        // From the window's first launch's start to its last's end, on the stream it opened on whichever stream
        // closes it; a window that launched nothing measured nothing.
        let (Some(first), Some(last)) = (first, last) else {
            return Err(ProfileError::NotMeasured {
                backtrace: BackTrace::capture(),
            });
        };
        let span = (|| {
            let mut resolved = self.streams.resolve(origin, std::iter::empty())?;
            // SAFETY: the stream's own queue.
            unsafe { cl::flush(resolved.current().queue) }?;
            last.wait().map_err(ClError::into_sync)?;
            let start = first.time_ns(CL_PROFILING_COMMAND_START)?;
            let end = last.time_ns(CL_PROFILING_COMMAND_END)?;
            Ok::<_, ServerError>(Duration::from_nanos(end.saturating_sub(start)))
        })();
        let span = span.map_err(|err| ProfileError::from(&err))?;
        let base = Instant::now();
        let ticks = ProfileTicks::from_start_end(base, base + span);
        Ok(ProfileDuration::new_device_time(async move { ticks }))
    }

    fn abandon_profile(&mut self, _stream_id: StreamId, token: ProfilingToken) {
        self.windows.remove(&token);
        self.timestamps.abandon(token);
    }

    fn memory_report(
        &mut self,
        stream_id: StreamId,
    ) -> Option<cubecl_server::memory_management::StreamMemoryReport> {
        let stream = self.streams.try_stream_mut(&stream_id)?;
        Some(cubecl_server::memory_management::StreamMemoryReport {
            stream: stream_id,
            pools: stream.memory_management.memory_report(),
            auxiliary: Vec::new(),
        })
    }

    fn stream_ids(&self) -> Vec<StreamId> {
        self.streams.stream_ids().collect()
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut resolved = self.streams.resolve(stream_id, std::iter::empty())?;
        Relocating(&mut resolved).reclaim()
    }

    fn allocation_mode(
        &mut self,
        mode: cubecl_server::memory_management::MemoryAllocationMode,
        stream_id: StreamId,
    ) {
        if let Ok(mut resolved) = self.streams.resolve(stream_id, std::iter::empty()) {
            resolved.current().memory_management.mode(mode);
        }
    }
}

/// A command's streams, as relocating the current stream's memory needs them.
struct Relocating<'r, 'a>(&'r mut ResolvedStreams<'a, OpenClStreamBackend>);

impl RelocatingStreams for Relocating<'_, '_> {
    fn recording(&mut self) -> bool {
        false
    }

    fn has_outdated(&mut self) -> bool {
        self.0.current().memory_management.has_outdated()
    }

    fn relocation_need(&mut self) -> RelocationNeed {
        self.0.current().memory_management.relocation_need()
    }

    fn bytes_allocated(&mut self) -> u64 {
        self.0
            .all()
            .map(|stream| stream.memory_management.bytes_allocated())
            .sum()
    }

    fn finish(&mut self) {
        for stream in self.0.all() {
            if let Err(err) = stream.finish() {
                log::warn!("{err}");
            }
        }
    }

    fn cleanup_memory(&mut self) {
        let (stream, failures) = self.0.current_and_failures();
        stream
            .memory_management
            .cleanup(Cleanup::Explicit, failures);
    }

    fn relocate_memory(&mut self, reason: RelocationReason) {
        let (stream, failures) = self.0.current_and_failures();
        stream.relocate(reason, failures);
    }

    fn device_has_outdated(&mut self) -> bool {
        self.0
            .all()
            .any(|stream| stream.memory_management.has_outdated())
    }

    fn relocate_device_memory(&mut self, reason: RelocationReason) {
        for stream_id in self.0.stream_ids() {
            let (stream, failures) = self.0.get_and_failures(&stream_id);
            if stream.memory_management.has_outdated() {
                stream.relocate(reason, failures);
            }
        }
    }
}

impl ServerStorage for OpenClServer {
    type Storage = OpenClStorage;

    fn get_resource(
        &mut self,
        binding: BufferBinding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<<OpenClStorage as ComputeStorage>::Resource>, ServerError> {
        self.streams.ensure_written([&binding].into_iter())?;
        let mut resolved = self.streams.resolve(stream_id, std::iter::once(&binding))?;
        let stream = resolved.get(&binding.stream);
        Ok(stream.memory_management.managed_resource(
            binding.memory,
            binding.offset_start,
            binding.offset_end,
        )?)
    }
}
