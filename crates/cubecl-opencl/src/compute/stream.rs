use crate::compute::cl::{self, ClError, Event};
use crate::memory::{Memory, OpenClStorage};
use cubecl_common::bytes::Bytes;
use cubecl_core::{MemoryConfiguration, server::ServerError};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_server::memory_management::relocation::{CopyQueue, RelocationReason, StorageCopy};
use cubecl_server::server::IoError;
use cubecl_server::storage::ComputeStorage;
use cubecl_server::{
    logging::ServerLogger,
    memory_management::{ErrorGraph, FailureId, MemoryManagement, MemoryManagementOptions},
    server::BufferBinding,
    stream::{EventStreamBackend, StreamMemory},
};
use opencl_sys::{cl_command_queue, cl_context, cl_device_id};
use std::sync::Arc;

/// Launches before a stream's first flush after it synced (a read or a sync, not the flushes other
/// streams and released memory ask for); each flush doubles the batch, up to [`LAST_BATCH`]: work
/// reaches the device soon after a sync, and a host running ahead submits few, large batches (a
/// submission can leave the device idle a while).
const FIRST_BATCH: usize = 32;
const LAST_BATCH: usize = 1024;

/// A stream: a command queue and the memory it allocates.
#[derive(Debug)]
pub struct OpenClStream {
    pub(crate) queue: cl_command_queue,
    /// How the device's memory is allocated, handed to kernels and copied.
    pub(crate) memory: Memory,
    pub(crate) memory_management: MemoryManagement<OpenClStorage>,
    batch_ops: usize,
    /// Commands the current batch takes before it is flushed.
    batch: usize,
    /// The queue is out of order, kept in order by a barrier after every command (see
    /// [`OpenClStreamBackend::new`]).
    pub(crate) barriers: bool,
    /// Host data of copies still in flight, until their event completes.
    in_flight: Vec<(Event, Bytes)>,
}

// SAFETY: the queue is a thread-safe OpenCL handle the stream owns a reference to, and the server owns the stream.
unsafe impl Send for OpenClStream {}

impl Drop for OpenClStream {
    fn drop(&mut self) {
        // SAFETY: the stream's own queue, alive until released below.
        if unsafe { cl::finish(self.queue) }.is_err() {
            // Copies from the host data may still run: leaked rather than freed under them.
            self.in_flight
                .drain(..)
                .for_each(|(_, data)| core::mem::forget(data));
        }
        self.in_flight.clear();
        // The storage holds its own reference to the queue.
        cl::api().clReleaseCommandQueue(self.queue);
    }
}

impl StreamMemory for OpenClStream {
    fn failure(&self, binding: &BufferBinding) -> Option<FailureId> {
        self.memory_management
            .failure(&binding.memory, binding.range())
    }

    fn taint(&mut self, binding: &BufferBinding, failure: FailureId, failures: &mut ErrorGraph) {
        self.memory_management
            .taint(&binding.memory, binding.range(), failure, failures)
    }

    fn written(&mut self, binding: &BufferBinding, failures: &mut ErrorGraph) {
        self.memory_management
            .written(&binding.memory, binding.range(), failures)
    }
}

impl OpenClStream {
    /// Orders what follows after the command just enqueued, and submits the batch when it is full.
    pub(crate) fn enqueued(&mut self) -> Result<(), ClError> {
        // SAFETY: the stream's own queue.
        unsafe {
            if self.barriers
                && let Err(err) = cl::barrier(self.queue)
            {
                // Nothing orders the command before the next one: wait for it instead.
                let _ = cl::finish(self.queue);
                return Err(err);
            }
            self.batch_ops += 1;
            if self.batch_ops >= self.batch {
                self.batch_ops = 0;
                self.batch = (2 * self.batch).min(LAST_BATCH);
                cl::flush(self.queue)?;
            }
        }
        Ok(())
    }

    /// Keeps `data` alive until the copies just enqueued from it have run. When they could not all
    /// be `enqueued`, waits for the ones that were instead, and leaks `data` if even that fails.
    pub(crate) fn hold(
        &mut self,
        data: Bytes,
        enqueued: Result<(), ClError>,
    ) -> Result<(), ClError> {
        self.release_copied();
        // SAFETY: the stream's own queue.
        match enqueued.and_then(|()| unsafe { cl::marker(self.queue) }) {
            Ok(copied) => {
                self.in_flight.push((copied, data));
                Ok(())
            }
            Err(err) => {
                match self.finish() {
                    Ok(()) => drop(data),
                    Err(_) => core::mem::forget(data),
                }
                Err(err)
            }
        }
    }

    /// Drops the host data of the copies that have run.
    fn release_copied(&mut self) {
        self.in_flight.retain(|(event, _)| !event.done());
    }

    /// Waits until everything the stream enqueued has run.
    pub(crate) fn finish(&mut self) -> Result<(), ClError> {
        (self.batch_ops, self.batch) = (0, FIRST_BATCH);
        // SAFETY: the stream's own queue.
        unsafe { cl::finish(self.queue) }?;
        self.in_flight.clear();
        Ok(())
    }

    /// The host waits for the stream: its next commands start the device early again.
    pub(crate) fn synced(&mut self) {
        self.batch = FIRST_BATCH;
    }

    /// Empties the outdated pools into the current pages, behind the stream's work.
    pub(crate) fn relocate(&mut self, reason: RelocationReason, failures: &mut ErrorGraph) {
        let mut copier = OpenClCopies {
            queue: self.queue,
            memory: self.memory,
        };
        self.memory_management
            .relocate(&mut copier, reason, failures);
    }
}

/// A relocation's copies, on the stream's own queue.
struct OpenClCopies {
    queue: cl_command_queue,
    memory: Memory,
}

impl CopyQueue<OpenClStorage> for OpenClCopies {
    fn wait_device(&mut self) -> Result<(), ServerError> {
        // SAFETY: the stream's queue, alive while it relocates.
        unsafe { cl::finish(self.queue) }.map_err(ClError::into_sync)
    }

    fn copy(&mut self, storage: &mut OpenClStorage, copy: &StorageCopy) -> Result<(), IoError> {
        let source = storage.get(&copy.source)?;
        let target = storage.get(&copy.target)?;
        // SAFETY: two live slices of this device the same size, untouched until the copy lands (`wait_copies`).
        unsafe {
            self.memory.copy(
                self.queue,
                false,
                target.at(0),
                source.at(0).cast_const(),
                copy.source.size() as usize,
            )
        }
        .map_err(IoError::from)
    }

    fn wait_copies(&mut self) -> Result<(), ServerError> {
        // SAFETY: as in `wait_device`.
        unsafe { cl::finish(self.queue) }.map_err(ClError::into_sync)
    }
}

/// Creates the streams of one device.
#[derive(Debug)]
pub struct OpenClStreamBackend {
    memory: Memory,
    context: cl_context,
    device: cl_device_id,
    out_of_order: bool,
    mem_props: MemoryDeviceProperties,
    mem_config: MemoryConfiguration,
    logger: Arc<ServerLogger>,
}

// SAFETY: OpenCL contexts and devices are thread-safe handles; the server keeps the context alive.
unsafe impl Send for OpenClStreamBackend {}

impl OpenClStreamBackend {
    /// `out_of_order`: streams on out-of-order queues kept in order by barriers (Intel's driver submits an in-order
    /// queue's every command on its own; out of order it batches them).
    pub(crate) fn new(
        memory: Memory,
        context: cl_context,
        device: cl_device_id,
        out_of_order: bool,
        mem_props: MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
    ) -> Self {
        Self {
            memory,
            context,
            device,
            out_of_order,
            mem_props,
            mem_config,
            logger,
        }
    }
}

impl EventStreamBackend for OpenClStreamBackend {
    type Stream = OpenClStream;
    /// A flush's marker, or why it couldn't be made.
    type Event = Result<Event, ClError>;

    fn create_stream(&self) -> Result<Self::Stream, ServerError> {
        // Profiled, for the device-timed profiling windows.
        let mut flags = opencl_sys::CL_QUEUE_PROFILING_ENABLE;
        if self.out_of_order {
            flags |= opencl_sys::CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        }
        let properties = [
            opencl_sys::CL_QUEUE_PROPERTIES as opencl_sys::cl_queue_properties,
            flags,
            0,
        ];
        // SAFETY: a queue on this backend's context and device, both alive while the server is.
        let queue = unsafe {
            cl3::command_queue::create_command_queue_with_properties(
                self.context,
                self.device,
                properties.as_ptr(),
            )
        }
        .map_err(|code| ClError {
            call: "clCreateCommandQueueWithProperties",
            code: Some(code),
        })?;
        let storage = OpenClStorage::new(
            self.memory,
            self.context,
            queue,
            self.mem_props.alignment as usize,
            self.mem_props.max_page_size,
        );
        let memory_management = MemoryManagement::from_configuration(
            storage,
            &self.mem_props,
            self.mem_config.clone(),
            self.logger.clone(),
            MemoryManagementOptions::new("OpenCL device memory"),
        );
        Ok(OpenClStream {
            queue,
            memory: self.memory,
            memory_management,
            batch_ops: 0,
            batch: FIRST_BATCH,
            barriers: self.out_of_order,
            in_flight: Vec::new(),
        })
    }

    fn flush(stream: &mut Self::Stream, _failures: &mut ErrorGraph) -> Self::Event {
        stream.batch_ops = 0;
        stream.release_copied();
        // SAFETY: the stream's own queue.
        unsafe {
            cl::marker(stream.queue).and_then(|event| cl::flush(stream.queue).map(|()| event))
        }
    }

    fn handle_cursor(stream: &Self::Stream, handle: &BufferBinding) -> u64 {
        stream
            .memory_management
            .get_cursor(handle.memory.clone())
            .unwrap_or(u64::MAX)
    }

    fn wait_event(stream: &mut Self::Stream, event: Self::Event) -> Result<(), ServerError> {
        // SAFETY: the stream's own queue.
        Ok(unsafe { cl::wait_on(stream.queue, &event?) }?)
    }

    fn wait_event_sync(event: Self::Event) -> Result<(), ServerError> {
        event?.wait().map_err(ClError::into_sync)
    }
}
