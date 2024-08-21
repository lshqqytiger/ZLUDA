#![allow(warnings)]
mod cudart;
pub use cudart::*;

use hip_runtime_sys::*;
use std::mem;

#[cfg(debug_assertions)]
fn unsupported() -> cudaError_t {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
fn unsupported() -> cudaError_t {
    cudaError_t::cudaErrorNotSupported
}

fn to_cuda(status: hipError_t) -> cudaError_t {
    match status {
        hipError_t::hipSuccess => cudaError_t::cudaSuccess,
        hipError_t::hipErrorInvalidValue => cudaError_t::cudaErrorInvalidValue,
        hipError_t::hipErrorOutOfMemory => cudaError_t::cudaErrorMemoryAllocation,
        hipError_t::hipErrorInvalidContext => cudaError_t::cudaErrorDeviceUninitialized,
        hipError_t::hipErrorInvalidResourceHandle => cudaError_t::cudaErrorInvalidResourceHandle,
        hipError_t::hipErrorNotSupported => cudaError_t::cudaErrorNotSupported,
        err => panic!("[ZLUDA] HIP Runtime failed: {}", err.0),
    }
}

fn to_hip(status: cudaError_t) -> hipError_t {
    match status {
        cudaError_t::cudaSuccess => hipError_t::hipSuccess,
        cudaError_t::cudaErrorInvalidValue => hipError_t::hipErrorInvalidValue,
        cudaError_t::cudaErrorMemoryAllocation => hipError_t::hipErrorOutOfMemory,
        cudaError_t::cudaErrorDeviceUninitialized => hipError_t::hipErrorInvalidContext,
        cudaError_t::cudaErrorInvalidResourceHandle => hipError_t::hipErrorInvalidResourceHandle,
        cudaError_t::cudaErrorNotSupported => hipError_t::hipErrorNotSupported,
        err => panic!("[ZLUDA] HIP Runtime failed: {}", err.0),
    }
}

unsafe fn to_stream(stream: cudaStream_t) -> hipStream_t {
    let lib = hip_common::zluda_ext::get_cuda_library().unwrap();
    let cu_get_export_table = lib
        .get::<unsafe extern "C" fn(
            ppExportTable: *mut *const ::std::os::raw::c_void,
            pExportTableId: *const cuda_types::CUuuid,
        ) -> cuda_types::CUresult>(b"cuGetExportTable\0")
        .unwrap();
    let mut export_table = std::ptr::null();
    let error = (cu_get_export_table)(&mut export_table, &zluda_dark_api::ZludaExt::GUID);
    assert_eq!(error, cuda_types::CUresult::CUDA_SUCCESS);
    let zluda_ext = zluda_dark_api::ZludaExt::new(export_table);
    let maybe_hip_stream: Result<_, _> = zluda_ext.get_hip_stream(stream as _).into();
    maybe_hip_stream.unwrap() as _
}

fn to_hip_memcpy_kind(memcpy_kind: cudaMemcpyKind) -> hipMemcpyKind {
    match memcpy_kind {
        cudaMemcpyKind::cudaMemcpyHostToHost => hipMemcpyKind::hipMemcpyHostToHost,
        cudaMemcpyKind::cudaMemcpyHostToDevice => hipMemcpyKind::hipMemcpyHostToDevice,
        cudaMemcpyKind::cudaMemcpyDeviceToHost => hipMemcpyKind::hipMemcpyDeviceToHost,
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => hipMemcpyKind::hipMemcpyDeviceToDevice,
        cudaMemcpyKind::cudaMemcpyDefault => hipMemcpyKind::hipMemcpyDefault,
        _ => panic!(),
    }
}

fn to_hip_mem_pool_attr(mem_pool_attr: cudaMemPoolAttr) -> hipMemPoolAttr {
    match mem_pool_attr {
        cudaMemPoolAttr::cudaMemPoolReuseFollowEventDependencies => hipMemPoolAttr::hipMemPoolReuseFollowEventDependencies,
        cudaMemPoolAttr::cudaMemPoolReuseAllowOpportunistic => hipMemPoolAttr::hipMemPoolReuseAllowOpportunistic,
        cudaMemPoolAttr::cudaMemPoolReuseAllowInternalDependencies => hipMemPoolAttr::hipMemPoolReuseAllowInternalDependencies,
        cudaMemPoolAttr::cudaMemPoolAttrReleaseThreshold => hipMemPoolAttr::hipMemPoolAttrReleaseThreshold,
        cudaMemPoolAttr::cudaMemPoolAttrReservedMemCurrent => hipMemPoolAttr::hipMemPoolAttrReservedMemCurrent,
        cudaMemPoolAttr::cudaMemPoolAttrReservedMemHigh => hipMemPoolAttr::hipMemPoolAttrReservedMemHigh,
        cudaMemPoolAttr::cudaMemPoolAttrUsedMemCurrent => hipMemPoolAttr::hipMemPoolAttrUsedMemCurrent,
        cudaMemPoolAttr::cudaMemPoolAttrUsedMemHigh => hipMemPoolAttr::hipMemPoolAttrUsedMemHigh,
        _ => panic!(),
    }
}

fn to_cuda_stream_capture_status(stream_capture_status: hipStreamCaptureStatus) -> cudaStreamCaptureStatus {
    match stream_capture_status {
        hipStreamCaptureStatus::hipStreamCaptureStatusNone => cudaStreamCaptureStatus::cudaStreamCaptureStatusNone,
        hipStreamCaptureStatus::hipStreamCaptureStatusActive => cudaStreamCaptureStatus::cudaStreamCaptureStatusActive,
        hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated => cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated,
        _ => panic!(),
    }
}

fn to_hip_dim3(dim: cudart::dim3) -> hip_runtime_api::dim3 {
    hip_runtime_api::dim3 {
        x: dim.x,
        y: dim.y,
        z: dim.z,
    }
}

unsafe fn pop_call_configuration(
    grid_dim: *mut cudart::dim3,
    block_dim: *mut cudart::dim3,
    shared_mem: *mut usize,
    stream: *mut cudaStream_t,
) -> cudaError_t {
    to_cuda(__hipPopCallConfiguration(
        grid_dim.cast(),
        block_dim.cast(),
        shared_mem,
        stream.cast(),
    ))
}

unsafe fn push_call_configuration(
    grid_dim: cudart::dim3,
    block_dim: cudart::dim3,
    shared_mem: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    let grid_dim = to_hip_dim3(grid_dim);
    let block_dim = to_hip_dim3(block_dim);
    let stream = to_stream(stream);
    to_cuda(__hipPushCallConfiguration(
        grid_dim,
        block_dim,
        shared_mem,
        stream,
    ))
}

unsafe fn register_fat_binary(
    fat_cubin: *mut ::std::os::raw::c_void,
) -> *mut *mut ::std::os::raw::c_void {
    __hipRegisterFatBinary(fat_cubin)
}

unsafe fn register_fat_binary_end(
    _fat_cubin_handle: *mut *mut ::std::os::raw::c_void,
) -> () {
    //__hipRegisterFatBinaryEnd(fat_cubin_handle)
}

unsafe fn register_function(
    fat_cubin_handle: *mut *mut ::std::os::raw::c_void,
    host_fun: *const ::std::os::raw::c_char,
    device_fun: *mut ::std::os::raw::c_char,
    device_name: *const ::std::os::raw::c_char,
    thread_limit: i32,
    tid: *mut ::std::os::raw::c_void,
    bid: *mut ::std::os::raw::c_void,
    b_dim: *mut cudart::dim3,
    g_dim: *mut cudart::dim3,
    w_size: *mut i32,
) -> ::std::os::raw::c_void {
    __hipRegisterFunction(
        fat_cubin_handle,
        host_fun.cast(),
        device_fun,
        device_name,
        thread_limit as _,
        tid,
        bid,
        b_dim.cast(),
        g_dim.cast(),
        w_size,
    )
}

unsafe fn register_host_var(
    fat_cubin_handle: *mut *mut ::std::os::raw::c_void,
    device_name: *const ::std::os::raw::c_char,
    host_var: *mut ::std::os::raw::c_char,
    size: usize,
) -> ::std::os::raw::c_void {
    __hipRegisterVar(
        fat_cubin_handle,
        host_var.cast(),
        host_var,
        device_name.cast_mut(),
        0,
        size,
        0,
        0,
    )
}

unsafe fn register_managed_var(
    fat_cubin_handle: *mut *mut ::std::os::raw::c_void,
    host_var_ptr_address: *mut *mut ::std::os::raw::c_void,
    device_address: *mut ::std::os::raw::c_char,
    device_name: *const ::std::os::raw::c_char,
    ext: i32,
    size: usize,
    constant: i32,
    global: i32,
) -> ::std::os::raw::c_void {
    __hipRegisterVar(
        fat_cubin_handle,
        *host_var_ptr_address,
        device_address,
        device_name.cast_mut(),
        ext,
        size,
        constant,
        global,
    )
}

unsafe fn register_surface(
    fat_cubin_handle: *mut *mut ::std::os::raw::c_void,
    host_var: *const ::std::os::raw::c_void,
    device_address: *const *mut ::std::os::raw::c_void,
    device_name: *const ::std::os::raw::c_char,
    dim: i32,
    ext: i32,
) -> ::std::os::raw::c_void {
    __hipRegisterSurface(
        fat_cubin_handle,
        host_var.cast_mut(),
        (*device_address).cast(),
        device_name.cast_mut(),
        dim,
        ext,
    )
}

unsafe fn register_texture(
    fat_cubin_handle: *mut *mut ::std::os::raw::c_void,
    host_var: *const ::std::os::raw::c_void,
    device_address: *const *mut ::std::os::raw::c_void,
    device_name: *const ::std::os::raw::c_char,
    dim: i32,
    norm: i32,
    ext: i32,
) -> ::std::os::raw::c_void {
    __hipRegisterTexture(
        fat_cubin_handle,
        host_var.cast_mut(),
        (*device_address).cast(),
        device_name.cast_mut(),
        dim,
        norm,
        ext,
    )
}

unsafe fn register_var(
    fat_cubin_handle: *mut *mut ::std::os::raw::c_void,
    host_var: *mut ::std::os::raw::c_char,
    device_address: *mut ::std::os::raw::c_char,
    device_name: *const ::std::os::raw::c_char,
    ext: i32,
    size: usize,
    constant: i32,
    global: i32,
) -> ::std::os::raw::c_void {
    __hipRegisterVar(
        fat_cubin_handle,
        host_var.cast(),
        device_address,
        device_name.cast_mut(),
        ext,
        size,
        constant,
        global,
    )
}

unsafe fn unregister_fat_binary(
    fat_cubin_handle: *mut *mut ::std::os::raw::c_void,
) -> ::std::os::raw::c_void {
    __hipUnregisterFatBinary(fat_cubin_handle)
}

unsafe fn device_reset() -> cudaError_t {
    to_cuda(hipDeviceReset())
}

unsafe fn device_synchronize() -> cudaError_t {
    to_cuda(hipDeviceSynchronize())
}

unsafe fn device_get_stream_priority_range(
    least_priority: *mut i32,
    greatest_priority: *mut i32,
) -> cudaError_t {
    to_cuda(hipDeviceGetStreamPriorityRange(
        least_priority,
        greatest_priority,
    ))
}

unsafe fn device_get_by_pci_bus_id(
    device: *mut i32,
    pci_bus_id: *const ::std::os::raw::c_char,
) -> cudaError_t {
    to_cuda(hipDeviceGetByPCIBusId(
        device,
        pci_bus_id,
    ))
}

unsafe fn device_get_pci_bus_id(
    pci_bus_id: *mut ::std::os::raw::c_char,
    len: i32,
    device: i32,
) -> cudaError_t {
    to_cuda(hipDeviceGetPCIBusId(
        pci_bus_id,
        len,
        device,
    ))
}

unsafe fn ipc_get_event_handle(
    handle: *mut cudaIpcEventHandle_t,
    event: cudaEvent_t,
) -> cudaError_t {
    to_cuda(hipIpcGetEventHandle(
        handle.cast(),
        event.cast(),
    ))
}

unsafe fn ipc_get_mem_handle(
    handle: *mut cudaIpcMemHandle_t,
    dev_ptr: *mut ::std::os::raw::c_void,
) -> cudaError_t {
    to_cuda(hipIpcGetMemHandle(
        handle.cast(),
        dev_ptr,
    ))
}

unsafe fn ipc_close_mem_handle(
    dev_ptr: *mut ::std::os::raw::c_void,
) -> cudaError_t {
    to_cuda(hipIpcCloseMemHandle(dev_ptr))
}

unsafe fn get_last_error() -> cudaError_t {
    to_cuda(hipGetLastError())
}

unsafe fn peek_at_last_error() -> cudaError_t {
    to_cuda(hipPeekAtLastError())
}

unsafe fn get_error_name(
    error: cudaError_t,
) -> *const ::std::os::raw::c_char {
    let error = to_hip(error);
    hipGetErrorName(error)
}

unsafe fn get_error_string(
    error: cudaError_t,
) -> *const ::std::os::raw::c_char {
    let error = to_hip(error);
    hipGetErrorString(error)
}

unsafe fn get_device_count(
    count: *mut i32,
) -> cudaError_t {
    to_cuda(hipGetDeviceCount(count))
}

unsafe fn get_device_properties(
    prop: *mut cudaDeviceProp,
    device: i32,
) -> cudaError_t {
    to_cuda(hipGetDevicePropertiesR0600(
        prop.cast(),
        device,
    ))
}

unsafe fn device_get_default_mem_pool(
    mem_pool: *mut cudaMemPool_t,
    device: i32,
) -> cudaError_t {
    to_cuda(hipDeviceGetDefaultMemPool(
        mem_pool.cast(),
        device,
    ))
}

unsafe fn device_set_mem_pool(
    device: i32,
    mem_pool: cudaMemPool_t,
) -> cudaError_t {
    to_cuda(hipDeviceSetMemPool(
        device,
        mem_pool.cast(),
    ))
}

unsafe fn device_get_mem_pool(
    mem_pool: *mut cudaMemPool_t,
    device: i32,
) -> cudaError_t {
    to_cuda(hipDeviceGetMemPool(
        mem_pool.cast(),
        device,
    ))
}

unsafe fn set_device(
    device: i32,
) -> cudaError_t {
    to_cuda(hipSetDevice(device))
}

unsafe fn get_device(
    device: *mut i32,
) -> cudaError_t {
    to_cuda(hipGetDevice(device))
}

unsafe fn set_device_flags(
    flags: u32,
) -> cudaError_t {
    to_cuda(hipSetDeviceFlags(flags))
}

unsafe fn get_device_flags(
    flags: *mut u32,
) -> cudaError_t {
    to_cuda(hipGetDeviceFlags(flags))
}

unsafe fn stream_create(
    p_stream: *mut cudaStream_t,
) -> cudaError_t {
    stream_create_with_flags(
        p_stream,
        0,
    )
}

unsafe fn stream_create_with_flags(
    p_stream: *mut cudaStream_t,
    flags: u32,
) -> cudaError_t {
    let lib = hip_common::zluda_ext::get_cuda_library().unwrap();
    let cu_stream_create = lib
        .get::<unsafe extern "C" fn(
            phStream: *mut cuda_types::CUstream,
            Flags: ::std::os::raw::c_uint,
        ) -> cuda_types::CUresult>(b"cuStreamCreate\0")
        .unwrap();
    cudaError_t((cu_stream_create)(
        p_stream.cast(),
        flags,
    ).0)
}

unsafe fn stream_create_with_priority(
    p_stream: *mut cudaStream_t,
    flags: u32,
    priority: i32,
) -> cudaError_t {
    let lib = hip_common::zluda_ext::get_cuda_library().unwrap();
    let cu_stream_create_with_priority = lib
        .get::<unsafe extern "C" fn(
            phStream: *mut cuda_types::CUstream,
            flags: ::std::os::raw::c_uint,
            priority: ::std::os::raw::c_int,
        ) -> cuda_types::CUresult>(b"cuStreamCreateWithPriority\0")
        .unwrap();
    cudaError_t((cu_stream_create_with_priority)(
        p_stream.cast(),
        flags,
        priority,
    ).0)
}

unsafe fn stream_get_priority(
    h_stream: cudaStream_t,
    priority: *mut i32,
) -> cudaError_t {
    let h_stream = to_stream(h_stream);
    to_cuda(hipStreamGetPriority(
        h_stream,
        priority,
    ))
}

unsafe fn stream_get_priority_ptsz(
    h_stream: cudaStream_t,
    priority: *mut i32,
) -> cudaError_t {
    let h_stream = to_stream(h_stream);
    to_cuda(hipStreamGetPriority_spt(
        h_stream,
        priority,
    ))
}

unsafe fn stream_get_flags(
    h_stream: cudaStream_t,
    flags: *mut u32,
) -> cudaError_t {
    let h_stream = to_stream(h_stream);
    to_cuda(hipStreamGetFlags(
        h_stream,
        flags,
    ))
}

unsafe fn stream_get_flags_ptsz(
    h_stream: cudaStream_t,
    flags: *mut u32,
) -> cudaError_t {
    let h_stream = to_stream(h_stream);
    to_cuda(hipStreamGetFlags_spt(
        h_stream,
        flags,
    ))
}

unsafe fn stream_destroy(
    stream: cudaStream_t,
) -> cudaError_t {
    let lib = hip_common::zluda_ext::get_cuda_library().unwrap();
    let cu_stream_destroy = lib
        .get::<unsafe extern "C" fn(hStream: cuda_types::CUstream) -> cuda_types::CUresult>(b"cuStreamDestroy\0")
        .unwrap();
    cudaError_t((cu_stream_destroy)(
        stream.cast(),
    ).0)
}

unsafe fn stream_wait_event(
    stream: cudaStream_t,
    event: cudaEvent_t,
    flags: u32,
) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipStreamWaitEvent(
        stream,
        event.cast(),
        flags,
    ))
}

unsafe fn stream_wait_event_ptsz(
    stream: cudaStream_t,
    event: cudaEvent_t,
    flags: u32,
) -> cudaError_t {
    let lib = hip_common::zluda_ext::get_cuda_library().unwrap();
    let cu_stream_wait_event = lib
        .get::<unsafe extern "C" fn(
            hStream: cuda_types::CUstream,
            hEvent: cuda_types::CUevent,
            Flags: ::std::os::raw::c_uint,
        ) -> cuda_types::CUresult>(b"cuStreamWaitEvent_ptsz\0")
        .unwrap();
    cudaError_t((cu_stream_wait_event)(
        stream.cast(),
        event.cast(),
        flags,
    ).0)
}

unsafe fn stream_synchronize(
    stream: cudaStream_t,
) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipStreamSynchronize(stream))
}

unsafe fn stream_synchronize_ptsz(
    stream: cudaStream_t,
) -> cudaError_t {
    let lib = hip_common::zluda_ext::get_cuda_library().unwrap();
    let cu_stream_synchronize = lib
        .get::<unsafe extern "C" fn(
            hStream: cuda_types::CUstream,
        ) -> cuda_types::CUresult>(b"cuStreamSynchronize_ptsz\0")
        .unwrap();
    cudaError_t((cu_stream_synchronize)(
        stream.cast(),
    ).0)
}

unsafe fn stream_query(
    stream: cudaStream_t,
) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipStreamQuery(stream))
}

unsafe fn stream_attach_mem_async(
    stream: cudaStream_t,
    dev_ptr: *mut ::std::os::raw::c_void,
    length: usize,
    flags: u32,
) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipStreamAttachMemAsync(
        stream,
        dev_ptr,
        length,
        flags,
    ))
}

unsafe fn stream_end_capture(
    stream: cudaStream_t,
    p_graph: *mut cudaGraph_t,
) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipStreamEndCapture(
        stream,
        p_graph.cast(),
    ))
}

unsafe fn stream_is_capturing(
    stream: cudaStream_t,
    p_capture_status: *mut cudaStreamCaptureStatus,
) -> cudaError_t {
    let stream = to_stream(stream);
    let mut capture_status = mem::zeroed();
    let status = to_cuda(hipStreamIsCapturing(
        stream,
        &mut capture_status,
    ));
    *p_capture_status = to_cuda_stream_capture_status(capture_status);
    status
}

unsafe fn stream_get_capture_info(
    stream: cudaStream_t,
    p_capture_status: *mut cudaStreamCaptureStatus,
    p_id: *mut u64,
) -> cudaError_t {
    let stream = to_stream(stream);
    let mut capture_status = mem::zeroed();
    let status = to_cuda(hipStreamGetCaptureInfo(
        stream,
        &mut capture_status,
        p_id,
    ));
    *p_capture_status = to_cuda_stream_capture_status(capture_status);
    status
}

unsafe fn event_create(
    event: *mut cudaEvent_t,
) -> cudaError_t {
    to_cuda(hipEventCreate(
        event.cast(),
    ))
}

unsafe fn event_create_with_flags(
    event: *mut cudaEvent_t,
    flags: u32,
) -> cudaError_t {
    to_cuda(hipEventCreateWithFlags(
        event.cast(),
        flags,
    ))
}

unsafe fn event_record(
    event: cudaEvent_t,
    stream: cudaStream_t,
) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipEventRecord(
        event.cast(),
        stream,
    ))
}

unsafe fn event_query(
    event: cudaEvent_t,
) -> cudaError_t {
    to_cuda(hipEventQuery(
        event.cast(),
    ))
}

unsafe fn event_synchronize(
    event: cudaEvent_t,
) -> cudaError_t {
    to_cuda(hipEventSynchronize(
        event.cast(),
    ))
}

unsafe fn event_destroy(
    event: cudaEvent_t,
) -> cudaError_t {
    to_cuda(hipEventDestroy(
        event.cast(),
    ))
}

unsafe fn event_elapsed_time(
    ms: *mut f32,
    start: cudaEvent_t,
    end: cudaEvent_t,
) -> cudaError_t {
    to_cuda(hipEventElapsedTime(
        ms,
        start.cast(),
        end.cast(),
    ))
}

unsafe fn launch_kernel(
    func: *const ::std::os::raw::c_void,
    grid_dim: cudart::dim3,
    block_dim: cudart::dim3,
    args: *mut *mut ::std::os::raw::c_void,
    shared_mem: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    let grid_dim = to_hip_dim3(grid_dim);
    let block_dim = to_hip_dim3(block_dim);
    let stream = to_stream(stream); // TODO
    to_cuda(hipLaunchKernel(
        func,
        grid_dim,
        block_dim,
        args,
        shared_mem,
        stream,
    ))
}

unsafe fn launch_cooperative_kernel(
    func: *const ::std::os::raw::c_void,
    grid_dim: cudart::dim3,
    block_dim: cudart::dim3,
    args: *mut *mut ::std::os::raw::c_void,
    shared_mem: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    let grid_dim = to_hip_dim3(grid_dim);
    let block_dim = to_hip_dim3(block_dim);
    let stream = to_stream(stream);
    to_cuda(hipLaunchCooperativeKernel(
        func,
        grid_dim,
        block_dim,
        args,
        shared_mem as _,
        stream,
    ))
}

unsafe fn launch_host_func(
    stream: cudaStream_t,
    fn_: cudaHostFn_t,
    user_data: *mut ::std::os::raw::c_void,
) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipLaunchHostFunc(
        stream,
        fn_,
        user_data,
    ))
}

unsafe fn occupancy_max_active_blocks_per_multiprocessor_with_flags(
    num_blocks: *mut i32,
    func: *const ::std::os::raw::c_void,
    block_size: i32,
    dynamic_s_mem_size: usize,
    flags: u32,
) -> cudaError_t {
    let lib = hip_common::zluda_ext::get_cuda_library().unwrap();
    let cu_stream_synchronize = lib
        .get::<unsafe extern "C" fn(
            numBlocks: *mut ::std::os::raw::c_int,
            func: *const cuda_types::CUfunc_st,
            blockSize: ::std::os::raw::c_int,
            dynamicSMemSize: usize,
            flags: ::std::os::raw::c_uint,
        ) -> cuda_types::CUresult>(b"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags\0")
        .unwrap();
    cudaError_t((cu_stream_synchronize)(
        num_blocks,
        func.cast(),
        block_size,
        dynamic_s_mem_size,
        flags,
    ).0)
}

unsafe fn malloc_managed(
    dev_ptr: *mut *mut ::std::os::raw::c_void,
    size: usize,
    flags: u32,
) -> cudaError_t {
    to_cuda(hipMallocManaged(
        dev_ptr,
        size,
        flags,
    ))
}

unsafe fn malloc(
    dev_ptr: *mut *mut ::std::os::raw::c_void,
    size: usize,
) -> cudaError_t {
    to_cuda(hipMalloc(
        dev_ptr,
        size,
    ))
}

unsafe fn malloc_host(
    ptr: *mut *mut ::std::os::raw::c_void,
    size: usize,
) -> cudaError_t {
    to_cuda(hipMallocHost(
        ptr,
        size,
    ))
}

unsafe fn malloc_pitch(
    dev_ptr: *mut *mut ::std::os::raw::c_void,
    pitch: *mut usize,
    width: usize,
    height: usize,
) -> cudaError_t {
    to_cuda(hipMallocPitch(
        dev_ptr,
        pitch,
        width,
        height,
    ))
}

unsafe fn free(
    dev_ptr: *mut ::std::os::raw::c_void,
) -> cudaError_t {
    to_cuda(hipFree(dev_ptr))
}

unsafe fn free_host(
    ptr: *mut ::std::os::raw::c_void,
) -> cudaError_t {
    to_cuda(hipFreeHost(ptr))
}

unsafe fn free_array(
    array: cudaArray_t,
) -> cudaError_t {
    to_cuda(hipFreeArray(
        array.cast(),
    ))
}

unsafe fn free_mipmapped_array(
    mipmapped_array: cudaMipmappedArray_t,
) -> cudaError_t {
    to_cuda(hipFreeMipmappedArray(
        mipmapped_array.cast(),
    ))
}

unsafe fn host_alloc(
    p_host: *mut *mut ::std::os::raw::c_void,
    size: usize,
    flags: u32,
) -> cudaError_t {
    to_cuda(hipHostAlloc(
        p_host,
        size,
        flags,
    ))
}

unsafe fn host_register(
    ptr: *mut ::std::os::raw::c_void,
    size: usize,
    flags: u32,
) -> cudaError_t {
    to_cuda(hipHostRegister(
        ptr,
        size,
        flags,
    ))
}

unsafe fn host_unregister(
    ptr: *mut ::std::os::raw::c_void,
) -> cudaError_t {
    to_cuda(hipHostUnregister(ptr))
}

unsafe fn host_get_device_pointer(
    p_device: *mut *mut ::std::os::raw::c_void,
    p_host: *mut ::std::os::raw::c_void,
    flags: u32,
) -> cudaError_t {
    to_cuda(hipHostGetDevicePointer(
        p_device,
        p_host,
        flags,
    ))
}

unsafe fn host_get_flags(
    p_flags: *mut u32,
    p_host: *mut ::std::os::raw::c_void,
) -> cudaError_t {
    to_cuda(hipHostGetFlags(
        p_flags,
        p_host,
    ))
}

unsafe fn get_mipmapped_array_level(
    level_array: *mut cudaArray_t,
    mipmapped_array: cudaMipmappedArray_const_t,
    level: u32,
) -> cudaError_t {
    to_cuda(hipGetMipmappedArrayLevel(
        level_array.cast(),
        mipmapped_array.cast(),
        level,
    ))
}

unsafe fn mem_get_info(
    free: *mut usize,
    total: *mut usize,
) -> cudaError_t {
    to_cuda(hipMemGetInfo(
        free,
        total,
    ))
}

unsafe fn memcpy(
    dst: *mut ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpy(
        dst,
        src,
        count,
        kind,
    ))
}

unsafe fn memcpy_ptds(
    dst: *mut ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpy_spt(
        dst,
        src,
        count,
        kind,
    ))
}

unsafe fn memcpy_peer(
    dst: *mut ::std::os::raw::c_void,
    dst_device: i32,
    src: *const ::std::os::raw::c_void,
    src_device: i32,
    count: usize,
) -> cudaError_t {
    to_cuda(hipMemcpyPeer(
        dst,
        dst_device,
        src,
        src_device,
        count,
    ))
}

unsafe fn memcpy_2d(
    dst: *mut ::std::os::raw::c_void,
    dpitch: usize,
    src: *const ::std::os::raw::c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpy2D(
        dst,
        dpitch,
        src,
        spitch,
        width,
        height,
        kind,
    ))
}

unsafe fn memcpy_2d_ptds(
    dst: *mut ::std::os::raw::c_void,
    dpitch: usize,
    src: *const ::std::os::raw::c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpy2D_spt(
        dst,
        dpitch,
        src,
        spitch,
        width,
        height,
        kind,
    ))
}

unsafe fn memcpy_2d_to_array(
    dst: cudaArray_t,
    w_offset: usize,
    h_offset: usize,
    src: *const ::std::os::raw::c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpy2DToArray(
        dst.cast(),
        w_offset,
        h_offset,
        src,
        spitch,
        width,
        height,
        kind,
    ))
}

unsafe fn memcpy_2d_to_array_ptds(
    dst: cudaArray_t,
    w_offset: usize,
    h_offset: usize,
    src: *const ::std::os::raw::c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpy2DToArray_spt(
        dst.cast(),
        w_offset,
        h_offset,
        src,
        spitch,
        width,
        height,
        kind,
    ))
}

unsafe fn memcpy_2d_from_array(
    dst: *mut ::std::os::raw::c_void,
    dpitch: usize,
    src: cudaArray_const_t,
    w_offset: usize,
    h_offset: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpy2DFromArray(
        dst,
        dpitch,
        src.cast(),
        w_offset,
        h_offset,
        width,
        height,
        kind,
    ))
}

unsafe fn memcpy_2d_from_array_ptds(
    dst: *mut ::std::os::raw::c_void,
    dpitch: usize,
    src: cudaArray_const_t,
    w_offset: usize,
    h_offset: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpy2DFromArray_spt(
        dst,
        dpitch,
        src.cast(),
        w_offset,
        h_offset,
        width,
        height,
        kind,
    ))
}

unsafe fn memcpy_to_symbol(
    symbol: *const ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpyToSymbol(
        symbol,
        src,
        count,
        offset,
        kind,
    ))
}

unsafe fn memcpy_to_symbol_ptds(
    symbol: *const ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpyToSymbol_spt(
        symbol,
        src,
        count,
        offset,
        kind,
    ))
}

unsafe fn memcpy_from_symbol(
    dst: *mut ::std::os::raw::c_void,
    symbol: *const ::std::os::raw::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpyFromSymbol(
        dst,
        symbol,
        count,
        offset,
        kind,
    ))
}

unsafe fn memcpy_from_symbol_ptds(
    dst: *mut ::std::os::raw::c_void,
    symbol: *const ::std::os::raw::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpyFromSymbol_spt(
        dst,
        symbol,
        count,
        offset,
        kind,
    ))
}

unsafe fn memcpy_async(
    dst: *mut ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    let stream = to_stream(stream);
    to_cuda(hipMemcpyAsync(
        dst,
        src,
        count,
        kind,
        stream,
    ))
}

unsafe fn memcpy_peer_async(
    dst: *mut ::std::os::raw::c_void,
    dst_device: i32,
    src: *const ::std::os::raw::c_void,
    src_device: i32,
    count: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipMemcpyPeerAsync(
        dst,
        dst_device,
        src,
        src_device,
        count,
        stream,
    ))
}

unsafe fn memcpy_2d_async(
    dst: *mut ::std::os::raw::c_void,
    dpitch: usize,
    src: *const ::std::os::raw::c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    let stream = to_stream(stream);
    to_cuda(hipMemcpy2DAsync(
        dst,
        dpitch,
        src,
        spitch,
        width,
        height,
        kind,
        stream,
    ))
}

unsafe fn memcpy_2d_to_array_async(
    dst: cudaArray_t,
    w_offset: usize,
    h_offset: usize,
    src: *const ::std::os::raw::c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    let stream = to_stream(stream);
    to_cuda(hipMemcpy2DToArrayAsync(
        dst.cast(),
        w_offset,
        h_offset,
        src,
        spitch,
        width,
        height,
        kind,
        stream,
    ))
}

unsafe fn memcpy_2d_from_array_async(
    dst: *mut ::std::os::raw::c_void,
    dpitch: usize,
    src: cudaArray_const_t,
    w_offset: usize,
    h_offset: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    let stream = to_stream(stream);
    to_cuda(hipMemcpy2DFromArrayAsync(
        dst,
        dpitch,
        src.cast(),
        w_offset,
        h_offset,
        width,
        height,
        kind,
        stream,
    ))
}

unsafe fn memcpy_to_symbol_async(
    symbol: *const ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    let stream = to_stream(stream);
    to_cuda(hipMemcpyToSymbolAsync(
        symbol,
        src,
        count,
        offset,
        kind,
        stream,
    ))
}

unsafe fn memcpy_from_symbol_async(
    dst: *mut ::std::os::raw::c_void,
    symbol: *const ::std::os::raw::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    let stream = to_stream(stream);
    to_cuda(hipMemcpyFromSymbolAsync(
        dst,
        symbol,
        count,
        offset,
        kind,
        stream,
    ))
}

unsafe fn memset(
    dev_ptr: *mut ::std::os::raw::c_void,
    value: i32,
    count: usize,
) -> cudaError_t {
    to_cuda(hipMemset(
        dev_ptr,
        value,
        count,
    ))
}

unsafe fn memset_ptds(
    dev_ptr: *mut ::std::os::raw::c_void,
    value: i32,
    count: usize,
) -> cudaError_t {
    to_cuda(hipMemset_spt(
        dev_ptr,
        value,
        count,
    ))
}

unsafe fn memset_2d(
    dev_ptr: *mut ::std::os::raw::c_void,
    pitch: usize,
    value: i32,
    width: usize,
    height: usize,
) -> cudaError_t {
    to_cuda(hipMemset2D(
        dev_ptr,
        pitch,
        value,
        width,
        height,
    ))
}

unsafe fn memset_2d_ptds(
    dev_ptr: *mut ::std::os::raw::c_void,
    pitch: usize,
    value: i32,
    width: usize,
    height: usize,
) -> cudaError_t {
    to_cuda(hipMemset2D_spt(
        dev_ptr,
        pitch,
        value,
        width,
        height,
    ))
}

unsafe fn memset_async(
    dev_ptr: *mut ::std::os::raw::c_void,
    value: i32,
    count: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipMemsetAsync(
        dev_ptr,
        value,
        count,
        stream,
    ))
}

unsafe fn memset_2d_async(
    dev_ptr: *mut ::std::os::raw::c_void,
    pitch: usize,
    value: i32,
    width: usize,
    height: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipMemset2DAsync(
        dev_ptr,
        pitch,
        value,
        width,
        height,
        stream,
    ))
}

unsafe fn get_symbol_address(
    dev_ptr: *mut *mut ::std::os::raw::c_void,
    symbol: *const ::std::os::raw::c_void,
) -> cudaError_t {
    to_cuda(hipGetSymbolAddress(
        dev_ptr,
        symbol,
    ))
}

unsafe fn get_symbol_size(
    size: *mut usize,
    symbol: *const ::std::os::raw::c_void,
) -> cudaError_t {
    to_cuda(hipGetSymbolSize(
        size,
        symbol,
    ))
}

unsafe fn mem_prefetch_async(
    dev_ptr: *const ::std::os::raw::c_void,
    count: usize,
    dst_device: i32,
    stream: cudaStream_t,
) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipMemPrefetchAsync(
        dev_ptr,
        count,
        dst_device,
        stream,
    ))
}

unsafe fn memcpy_to_array(
    dst: cudaArray_t,
    w_offset: usize,
    h_offset: usize,
    src: *const ::std::os::raw::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpyToArray(
        dst.cast(),
        w_offset,
        h_offset,
        src,
        count,
        kind,
    ))
}

unsafe fn memcpy_from_array(
    dst: *mut ::std::os::raw::c_void,
    src: cudaArray_const_t,
    w_offset: usize,
    h_offset: usize,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpyFromArray(
        dst,
        src.cast(),
        w_offset,
        h_offset,
        count,
        kind,
    ))
}

unsafe fn memcpy_from_array_ptds(
    dst: *mut ::std::os::raw::c_void,
    src: cudaArray_const_t,
    w_offset: usize,
    h_offset: usize,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpyFromArray_spt(
        dst,
        src.cast(),
        w_offset,
        h_offset,
        count,
        kind,
    ))
}

unsafe fn memcpy_to_array_async(
    dst: cudaArray_t,
    w_offset: usize,
    h_offset: usize,
    src: *const ::std::os::raw::c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    let stream = to_stream(stream);
    to_cuda(hipMemcpy2DToArrayAsync(
        dst.cast(),
        w_offset,
        h_offset,
        src,
        count,
        w_offset,
        h_offset,
        kind,
        stream,
    ))
}

unsafe fn memcpy_from_array_async(
    dst: *mut ::std::os::raw::c_void,
    src: cudaArray_const_t,
    w_offset: usize,
    h_offset: usize,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    let stream = to_stream(stream);
    to_cuda(hipMemcpy2DFromArrayAsync(
        dst,
        count,
        src.cast(),
        w_offset,
        h_offset,
        w_offset,
        h_offset,
        kind, 
        stream,
    ))
}

unsafe fn malloc_async(
    dev_ptr: *mut *mut ::std::os::raw::c_void,
    size: usize,
    h_stream: cudaStream_t,
) -> cudaError_t {
    let h_stream = to_stream(h_stream);
    to_cuda(hipMallocAsync(
        dev_ptr,
        size,
        h_stream,
    ))
}

unsafe fn free_async(
    dev_ptr: *mut ::std::os::raw::c_void,
    h_stream: cudaStream_t,
) -> cudaError_t {
    let h_stream = to_stream(h_stream);
    to_cuda(hipFreeAsync(
        dev_ptr,
        h_stream,
    ))
}

unsafe fn mem_pool_trim_to(
    mem_pool: cudaMemPool_t,
    min_bytes_to_keep: usize,
) -> cudaError_t {
    to_cuda(hipMemPoolTrimTo(
        mem_pool.cast(),
        min_bytes_to_keep,
    ))
}

unsafe fn mem_pool_set_attribute(
    mem_pool: cudaMemPool_t,
    attr: cudaMemPoolAttr,
    value: *mut ::std::os::raw::c_void,
) -> cudaError_t {
    let attr = to_hip_mem_pool_attr(attr);
    to_cuda(hipMemPoolSetAttribute(
        mem_pool.cast(),
        attr,
        value,
    ))
}

unsafe fn mem_pool_get_attribute(
    mem_pool: cudaMemPool_t,
    attr: cudaMemPoolAttr,
    value: *mut ::std::os::raw::c_void,
) -> cudaError_t {
    let attr = to_hip_mem_pool_attr(attr);
    to_cuda(hipMemPoolGetAttribute(
        mem_pool.cast(),
        attr,
        value,
    ))
}

unsafe fn mem_pool_destroy(
    mem_pool: cudaMemPool_t,
) -> cudaError_t {
    to_cuda(hipMemPoolDestroy(
        mem_pool.cast(),
    ))
}

unsafe fn malloc_from_pool_async(
    ptr: *mut *mut ::std::os::raw::c_void,
    size: usize,
    mem_pool: cudaMemPool_t,
    stream: cudaStream_t,
) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipMallocFromPoolAsync(
        ptr,
        size,
        mem_pool.cast(),
        stream,
    ))
}

unsafe fn mem_pool_export_pointer(
    export_data: *mut cudaMemPoolPtrExportData,
    ptr: *mut ::std::os::raw::c_void,
) -> cudaError_t {
    to_cuda(hipMemPoolExportPointer(
        export_data.cast(),
        ptr,
    ))
}

unsafe fn mem_pool_import_pointer(
    ptr: *mut *mut ::std::os::raw::c_void,
    mem_pool: cudaMemPool_t,
    export_data: *mut cudaMemPoolPtrExportData,
) -> cudaError_t {
    to_cuda(hipMemPoolImportPointer(
        ptr,
        mem_pool.cast(),
        export_data.cast(),
    ))
}

unsafe fn device_can_access_peer(
    can_access_peer: *mut i32,
    device: i32,
    peer_device: i32,
) -> cudaError_t {
    to_cuda(hipDeviceCanAccessPeer(
        can_access_peer,
        device,
        peer_device,
    ))
}

unsafe fn device_enable_peer_access(
    peer_device: i32,
    flags: u32,
) -> cudaError_t {
    to_cuda(hipDeviceEnablePeerAccess(
        peer_device,
        flags,
    ))
}

unsafe fn device_disable_peer_access(
    peer_device: i32,
) -> cudaError_t {
    to_cuda(hipDeviceDisablePeerAccess(peer_device))
}

unsafe fn graphics_unregister_resource(
    resource: cudaGraphicsResource_t,
) -> cudaError_t {
    to_cuda(hipGraphicsUnregisterResource(
        resource.cast(),
    ))
}

unsafe fn graphics_map_resources(
    count: i32,
    resources: *mut cudaGraphicsResource_t,
    stream: cudaStream_t,
) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipGraphicsMapResources(
        count,
        resources.cast(),
        stream,
    ))
}

unsafe fn graphics_unmap_resources(
    count: i32,
    resources: *mut cudaGraphicsResource_t,
    stream: cudaStream_t,
) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipGraphicsUnmapResources(
        count,
        resources.cast(),
        stream,
    ))
}

unsafe fn graphics_resource_get_mapped_pointer(
    dev_ptr: *mut *mut ::std::os::raw::c_void,
    size: *mut usize,
    resource: cudaGraphicsResource_t,
) -> cudaError_t {
    to_cuda(hipGraphicsResourceGetMappedPointer(
        dev_ptr,
        size,
        resource.cast(),
    ))
}

unsafe fn graphics_sub_resource_get_mapped_array(
    array: *mut cudaArray_t,
    resource: cudaGraphicsResource_t,
    array_index: u32,
    mip_level: u32,
) -> cudaError_t {
    to_cuda(hipGraphicsSubResourceGetMappedArray(
        array.cast(),
        resource.cast(),
        array_index,
        mip_level,
    ))
}

unsafe fn graph_create(
    p_graph: *mut cudaGraph_t,
    flags: u32,
) -> cudaError_t {
    to_cuda(hipGraphCreate(
        p_graph.cast(),
        flags,
    ))
}

unsafe fn graph_kernel_node_copy_attributes(
    h_src: cudaGraphNode_t,
    h_dst: cudaGraphNode_t,
) -> cudaError_t {
    to_cuda(hipGraphKernelNodeCopyAttributes(
        h_src.cast(),
        h_dst.cast(),
    ))
}

unsafe fn graph_add_memcpy_node_to_symbol(
    p_graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    p_dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    symbol: *const ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipGraphAddMemcpyNodeToSymbol(
        p_graph_node.cast(),
        graph.cast(),
        p_dependencies.cast(),
        num_dependencies,
        symbol,
        src,
        count,
        offset,
        kind,
    ))
}

unsafe fn graph_add_memcpy_node_from_symbol(
    p_graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    p_dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    dst: *mut ::std::os::raw::c_void,
    symbol: *const ::std::os::raw::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipGraphAddMemcpyNodeFromSymbol(
        p_graph_node.cast(),
        graph.cast(),
        p_dependencies.cast(),
        num_dependencies,
        dst,
        symbol,
        count,
        offset,
        kind,
    ))
}

unsafe fn graph_add_memcpy_node_1d(
    p_graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    p_dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    dst: *mut ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipGraphAddMemcpyNode1D(
        p_graph_node.cast(),
        graph.cast(),
        p_dependencies.cast(),
        num_dependencies,
        dst,
        src,
        count,
        kind,
    ))
}

unsafe fn graph_memcpy_node_set_params_to_symbol(
    node: cudaGraphNode_t,
    symbol: *const ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipGraphMemcpyNodeSetParamsToSymbol(
        node.cast(),
        symbol,
        src,
        count,
        offset,
        kind,
    ))
}

unsafe fn graph_memcpy_node_set_params_from_symbol(
    node: cudaGraphNode_t,
    dst: *mut ::std::os::raw::c_void,
    symbol: *const ::std::os::raw::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipGraphMemcpyNodeSetParamsFromSymbol(
        node.cast(),
        dst,
        symbol,
        count,
        offset,
        kind,
    ))
}

unsafe fn graph_memcpy_node_set_params_1d(
    node: cudaGraphNode_t,
    dst: *mut ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipGraphMemcpyNodeSetParams1D(
        node.cast(),
        dst,
        src,
        count,
        kind,
    ))
}

unsafe fn graph_add_child_graph_node(
    p_graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    p_dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    child_graph: cudaGraph_t,
) -> cudaError_t {
    to_cuda(hipGraphAddChildGraphNode(
        p_graph_node.cast(),
        graph.cast(),
        p_dependencies.cast(),
        num_dependencies,
        child_graph.cast(),
    ))
}

unsafe fn graph_child_graph_node_get_graph(
    node: cudaGraphNode_t,
    p_graph: *mut cudaGraph_t,
) -> cudaError_t {
    to_cuda(hipGraphChildGraphNodeGetGraph(
        node.cast(),
        p_graph.cast(),
    ))
}

unsafe fn graph_add_empty_node(
    p_graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    p_dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
) -> cudaError_t {
    to_cuda(hipGraphAddEmptyNode(
        p_graph_node.cast(),
        graph.cast(),
        p_dependencies.cast(),
        num_dependencies,
    ))
}

unsafe fn graph_add_event_record_node(
    p_graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    p_dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    event: cudaEvent_t,
) -> cudaError_t {
    to_cuda(hipGraphAddEventRecordNode(
        p_graph_node.cast(),
        graph.cast(),
        p_dependencies.cast(),
        num_dependencies,
        event.cast(),
    ))
}

unsafe fn graph_event_record_node_get_event(
    node: cudaGraphNode_t,
    event_out: *mut cudaEvent_t,
) -> cudaError_t {
    to_cuda(hipGraphEventRecordNodeGetEvent(
        node.cast(),
        event_out.cast(),
    ))
}

unsafe fn graph_event_record_node_set_event(
    node: cudaGraphNode_t,
    event: cudaEvent_t,
) -> cudaError_t {
    to_cuda(hipGraphEventRecordNodeSetEvent(
        node.cast(),
        event.cast(),
    ))
}

unsafe fn graph_add_mem_free_node(
    p_graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    p_dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    dptr: *mut ::std::os::raw::c_void,
) -> cudaError_t {
    to_cuda(hipGraphAddMemFreeNode(
        p_graph_node.cast(),
        graph.cast(),
        p_dependencies.cast(),
        num_dependencies,
        dptr,
    ))
}

unsafe fn graph_mem_free_node_get_params(
    node: cudaGraphNode_t,
    dptr_out: *mut ::std::os::raw::c_void,
) -> cudaError_t {
    to_cuda(hipGraphMemFreeNodeGetParams(
        node.cast(),
        dptr_out,
    ))
}

unsafe fn graph_add_event_wait_node(
    p_graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    p_dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    event: cudaEvent_t,
) -> cudaError_t {
    to_cuda(hipGraphAddEventWaitNode(
        p_graph_node.cast(),
        graph.cast(),
        p_dependencies.cast(),
        num_dependencies,
        event.cast(),
    ))
}

unsafe fn graph_event_wait_node_get_event(
    node: cudaGraphNode_t,
    event_out: *mut cudaEvent_t,
) -> cudaError_t {
    to_cuda(hipGraphEventWaitNodeGetEvent(
        node.cast(),
        event_out.cast(),
    ))
}

unsafe fn graph_event_wait_node_set_event(
    node: cudaGraphNode_t,
    event: cudaEvent_t,
) -> cudaError_t {
    to_cuda(hipGraphEventWaitNodeSetEvent(
        node.cast(),
        event.cast(),
    ))
}

unsafe fn device_graph_mem_trim(
    device: i32,
) -> cudaError_t {
    to_cuda(hipDeviceGraphMemTrim(device))
}

unsafe fn graph_clone(
    p_graph_clone: *mut cudaGraph_t,
    original_graph: cudaGraph_t,
) -> cudaError_t {
    to_cuda(hipGraphClone(
        p_graph_clone.cast(),
        original_graph.cast(),
    ))
}

unsafe fn graph_node_find_in_close(
    p_node: *mut cudaGraphNode_t,
    original_node: cudaGraphNode_t,
    cloned_graph: cudaGraph_t,
) -> cudaError_t {
    to_cuda(hipGraphNodeFindInClone(
        p_node.cast(),
        original_node.cast(),
        cloned_graph.cast(),
    ))
}

unsafe fn graph_get_nodes(
    graph: cudaGraph_t,
    nodes: *mut cudaGraphNode_t,
    num_nodes: *mut usize,
) -> cudaError_t {
    to_cuda(hipGraphGetNodes(
        graph.cast(),
        nodes.cast(),
        num_nodes,
    ))
}

unsafe fn graph_get_root_nodes(
    graph: cudaGraph_t,
    p_root_nodes: *mut cudaGraphNode_t,
    p_num_root_nodes: *mut usize,
) -> cudaError_t {
    to_cuda(hipGraphGetRootNodes(
        graph.cast(),
        p_root_nodes.cast(),
        p_num_root_nodes,
    ))
}

unsafe fn graph_get_edges(
    graph: cudaGraph_t,
    from: *mut cudaGraphNode_t,
    to: *mut cudaGraphNode_t,
    num_edges: *mut usize,
) -> cudaError_t {
    to_cuda(hipGraphGetEdges(
        graph.cast(),
        from.cast(),
        to.cast(),
        num_edges,
    ))
}

unsafe fn graph_node_get_dependencies(
    node: cudaGraphNode_t,
    p_dependencies: *mut cudaGraphNode_t,
    p_num_dependencies: *mut usize,
) -> cudaError_t {
    to_cuda(hipGraphNodeGetDependencies(
        node.cast(),
        p_dependencies.cast(),
        p_num_dependencies,
    ))
}

unsafe fn graph_node_get_dependent_nodes(
    node: cudaGraphNode_t,
    p_dependent_nodes: *mut cudaGraphNode_t,
    p_num_dependent_nodes: *mut usize,
) -> cudaError_t {
    to_cuda(hipGraphNodeGetDependentNodes(
        node.cast(),
        p_dependent_nodes.cast(),
        p_num_dependent_nodes,
    ))
}

unsafe fn graph_node_get_enabled(
    h_graph_exec: cudaGraphExec_t,
    h_node: cudaGraphNode_t,
    is_enabled: *mut u32,
) -> cudaError_t {
    to_cuda(hipGraphNodeGetEnabled(
        h_graph_exec.cast(),
        h_node.cast(),
        is_enabled,
    ))
}

unsafe fn graph_node_set_enabled(
    h_graph_exec: cudaGraphExec_t,
    h_node: cudaGraphNode_t,
    is_enabled: u32,
) -> cudaError_t {
    to_cuda(hipGraphNodeSetEnabled(
        h_graph_exec.cast(),
        h_node.cast(),
        is_enabled,
    ))
}

unsafe fn graph_add_dependencies(
    graph: cudaGraph_t,
    from: *const cudaGraphNode_t,
    to: *const cudaGraphNode_t,
    num_dependencies: usize,
) -> cudaError_t {
    to_cuda(hipGraphAddDependencies(
        graph.cast(),
        from.cast(),
        to.cast(),
        num_dependencies,
    ))
}

unsafe fn graph_remove_dependencies(
    graph: cudaGraph_t,
    from: *const cudaGraphNode_t,
    to: *const cudaGraphNode_t,
    num_dependencies: usize,
) -> cudaError_t {
    to_cuda(hipGraphRemoveDependencies(
        graph.cast(),
        from.cast(),
        to.cast(),
        num_dependencies,
    ))
}

unsafe fn graph_destroy_node(
    node: cudaGraphNode_t,
) -> cudaError_t {
    to_cuda(hipGraphDestroyNode(
        node.cast(),
    ))
}

unsafe fn graph_instantiate(
    p_graph_exec: *mut cudaGraphExec_t,
    graph: cudaGraph_t,
    p_error_node: *mut cudaGraphNode_t,
    p_log_buffer: *mut ::std::os::raw::c_char,
    buffer_size: usize,
) -> cudaError_t {
    to_cuda(hipGraphInstantiate(
        p_graph_exec.cast(),
        graph.cast(),
        p_error_node.cast(),
        p_log_buffer,
        buffer_size,
    ))
}

unsafe fn graph_instantiate_with_flags(
    p_graph_exec: *mut cudaGraphExec_t,
    graph: cudaGraph_t,
    flags: u64,
) -> cudaError_t {
    to_cuda(hipGraphInstantiateWithFlags(
        p_graph_exec.cast(),
        graph.cast(),
        flags,
    ))
}

unsafe fn graph_exec_memcpy_node_set_params_to_symbol(
    h_graph_exec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    symbol: *const ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipGraphExecMemcpyNodeSetParamsToSymbol(
        h_graph_exec.cast(),
        node.cast(),
        symbol,
        src,
        count,
        offset,
        kind,
    ))
}

unsafe fn graph_exec_memcpy_node_set_params_from_symbol(
    h_graph_exec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    dst: *mut ::std::os::raw::c_void,
    symbol: *const ::std::os::raw::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipGraphExecMemcpyNodeSetParamsFromSymbol(
        h_graph_exec.cast(),
        node.cast(),
        dst,
        symbol,
        count,
        offset,
        kind,
    ))
}

unsafe fn graph_exec_memcpy_node_set_params_1d(
    h_graph_exec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    dst: *mut ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipGraphExecMemcpyNodeSetParams1D(
        h_graph_exec.cast(),
        node.cast(),
        dst,
        src,
        count,
        kind,
    ))
}

unsafe fn graph_exec_child_graph_node_set_params(
    h_graph_exec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    child_graph: cudaGraph_t,
) -> cudaError_t {
    to_cuda(hipGraphExecChildGraphNodeSetParams(
        h_graph_exec.cast(),
        node.cast(),
        child_graph.cast(),
    ))
}

unsafe fn graph_exec_event_record_node_set_event(
    h_graph_exec: cudaGraphExec_t,
    h_node: cudaGraphNode_t,
    event: cudaEvent_t,
) -> cudaError_t {
    to_cuda(hipGraphExecEventRecordNodeSetEvent(
        h_graph_exec.cast(),
        h_node.cast(),
        event.cast(),
    ))
}

unsafe fn graph_exec_event_wait_node_set_event(
    h_graph_exec: cudaGraphExec_t,
    h_node: cudaGraphNode_t,
    event: cudaEvent_t,
) -> cudaError_t {
    to_cuda(hipGraphExecEventWaitNodeSetEvent(
        h_graph_exec.cast(),
        h_node.cast(),
        event.cast(),
    ))
}

unsafe fn graph_upload(
    graph_exec: cudaGraphExec_t,
    stream: cudaStream_t,
) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipGraphUpload(
        graph_exec.cast(),
        stream,
    ))
}

unsafe fn graph_launch(
    graph_exec: cudaGraphExec_t,
    stream: cudaStream_t,
) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipGraphLaunch(
        graph_exec.cast(),
        stream,
    ))
}

unsafe fn graph_launch_ptsz(
    graph_exec: cudaGraphExec_t,
    stream: cudaStream_t,
) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipGraphLaunch_spt(
        graph_exec.cast(),
        stream,
    ))
}

unsafe fn graph_exec_destroy(
    graph_exec: cudaGraphExec_t,
) -> cudaError_t {
    to_cuda(hipGraphExecDestroy(
        graph_exec.cast(),
    ))
}

unsafe fn graph_destroy(
    graph: cudaGraph_t,
) -> cudaError_t {
    to_cuda(hipGraphDestroy(
        graph.cast(),
    ))
}

unsafe fn graph_debug_dot_print(
    graph: cudaGraph_t,
    path: *const ::std::os::raw::c_char,
    flags: u32,
) -> cudaError_t {
    to_cuda(hipGraphDebugDotPrint(
        graph.cast(),
        path,
        flags,
    ))
}

unsafe fn user_object_create(
    object_out: *mut cudaUserObject_t,
    ptr: *mut ::std::os::raw::c_void,
    destroy: cudaHostFn_t,
    initial_refcount: u32,
    flags: u32,
) -> cudaError_t {
    to_cuda(hipUserObjectCreate(
        object_out.cast(),
        ptr,
        destroy,
        initial_refcount,
        flags,
    ))
}

unsafe fn user_object_retain(
    object: cudaUserObject_t,
    count: u32,
) -> cudaError_t {
    to_cuda(hipUserObjectRetain(
        object.cast(),
        count,
    ))
}

unsafe fn user_object_release(
    object: cudaUserObject_t,
    count: u32,
) -> cudaError_t {
    to_cuda(hipUserObjectRelease(
        object.cast(),
        count,
    ))
}

unsafe fn graph_retain_user_object(
    graph: cudaGraph_t,
    object: cudaUserObject_t,
    count: u32,
    flags: u32,
) -> cudaError_t {
    to_cuda(hipGraphRetainUserObject(
        graph.cast(),
        object.cast(),
        count,
        flags,
    ))
}

unsafe fn graph_release_user_object(
    graph: cudaGraph_t,
    object: cudaUserObject_t,
    count: u32,
) -> cudaError_t {
    to_cuda(hipGraphReleaseUserObject(
        graph.cast(),
        object.cast(),
        count,
    ))
}

unsafe fn profiler_start() -> cudaError_t {
    to_cuda(hipProfilerStart())
}

unsafe fn profiler_stop() -> cudaError_t {
    to_cuda(hipProfilerStop())
}
