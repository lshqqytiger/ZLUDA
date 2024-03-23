mod cudart;
pub use cudart::*;

use hip_runtime_sys::*;

#[cfg(debug_assertions)]
fn unsupported() -> cudaError_t {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
fn unsupported() -> cudaError_t {
    cudaError::cudaErrorNotSupported
}

fn to_cuda(status: hipError_t) -> cudaError_t {
    match status {
        hipError_t::hipSuccess => cudaError_t::cudaSuccess,
        err => panic!("[ZLUDA] HIP_RUNTIME failed: {}", err.0),
    }
}

fn to_hip(status: cudaError_t) -> hipError_t {
    match status {
        cudaError_t::cudaSuccess => hipError_t::hipSuccess,
        err => panic!("[ZLUDA] HIP_RUNTIME failed: {}", err.0),
    }
}

fn to_hip_limit(limit: cudaLimit) -> hipLimit_t {
    match limit {
        _ => panic!()
    }
}

fn to_hip_func_cache(func_cache: cudaFuncCache) -> hipFuncCache_t {
    match func_cache {
        _ => panic!()
    }
}

fn to_cuda_func_cache(func_cache: hipFuncCache_t) -> cudaFuncCache {
    match func_cache {
        _ => panic!()
    }
}

fn to_hip_shared_mem_config(shared_mem_config: cudaSharedMemConfig) -> hipSharedMemConfig {
    match shared_mem_config {
        _ => panic!()
    }
}

fn to_cuda_shared_mem_config(shared_mem_config: hipSharedMemConfig) -> cudaSharedMemConfig {
    match shared_mem_config {
        _ => panic!()
    }
}

unsafe fn device_reset() -> cudaError_t {
    to_cuda(hipDeviceReset())
}

unsafe fn device_synchronize() -> cudaError_t {
    to_cuda(hipDeviceSynchronize())
}

unsafe fn device_set_limit(
    limit: cudaLimit,
    value: usize,
) -> cudaError_t {
    let limit = to_hip_limit(limit);
    to_cuda(hipDeviceSetLimit(limit, value))
}

unsafe fn device_get_limit(
    p_value: *mut usize,
    limit: cudaLimit,
) -> cudaError_t {
    let limit = to_hip_limit(limit);
    to_cuda(hipDeviceGetLimit(p_value, limit))
}

unsafe fn device_get_cache_config(
    p_cache_config: *mut cudaFuncCache,
) -> cudaError_t {
    let mut out_cache_confg = to_hip_func_cache(*p_cache_config);
    let status = to_cuda(hipDeviceGetCacheConfig(&mut out_cache_confg));
    *p_cache_config = to_cuda_func_cache(out_cache_confg);
    status
}

unsafe fn device_set_cache_config(
    cache_config: cudaFuncCache,
) -> cudaError_t {
    let cache_config = to_hip_func_cache(cache_config);
    to_cuda(hipDeviceSetCacheConfig(cache_config))
}

unsafe fn device_get_shared_mem_config(
    p_config: *mut cudaSharedMemConfig,
) -> cudaError_t {
    let mut out_config = to_hip_shared_mem_config(*p_config);
    let status = to_cuda(hipDeviceGetSharedMemConfig(&mut out_config));
    *p_config = to_cuda_shared_mem_config(out_config);
    status
}

unsafe fn device_set_shared_mem_config(
    config: cudaSharedMemConfig,
) -> cudaError_t {
    let config = to_hip_shared_mem_config(config);
    to_cuda(hipDeviceSetSharedMemConfig(config))
}

unsafe fn device_get_by_pci_bus_id(
    device: *mut i32,
    pci_bus_id: *const ::std::os::raw::c_char,
) -> cudaError_t {
    to_cuda(hipDeviceGetByPCIBusId(device, pci_bus_id))
}

unsafe fn device_get_pci_bus_id(
    pci_bus_id: *mut ::std::os::raw::c_char,
    len: i32,
    device: i32,
) -> cudaError_t {
    to_cuda(hipDeviceGetPCIBusId(pci_bus_id, len, device))
}

unsafe fn ipc_get_event_handle(
    handle: *mut cudaIpcEventHandle_t,
    event: cudaEvent_t,
) -> cudaError_t {
    to_cuda(hipIpcGetEventHandle(
        handle.cast(),
        event as _,
    ))
}

unsafe fn ipc_open_event_handle(
    _event: *mut cudaEvent_t,
    _handle: cudaIpcEventHandle_t,
) -> cudaError_t {
    panic!()
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

unsafe fn ipc_open_mem_handle(
    _dev_ptr: *mut *mut ::std::os::raw::c_void,
    _handle: cudaIpcMemHandle_t,
    _flags: u32,
) -> cudaError_t {
    panic!()
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
    to_cuda(hipStreamCreate(
        p_stream.cast(),
    ))
}

unsafe fn stream_create_with_flags(
    p_stream: *mut cudaStream_t,
    flags: u32,
) -> cudaError_t {
    to_cuda(hipStreamCreateWithFlags(
        p_stream.cast(),
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
        ptr,
        pitch,
        width,
        height,
    ))
}

unsafe fn malloc_array(
    array: *mut cudaArray_t,
    desc: *const cudaChannelFormatDesc,
    width: usize,
    height: usize,
    flags: u32,
) -> cudaError_t {
    to_cuda(hipMallocArray(
        array.cast(),
        desc.cast(),
        width,
        height,
        flags,
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

unsafe fn mem_get_info(
    free: *mut usize,
    total: *mut usize,
) -> cudaError_t {
    to_cuda(hipMemGetInfo(
        free,
        total,
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
