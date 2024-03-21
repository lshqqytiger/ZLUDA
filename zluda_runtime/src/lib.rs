mod cudart;
pub use cudart::*;

use hip_runtime_sys::*;

#[cfg(debug_assertions)]
fn unsupported() -> cudaError_t {
    println!("UNSUPPORTED");
    unimplemented!()
}

#[cfg(not(debug_assertions))]
fn unsupported() -> cudaError_t {
    println!("UNSUPPORTED");
    cudaError::cudaErrorNotSupported
}

fn to_cuda(status: hipError_t) -> cudaError_t {
    match status {
        hipError_t::hipSuccess => cudaError_t::cudaSuccess,
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
