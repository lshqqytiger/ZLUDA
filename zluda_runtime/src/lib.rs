mod cudart;
pub use cudart::*;

use hip_runtime_sys::*;
use std::ptr;

fn no_corresponding_function() -> cudaError_t {
    cudaError::cudaSuccess
}

fn to_cuda(status: hipError_t) -> cudaError_t {
    match status {
        hipError_t::hipSuccess => cudaError_t::cudaSuccess,
        hipError_t::hipErrorInvalidResourceHandle => cudaError_t::cudaErrorInvalidResourceHandle,
        hipError_t::hipErrorNotSupported => cudaError_t::cudaErrorNotSupported,
        err => panic!("[ZLUDA] HIP_RUNTIME failed: {}", err.0),
    }
}

fn to_hip(status: cudaError_t) -> hipError_t {
    match status {
        cudaError_t::cudaSuccess => hipError_t::hipSuccess,
        cudaError_t::cudaErrorInvalidResourceHandle => hipError_t::hipErrorInvalidResourceHandle,
        cudaError_t::cudaErrorNotSupported => hipError_t::hipErrorNotSupported,
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

fn to_cuda_read_mode(read_mode: hipTextureReadMode) -> cudaTextureReadMode {
    match read_mode {
        _ => panic!()
    }
}

fn to_hip_read_mode(read_mode: cudaTextureReadMode) -> hipTextureReadMode {
    match read_mode {
        _ => panic!()
    }
}

fn to_cuda_filter_mode(filter_mode: hipTextureFilterMode) -> cudaTextureFilterMode {
    match filter_mode {
        _ => panic!()
    }
}

fn to_hip_filter_mode(filter_mode: cudaTextureFilterMode) -> hipTextureFilterMode {
    match filter_mode {
        _ => panic!()
    }
}

fn to_cuda_address_mode(address_mode: hipTextureAddressMode) -> cudaTextureAddressMode {
    match address_mode {
        _ => panic!()
    }
}

fn to_hip_address_mode(address_mode: cudaTextureAddressMode) -> hipTextureAddressMode {
    match address_mode {
        _ => panic!()
    }
}

fn to_cuda_channel_format_kind(channel_format_kind: hipChannelFormatKind) -> cudaChannelFormatKind {
    match channel_format_kind {
        _ => panic!()
    }
}

fn to_hip_channel_format_kind(channel_format_kind: cudaChannelFormatKind) -> hipChannelFormatKind {
    match channel_format_kind {
        _ => panic!()
    }
}

fn to_cuda_resource_type(resource_type: hipResourceType) -> cudaResourceType {
    match resource_type {
        _ => panic!()
    }
}

fn to_hip_resource_type(resource_type: cudaResourceType) -> hipResourceType {
    match resource_type {
        _ => panic!()
    }
}

fn to_cuda_resource_view_format(resource_view_format: hipResourceViewFormat) -> cudaResourceViewFormat {
    match resource_view_format {
        _ => panic!()
    }
}

fn to_hip_resource_view_format(resource_view_format: cudaResourceViewFormat) -> hipResourceViewFormat {
    match resource_view_format {
        _ => panic!()
    }
}

fn to_hip_graph_mem_attribute_type(graph_mem_attribute_type: cudaGraphMemAttributeType) -> hipGraphMemAttributeType {
    match graph_mem_attribute_type {
        _ => panic!()
    }
}

fn to_hip_device_p2p_attr(device_p2p_attr: cudaDeviceP2PAttr) -> hipDeviceP2PAttr {
    match device_p2p_attr {
        _ => panic!()
    }
}

fn to_hip_func_attribute(func_attribute: cudaFuncAttribute) -> hipFuncAttribute {
    match func_attribute {
        _ => panic!()
    }
}

fn to_cuda_mem_allocation_type(mem_allocation_type: hipMemAllocationType) -> cudaMemAllocationType {
    match mem_allocation_type {
        _ => panic!()
    }
}

fn to_hip_mem_allocation_type(mem_allocation_type: cudaMemAllocationType) -> hipMemAllocationType {
    match mem_allocation_type {
        _ => panic!()
    }
}

fn to_cuda_mem_allocation_handle_type(mem_allocation_handle_type: hipMemAllocationHandleType) -> cudaMemAllocationHandleType {
    match mem_allocation_handle_type {
        _ => panic!()
    }
}

fn to_hip_mem_allocation_handle_type(mem_allocation_handle_type: cudaMemAllocationHandleType) -> hipMemAllocationHandleType {
    match mem_allocation_handle_type {
        _ => panic!()
    }
}

fn to_cuda_mem_location_type(mem_location_type: hipMemLocationType) -> cudaMemLocationType {
    match mem_location_type {
        _ => panic!()
    }
}

fn to_hip_mem_location_type(mem_location_type: cudaMemLocationType) -> hipMemLocationType {
    match mem_location_type {
        _ => panic!()
    }
}

fn to_cuda_mem_access_flags(mem_access_flags: hipMemAccessFlags) -> cudaMemAccessFlags {
    match mem_access_flags {
        _ => panic!()
    }
}

fn to_hip_mem_access_flags(mem_access_flags: cudaMemAccessFlags) -> hipMemAccessFlags {
    match mem_access_flags {
        _ => panic!()
    }
}

fn to_cuda_memcpy_kind(memcpy_kind: hipMemcpyKind) -> cudaMemcpyKind {
    match memcpy_kind {
        hipMemcpyKind::hipMemcpyHostToHost => cudaMemcpyKind::cudaMemcpyHostToHost,
        hipMemcpyKind::hipMemcpyHostToDevice => cudaMemcpyKind::cudaMemcpyHostToDevice,
        hipMemcpyKind::hipMemcpyDeviceToHost => cudaMemcpyKind::cudaMemcpyDeviceToHost,
        hipMemcpyKind::hipMemcpyDeviceToDevice => cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        hipMemcpyKind::hipMemcpyDefault => cudaMemcpyKind::cudaMemcpyDefault,
        _ => panic!()
    }
}

fn to_hip_memcpy_kind(memcpy_kind: cudaMemcpyKind) -> hipMemcpyKind {
    match memcpy_kind {
        cudaMemcpyKind::cudaMemcpyHostToHost => hipMemcpyKind::hipMemcpyHostToHost,
        cudaMemcpyKind::cudaMemcpyHostToDevice => hipMemcpyKind::hipMemcpyHostToDevice,
        cudaMemcpyKind::cudaMemcpyDeviceToHost => hipMemcpyKind::hipMemcpyDeviceToHost,
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => hipMemcpyKind::hipMemcpyDeviceToDevice,
        cudaMemcpyKind::cudaMemcpyDefault => hipMemcpyKind::hipMemcpyDefault,
        _ => panic!()
    }
}

fn to_cuda_graph_exec_update_result(graph_exec_update_result: hipGraphExecUpdateResult) -> cudaGraphExecUpdateResult {
    match graph_exec_update_result {
        _ => panic!()
    }
}

fn to_hip_graph_exec_update_result(graph_exec_update_result: cudaGraphExecUpdateResult) -> hipGraphExecUpdateResult {
    match graph_exec_update_result {
        _ => panic!()
    }
}

fn to_cuda_graph_node_type(graph_node_type: hipGraphNodeType) -> cudaGraphNodeType {
    match graph_node_type {
        _ => panic!()
    }
}

fn to_hip_kernel_node_attr_id(kernel_node_attr_id: cudaKernelNodeAttrID) -> hipKernelNodeAttrID {
    match kernel_node_attr_id {
        _ => panic!()
    }
}

fn to_hip_external_memory_handle_type(external_memory_handle_type: cudaExternalMemoryHandleType) -> hipExternalMemoryHandleType {
    match external_memory_handle_type {
        _ => panic!()
    }
}

fn to_hip_external_semaphore_handle_type(external_semaphore_handle_type: cudaExternalSemaphoreHandleType) -> hipExternalSemaphoreHandleType {
    match external_semaphore_handle_type {
        _ => panic!()
    }
}

fn to_hip_memory_advise(memory_advise: cudaMemoryAdvise) -> hipMemoryAdvise {
    match memory_advise {
        _ => panic!()
    }
}

fn to_cuda_mem_range_attribute(mem_range_attribute: hipMemRangeAttribute) -> cudaMemRangeAttribute {
    match mem_range_attribute {
        _ => panic!()
    }
}

fn to_hip_mem_range_attribute(mem_range_attribute: cudaMemRangeAttribute) -> hipMemRangeAttribute {
    match mem_range_attribute {
        _ => panic!()
    }
}

fn to_hip_mem_pool_attr(mem_pool_attr: cudaMemPoolAttr) -> hipMemPoolAttr {
    match mem_pool_attr {
        _ => panic!()
    }
}

fn to_cuda_memory_type(memory_type: hipMemoryType) -> cudaMemoryType {
    match memory_type {
        _ => panic!()
    }
}

fn to_hip_memory_type(memory_type: cudaMemoryType) -> hipMemoryType {
    match memory_type {
        _ => panic!()
    }
}

fn to_cuda_stream_capture_mode(stream_capture_mode: hipStreamCaptureMode) -> cudaStreamCaptureMode {
    match stream_capture_mode {
        _ => panic!()
    }
}

fn to_hip_stream_capture_mode(stream_capture_mode: cudaStreamCaptureMode) -> hipStreamCaptureMode {
    match stream_capture_mode {
        _ => panic!()
    }
}

fn to_cuda_stream_capture_status(stream_capture_status: hipStreamCaptureStatus) -> cudaStreamCaptureStatus {
    match stream_capture_status {
        hipStreamCaptureStatus::hipStreamCaptureStatusNone => cudaStreamCaptureStatus::cudaStreamCaptureStatusNone,
        hipStreamCaptureStatus::hipStreamCaptureStatusActive => cudaStreamCaptureStatus::cudaStreamCaptureStatusActive,
        hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated => cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated,
        _ => panic!()
    }
}

fn to_hip_device_attr(device_attr: cudaDeviceAttr) -> hipDeviceAttribute_t {
    match device_attr {
        _ => panic!()
    }
}

fn to_cuda_device_attr(device_attr: hipDeviceAttribute_t) -> cudaDeviceAttr {
    match device_attr {
        _ => panic!()
    }
}

fn to_cuda_dim3(dim3: hip_runtime_api::dim3) -> cudart::dim3 {
    cudart::dim3 {
        x: dim3.x,
        y: dim3.y,
        z: dim3.z,
    }
}

fn to_hip_dim3(dim3: cudart::dim3) -> hip_runtime_api::dim3 {
    hip_runtime_api::dim3 {
        x: dim3.x,
        y: dim3.y,
        z: dim3.z,
    }
}

fn to_cuda_pos(pos: hip_runtime_api::hipPos) -> cudart::cudaPos {
    cudart::cudaPos {
        x: pos.x,
        y: pos.y,
        z: pos.z,
    }
}

fn to_hip_pos(pos: cudart::cudaPos) -> hip_runtime_api::hipPos {
    hip_runtime_api::hipPos {
        x: pos.x,
        y: pos.y,
        z: pos.z,
    }
}

unsafe fn pop_call_configuration(
    grid_dim: *mut cudart::dim3,
    block_dim: *mut cudart::dim3,
    shared_mem: *mut usize,
    stream: *mut ::std::os::raw::c_void,
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
    stream: *mut ::std::os::raw::c_void,
) -> u32 {
    let grid_dim = to_hip_dim3(grid_dim);
    let block_dim = to_hip_dim3(block_dim);
    to_cuda(__hipPushCallConfiguration(
        grid_dim,
        block_dim,
        shared_mem,
        stream.cast(),
    )).0 as _
}

unsafe fn register_fat_binary(
    fat_cubin: *mut ::std::os::raw::c_void,
) -> *mut *mut ::std::os::raw::c_void {
    __hipRegisterFatBinary(fat_cubin)
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
        ptr::null_mut(),
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
    _ext: i32,
    size: usize,
    constant: i32,
    _global: i32,
) -> ::std::os::raw::c_void {
    __hipRegisterManagedVar(
        *fat_cubin_handle,
        host_var_ptr_address,
        device_address.cast(),
        device_name,
        size,
        constant as _,
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
        (*device_address).cast(),
        host_var as _,
        device_name as  _,
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
        (*device_address).cast(),
        host_var as _,
        device_name as _,
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
        device_address.cast(),
        host_var,
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
    let mut out_cache_config = to_hip_func_cache(*p_cache_config);
    let status = to_cuda(hipDeviceGetCacheConfig(
        &mut out_cache_config,
    ));
    *p_cache_config = to_cuda_func_cache(out_cache_config);
    status
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
    let status = to_cuda(hipDeviceGetSharedMemConfig(
        &mut out_config,
    ));
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

unsafe fn ipc_open_event_handle(
    event: *mut cudaEvent_t,
    handle: cudaIpcEventHandle_t,
) -> cudaError_t {
    let handle = hipIpcEventHandle_t {
        reserved: handle.reserved,
    };
    to_cuda(hipIpcOpenEventHandle(
        event.cast(),
        handle,
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

unsafe fn ipc_open_mem_handle(
    dev_ptr: *mut *mut ::std::os::raw::c_void,
    handle: cudaIpcMemHandle_t,
    flags: u32,
) -> cudaError_t {
    let handle = hipIpcMemHandle_t {
        reserved: handle.reserved,
    };
    to_cuda(hipIpcOpenMemHandle(
        dev_ptr,
        handle,
        flags,
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
    to_cuda(hipGetDeviceProperties(
        prop.cast(),
        device,
    ))
}

unsafe fn device_get_attribute(
    value: *mut i32,
    attr: cudaDeviceAttr,
    device: i32,
) -> cudaError_t {
    let attr = to_hip_device_attr(attr);
    to_cuda(hipDeviceGetAttribute(
        value,
        attr,
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

unsafe fn device_get_p2p_attribute(
    value: *mut i32,
    attr: cudaDeviceP2PAttr,
    src_device: i32,
    dst_device: i32,
) -> cudaError_t {
    let attr = to_hip_device_p2p_attr(attr);
    to_cuda(hipDeviceGetP2PAttribute(
        value,
        attr,
        src_device,
        dst_device,
    ))
}

unsafe fn choose_device(
    device: *mut i32,
    prop: *const cudaDeviceProp,
) -> cudaError_t {
    let prop = *prop;
    let prop = hipDeviceProp_t {
        name: prop.name,
        totalGlobalMem: prop.totalGlobalMem,
        sharedMemPerBlock: prop.sharedMemPerBlock,
        regsPerBlock: prop.regsPerBlock,
        warpSize: prop.warpSize,
        maxThreadsPerBlock: prop.maxThreadsPerBlock,
        maxThreadsDim: prop.maxThreadsDim,
        maxGridSize: prop.maxGridSize,
        clockRate: prop.clockRate,
        memoryClockRate: prop.memoryClockRate,
        memoryBusWidth: prop.memoryBusWidth,
        totalConstMem: prop.totalConstMem,
        major: prop.major,
        minor: prop.minor,
        multiProcessorCount: prop.multiProcessorCount,
        l2CacheSize: prop.l2CacheSize,
        maxThreadsPerMultiProcessor: prop.maxThreadsPerMultiProcessor,
        computeMode: prop.computeMode,
        clockInstructionRate: 0,
        arch: hipDeviceArch_t {
            _bitfield_align_1: [0; 0],
            _bitfield_1: __BindgenBitfieldUnit::new([0; 3]),
            __bindgen_padding_0: 0,
        },
        concurrentKernels: prop.concurrentKernels,
        pciDomainID: prop.pciDomainID,
        pciBusID: prop.pciBusID,
        pciDeviceID: prop.pciDeviceID,
        maxSharedMemoryPerMultiProcessor: 0,
        isMultiGpuBoard: prop.isMultiGpuBoard,
        canMapHostMemory: prop.canMapHostMemory,
        gcnArch: 0,
        gcnArchName: [0; 256],
        integrated: prop.integrated,
        cooperativeLaunch: prop.cooperativeLaunch,
        cooperativeMultiDeviceLaunch: prop.cooperativeMultiDeviceLaunch,
        maxTexture1DLinear: prop.maxTexture1DLinear,
        maxTexture1D: prop.maxTexture1D,
        maxTexture2D: prop.maxTexture2D,
        maxTexture3D: prop.maxTexture3D,
        hdpMemFlushCntl: ptr::null_mut(),
        hdpRegFlushCntl: ptr::null_mut(),
        memPitch: prop.memPitch,
        textureAlignment: prop.textureAlignment,
        texturePitchAlignment: prop.texturePitchAlignment,
        kernelExecTimeoutEnabled: prop.kernelExecTimeoutEnabled,
        ECCEnabled: prop.ECCEnabled,
        tccDriver: prop.tccDriver,
        cooperativeMultiDeviceUnmatchedFunc: 0,
        cooperativeMultiDeviceUnmatchedGridDim: 0,
        cooperativeMultiDeviceUnmatchedBlockDim: 0,
        cooperativeMultiDeviceUnmatchedSharedMem: 0,
        isLargeBar: 0,
        asicRevision: 0,
        managedMemory: prop.managedMemory,
        directManagedMemAccessFromHost: prop.directManagedMemAccessFromHost,
        concurrentManagedAccess: prop.concurrentManagedAccess,
        pageableMemoryAccess: prop.pageableMemoryAccess,
        pageableMemoryAccessUsesHostPageTables: prop.pageableMemoryAccessUsesHostPageTables,
    };
    to_cuda(hipChooseDevice(
        device,
        &prop,
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

unsafe fn stream_create_with_priority(
    p_stream: *mut cudaStream_t,
    flags: u32,
    priority: i32,
) -> cudaError_t {
    to_cuda(hipStreamCreateWithPriority(
        p_stream.cast(),
        flags,
        priority,
    ))
}

unsafe fn stream_get_priority(
    h_stream: cudaStream_t,
    priority: *mut i32,
) -> cudaError_t {
    to_cuda(hipStreamGetPriority(
        h_stream.cast(),
        priority,
    ))
}

unsafe fn stream_get_priority_ptsz(
    h_stream: cudaStream_t,
    priority: *mut i32,
) -> cudaError_t {
    to_cuda(hipStreamGetPriority_spt(
        h_stream.cast(),
        priority,
    ))
}

unsafe fn stream_get_flags(
    h_stream: cudaStream_t,
    flags: *mut u32,
) -> cudaError_t {
    to_cuda(hipStreamGetFlags(
        h_stream.cast(),
        flags,
    ))
}

unsafe fn stream_get_flags_ptsz(
    h_stream: cudaStream_t,
    flags: *mut u32,
) -> cudaError_t {
    to_cuda(hipStreamGetFlags_spt(
        h_stream.cast(),
        flags,
    ))
}

unsafe fn stream_destroy(
    stream: cudaStream_t,
) -> cudaError_t {
    to_cuda(hipStreamDestroy(
        stream.cast(),
    ))
}

unsafe fn stream_wait_event(
    stream: cudaStream_t,
    event: cudaEvent_t,
    flags: u32,
) -> cudaError_t {
    to_cuda(hipStreamWaitEvent(
        stream.cast(),
        event.cast(),
        flags,
    ))
}

unsafe fn stream_wait_event_ptsz(
    stream: cudaStream_t,
    event: cudaEvent_t,
    flags: u32,
) -> cudaError_t {
    to_cuda(hipStreamWaitEvent_spt(
        stream.cast(),
        event.cast(),
        flags,
    ))
}

unsafe fn stream_add_callback(
    _stream: cudaStream_t,
    _callback: cudaStreamCallback_t,
    _user_data: *mut ::std::os::raw::c_void,
    _flags: u32,
) -> cudaError_t {
    todo!()
}

unsafe fn stream_add_callback_ptsz(
    _stream: cudaStream_t,
    _callback: cudaStreamCallback_t,
    _user_data: *mut ::std::os::raw::c_void,
    _flags: u32,
) -> cudaError_t {
    todo!()
}

unsafe fn stream_synchronize(
    stream: cudaStream_t,
) -> cudaError_t {
    to_cuda(hipStreamSynchronize(
        stream.cast(),
    ))
}

unsafe fn stream_synchronize_ptsz(
    stream: cudaStream_t,
) -> cudaError_t {
    to_cuda(hipStreamSynchronize_spt(
        stream.cast(),
    ))
}

unsafe fn stream_query(
    stream: cudaStream_t,
) -> cudaError_t {
    to_cuda(hipStreamQuery(
        stream.cast(),
    ))
}

unsafe fn stream_query_ptsz(
    stream: cudaStream_t,
) -> cudaError_t {
    to_cuda(hipStreamQuery_spt(
        stream.cast(),
    ))
}

unsafe fn stream_attach_mem_async(
    stream: cudaStream_t,
    dev_ptr: *mut ::std::os::raw::c_void,
    length: usize,
    flags: u32,
) -> cudaError_t {
    to_cuda(hipStreamAttachMemAsync(
        stream.cast(),
        dev_ptr,
        length,
        flags,
    ))
}

unsafe fn stream_begin_capture(
    stream: cudaStream_t,
    mode: cudaStreamCaptureMode,
) -> cudaError_t {
    let mode = to_hip_stream_capture_mode(mode);
    to_cuda(hipStreamBeginCapture(
        stream.cast(),
        mode,
    ))
}

unsafe fn stream_begin_capture_ptsz(
    stream: cudaStream_t,
    mode: cudaStreamCaptureMode,
) -> cudaError_t {
    let mode = to_hip_stream_capture_mode(mode);
    to_cuda(hipStreamBeginCapture_spt(
        stream.cast(),
        mode,
    ))
}

unsafe fn thread_exchange_stream_capture_mode(
    mode: *mut cudaStreamCaptureMode,
) -> cudaError_t {
    let ptr = mode;
    let mut mode = to_hip_stream_capture_mode(*mode);
    let status = to_cuda(hipThreadExchangeStreamCaptureMode(
        &mut mode,
    ));
    *ptr = to_cuda_stream_capture_mode(mode);
    status
}

unsafe fn stream_end_capture(
    stream: cudaStream_t,
    p_graph: *mut cudaGraph_t,
) -> cudaError_t {
    to_cuda(hipStreamEndCapture(
        stream.cast(),
        p_graph.cast(),
    ))
}

unsafe fn stream_end_capture_ptsz(
    stream: cudaStream_t,
    p_graph: *mut cudaGraph_t,
) -> cudaError_t {
    to_cuda(hipStreamEndCapture_spt(
        stream.cast(),
        p_graph.cast(),
    ))
}

unsafe fn stream_is_capturing(
    stream: cudaStream_t,
    p_capture_status: *mut cudaStreamCaptureStatus,
) -> cudaError_t {
    let mut capture_status = hipStreamCaptureStatus(0);
    let status = to_cuda(hipStreamIsCapturing(
        stream.cast(),
        &mut capture_status,
    ));
    *p_capture_status = to_cuda_stream_capture_status(capture_status);
    status
}

unsafe fn stream_is_capturing_ptsz(
    stream: cudaStream_t,
    p_capture_status: *mut cudaStreamCaptureStatus,
) -> cudaError_t {
    let mut capture_status = hipStreamCaptureStatus(0);
    let status = to_cuda(hipStreamIsCapturing_spt(
        stream.cast(),
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
    let mut capture_status = hipStreamCaptureStatus(0);
    let status = to_cuda(hipStreamGetCaptureInfo(
        stream.cast(),
        &mut capture_status,
        p_id,
    ));
    *p_capture_status = to_cuda_stream_capture_status(capture_status);
    status
}

unsafe fn stream_get_capture_info_ptsz(
    stream: cudaStream_t,
    p_capture_status: *mut cudaStreamCaptureStatus,
    p_id: *mut u64,
) -> cudaError_t {
    let mut capture_status = hipStreamCaptureStatus(0);
    let status = to_cuda(hipStreamGetCaptureInfo_spt(
        stream.cast(),
        &mut capture_status,
        p_id,
    ));
    *p_capture_status = to_cuda_stream_capture_status(capture_status);
    status
}

unsafe fn stream_get_capture_info_v2(
    stream: cudaStream_t,
    capture_status_out: *mut cudaStreamCaptureStatus,
    id_out: *mut u64,
    graph_out: *mut cudaGraph_t,
    dependencies_out: *mut *const cudaGraphNode_t,
    num_dependencies_out: *mut usize,
) -> cudaError_t {
    let mut capture_status = hipStreamCaptureStatus(0);
    let status = to_cuda(hipStreamGetCaptureInfo_v2(
        stream.cast(),
        &mut capture_status,
        id_out,
        graph_out.cast(),
        dependencies_out.cast(),
        num_dependencies_out,
    ));
    *capture_status_out = to_cuda_stream_capture_status(capture_status);
    status
}

unsafe fn stream_get_capture_info_v2_ptsz(
    stream: cudaStream_t,
    capture_status_out: *mut cudaStreamCaptureStatus,
    id_out: *mut u64,
    graph_out: *mut cudaGraph_t,
    dependencies_out: *mut *const cudaGraphNode_t,
    num_dependencies_out: *mut usize,
) -> cudaError_t {
    let mut capture_status = hipStreamCaptureStatus(0);
    let status = to_cuda(hipStreamGetCaptureInfo_v2_spt(
        stream.cast(),
        &mut capture_status,
        id_out,
        graph_out.cast(),
        dependencies_out.cast(),
        num_dependencies_out,
    ));
    *capture_status_out = to_cuda_stream_capture_status(capture_status);
    status
}

unsafe fn stream_update_capture_dependencies(
    stream: cudaStream_t,
    dependencies: *mut cudaGraphNode_t,
    num_dependencies: usize,
    flags: u32,
) -> cudaError_t {
    to_cuda(hipStreamUpdateCaptureDependencies(
        stream.cast(),
        dependencies.cast(),
        num_dependencies,
        flags,
    ))
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
    to_cuda(hipEventRecord(
        event.cast(),
        stream.cast(),
    ))
}

unsafe fn event_record_ptsz(
    event: cudaEvent_t,
    stream: cudaStream_t,
) -> cudaError_t {
    to_cuda(hipEventRecord_spt(
        event.cast(),
        stream.cast(),
    ))
}

unsafe fn event_record_with_flags(
    event: cudaEvent_t,
    stream: cudaStream_t,
    _flags: u32,
) -> cudaError_t {
    event_record(event, stream)
}

unsafe fn event_record_with_flags_ptsz(
    event: cudaEvent_t,
    stream: cudaStream_t,
    _flags: u32,
) -> cudaError_t {
    event_record_ptsz(event, stream)
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

unsafe fn import_external_memory(
    ext_mem_out: *mut cudaExternalMemory_t,
    mem_handle_desc: *const cudaExternalMemoryHandleDesc,
) -> cudaError_t {
    let mem_handle_desc = *mem_handle_desc;
    let mem_handle_desc = hipExternalMemoryHandleDesc {
        type_: to_hip_external_memory_handle_type(mem_handle_desc.type_),
        handle: hipExternalMemoryHandleDesc_st__bindgen_ty_1 {
            win32: hipExternalMemoryHandleDesc_st__bindgen_ty_1__bindgen_ty_1 {
                handle: mem_handle_desc.handle.win32.handle,
                name: mem_handle_desc.handle.win32.name,
            },
        },
        size: mem_handle_desc.size,
        flags: mem_handle_desc.flags,
    };
    to_cuda(hipImportExternalMemory(
        ext_mem_out.cast(),
        &mem_handle_desc,
    ))
}

unsafe fn external_memory_get_mapped_buffer(
    dev_ptr: *mut *mut ::std::os::raw::c_void,
    ext_mem: cudaExternalMemory_t,
    buffer_desc: *const cudaExternalMemoryBufferDesc,
) -> cudaError_t {
    let buffer = *buffer_desc;
    let buffer = hipExternalMemoryBufferDesc {
        offset: buffer.offset,
        size: buffer.size,
        flags: buffer.flags,
    };
    to_cuda(hipExternalMemoryGetMappedBuffer(
        dev_ptr,
        ext_mem.cast(),
        &buffer,
    ))
}

unsafe fn external_memory_get_mapped_mipmapped_array(
    mipmap: *mut cudaMipmappedArray_t,
    ext_mem: cudaExternalMemory_t,
    mipmap_desc: *const cudaExternalMemoryMipmappedArrayDesc,
) -> cudaError_t {
    let mm = *mipmap_desc;
    let mm = hipExternalMemoryBufferDesc {
        offset: mm.offset,
        size: 0,
        flags: mm.flags,
    };
    to_cuda(hipExternalMemoryGetMappedBuffer(
        mipmap.cast(),
        ext_mem.cast(),
        &mm,
    ))
}

unsafe fn destroy_external_memory(
    ext_mem: cudaExternalMemory_t,
) -> cudaError_t {
    to_cuda(hipDestroyExternalMemory(
        ext_mem.cast(),
    ))
}

unsafe fn import_external_semaphore(
    ext_sem_out: *mut cudaExternalSemaphore_t,
    sem_handle_desc: *const cudaExternalSemaphoreHandleDesc,
) -> cudaError_t {
    let sem_handle_desc = *sem_handle_desc;
    let sem_handle_desc = hipExternalSemaphoreHandleDesc {
        type_: to_hip_external_semaphore_handle_type(sem_handle_desc.type_),
        handle: hipExternalSemaphoreHandleDesc_st__bindgen_ty_1 {
            win32: hipExternalSemaphoreHandleDesc_st__bindgen_ty_1__bindgen_ty_1 {
                handle: sem_handle_desc.handle.win32.handle,
                name: sem_handle_desc.handle.win32.name,
            },
        },
        flags: sem_handle_desc.flags,
    };
    to_cuda(hipImportExternalSemaphore(
        ext_sem_out.cast(),
        &sem_handle_desc,
    ))
}

unsafe fn signal_external_semaphores_async(
    ext_sem_array: *const cudaExternalSemaphore_t,
    params_array: *const cudaExternalSemaphoreSignalParams,
    num_ext_sems: u32,
    stream: cudaStream_t,
) -> cudaError_t {
    to_cuda(hipSignalExternalSemaphoresAsync(
        ext_sem_array.cast(),
        params_array.cast(), // TODO
        num_ext_sems,
        stream.cast(),
    ))
}

unsafe fn wait_external_semaphores_async(
    ext_sem_array: *const cudaExternalSemaphore_t,
    params_array: *const cudaExternalSemaphoreWaitParams,
    num_ext_sems: u32,
    stream: cudaStream_t,
) -> cudaError_t {
    to_cuda(hipWaitExternalSemaphoresAsync(
        ext_sem_array.cast(),
        params_array.cast(), // TODO
        num_ext_sems,
        stream.cast(),
    ))
}

unsafe fn destroy_external_semaphore(
    ext_sem: cudaExternalSemaphore_t,
) -> cudaError_t {
    to_cuda(hipDestroyExternalSemaphore(
        ext_sem.cast(),
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
    to_cuda(hipLaunchKernel(
        func,
        grid_dim,
        block_dim,
        args,
        shared_mem,
        stream.cast(),
    ))
}

unsafe fn launch_kernel_ptsz(
    func: *const ::std::os::raw::c_void,
    grid_dim: cudart::dim3,
    block_dim: cudart::dim3,
    args: *mut *mut ::std::os::raw::c_void,
    shared_mem: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    let grid_dim = to_hip_dim3(grid_dim);
    let block_dim = to_hip_dim3(block_dim);
    to_cuda(hipLaunchKernel_spt(
        func,
        grid_dim,
        block_dim,
        args,
        shared_mem,
        stream.cast(),
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
    to_cuda(hipLaunchCooperativeKernel(
        func,
        grid_dim,
        block_dim,
        args,
        shared_mem as _,
        stream.cast(),
    ))
}

unsafe fn launch_cooperative_kernel_ptsz(
    func: *const ::std::os::raw::c_void,
    grid_dim: cudart::dim3,
    block_dim: cudart::dim3,
    args: *mut *mut ::std::os::raw::c_void,
    shared_mem: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    let grid_dim = to_hip_dim3(grid_dim);
    let block_dim = to_hip_dim3(block_dim);
    to_cuda(hipLaunchCooperativeKernel_spt(
        func,
        grid_dim,
        block_dim,
        args,
        shared_mem as _,
        stream.cast(),
    ))
}

unsafe fn launch_cooperative_kernel_multi_device(
    launch_params_list: *mut cudaLaunchParams,
    num_devices: u32,
    flags: u32,
) -> cudaError_t {
    let ptr = launch_params_list;
    let mut launch_params_list = vec![];
    for i in 0..num_devices {
        let launch_param = *ptr.add(i as _);
        launch_params_list.push(hipLaunchParams {
            func: launch_param.func,
            gridDim: to_hip_dim3(launch_param.gridDim),
            blockDim: to_hip_dim3(launch_param.blockDim),
            args: launch_param.args,
            sharedMem: launch_param.sharedMem,
            stream: launch_param.stream.cast(),
        });
    }
    let launch_params_list = launch_params_list.as_mut_ptr();
    let status = to_cuda(hipLaunchCooperativeKernelMultiDevice(
        launch_params_list,
        num_devices as _,
        flags,
    ));
    for i in 0..num_devices {
        let launch_param = *launch_params_list.add(i as _);
        (*ptr.add(i as _)) = cudaLaunchParams {
            func: launch_param.func,
            gridDim: to_cuda_dim3(launch_param.gridDim),
            blockDim: to_cuda_dim3(launch_param.blockDim),
            args: launch_param.args,
            sharedMem: launch_param.sharedMem,
            stream: launch_param.stream.cast(),
        };
    }
    status
}

unsafe fn func_set_cache_config(
    func: *const ::std::os::raw::c_void,
    cache_config: cudaFuncCache,
) -> cudaError_t {
    let cache_config = to_hip_func_cache(cache_config);
    to_cuda(hipFuncSetCacheConfig(
        func,
        cache_config,
    ))
}

unsafe fn func_set_shared_mem_config(
    func: *const ::std::os::raw::c_void,
    config: cudaSharedMemConfig,
) -> cudaError_t {
    let config = to_hip_shared_mem_config(config);
    to_cuda(hipFuncSetSharedMemConfig(
        func,
        config,
    ))
}

unsafe fn func_get_attributes(
    attr: *mut cudaFuncAttributes,
    func: *const ::std::os::raw::c_void,
) -> cudaError_t {
    let ptr = attr;
    let attr = *attr;
    let mut attr = hipFuncAttributes {
        binaryVersion: attr.binaryVersion,
        cacheModeCA: attr.cacheModeCA,
        constSizeBytes: attr.constSizeBytes,
        localSizeBytes: attr.localSizeBytes,
        maxDynamicSharedSizeBytes: attr.maxDynamicSharedSizeBytes,
        maxThreadsPerBlock: attr.maxThreadsPerBlock,
        numRegs: attr.numRegs,
        preferredShmemCarveout: attr.preferredShmemCarveout,
        ptxVersion: attr.ptxVersion,
        sharedSizeBytes: attr.sharedSizeBytes,
    };
    let status = to_cuda(hipFuncGetAttributes(
        &mut attr,
        func,
    ));
    *ptr = cudaFuncAttributes {
        sharedSizeBytes: attr.sharedSizeBytes,
        constSizeBytes: attr.constSizeBytes,
        localSizeBytes: attr.localSizeBytes,
        maxThreadsPerBlock: attr.maxThreadsPerBlock,
        numRegs: attr.numRegs,
        ptxVersion: attr.ptxVersion,
        binaryVersion: attr.binaryVersion,
        cacheModeCA: attr.cacheModeCA,
        maxDynamicSharedSizeBytes: attr.maxDynamicSharedSizeBytes,
        preferredShmemCarveout: attr.preferredShmemCarveout,
    };
    status
}

unsafe fn func_set_attribute(
    func: *const ::std::os::raw::c_void,
    attr: cudaFuncAttribute,
    value: i32,
) -> cudaError_t {
    let attr = to_hip_func_attribute(attr);
    to_cuda(hipFuncSetAttribute(
        func,
        attr,
        value,
    ))
}

unsafe fn launch_host_func(
    stream: cudaStream_t,
    fn_: cudaHostFn_t,
    user_data: *mut ::std::os::raw::c_void,
) -> cudaError_t {
    to_cuda(hipLaunchHostFunc(
        stream.cast(),
        fn_,
        user_data,
    ))
}

unsafe fn launch_host_func_ptsz(
    stream: cudaStream_t,
    fn_: cudaHostFn_t,
    user_data: *mut ::std::os::raw::c_void,
) -> cudaError_t {
    to_cuda(hipLaunchHostFunc_spt(
        stream.cast(),
        fn_,
        user_data,
    ))
}

unsafe fn occupancy_max_active_blocks_per_multiprocessor(
    num_blocks: *mut i32,
    func: *const ::std::os::raw::c_void,
    block_size: i32,
    dynamic_s_mem_size: usize,
) -> cudaError_t {
    to_cuda(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        num_blocks,
        func,
        block_size,
        dynamic_s_mem_size,
    ))
}

unsafe fn occupancy_max_active_blocks_per_multiprocessor_with_flags(
    num_blocks: *mut i32,
    func: *const ::std::os::raw::c_void,
    block_size: i32,
    dynamic_s_mem_size: usize,
    flags: u32,
) -> cudaError_t {
    to_cuda(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        num_blocks,
        func,
        block_size,
        dynamic_s_mem_size,
        flags,
    ))
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

unsafe fn malloc_array(
    array: *mut cudaArray_t,
    desc: *const cudaChannelFormatDesc,
    width: usize,
    height: usize,
    flags: u32,
) -> cudaError_t {
    let desc = to_hip_channel_format_desc(*desc);
    to_cuda(hipMallocArray(
        array.cast(),
        &desc,
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

unsafe fn malloc_3d(
    pitched_dev_ptr: *mut cudaPitchedPtr,
    extent: cudaExtent,
) -> cudaError_t {
    let mut pitched_dev = to_hip_pitched_ptr(*pitched_dev_ptr);
    let extent = to_hip_extent(extent);
    let status = to_cuda(hipMalloc3D(
        &mut pitched_dev,
        extent,
    ));
    *pitched_dev_ptr = to_cuda_pitched_ptr(pitched_dev);
    status
}

unsafe fn malloc_3d_array(
    array: *mut cudaArray_t,
    desc: *const cudaChannelFormatDesc,
    extent: cudaExtent,
    flags: u32,
) -> cudaError_t {
    let desc = to_hip_channel_format_desc(*desc);
    let extent = to_hip_extent(extent);
    to_cuda(hipMalloc3DArray(
        array.cast(),
        &desc,
        extent,
        flags,
    ))
}

unsafe fn malloc_mipmapped_array(
    mipmapped_array: *mut cudaMipmappedArray_t,
    desc: *const cudaChannelFormatDesc,
    extent: cudaExtent,
    num_levels: u32,
    flags: u32,
) -> cudaError_t {
    let desc = to_hip_channel_format_desc(*desc);
    let extent = to_hip_extent(extent);
    to_cuda(hipMallocMipmappedArray(
        mipmapped_array.cast(),
        &desc,
        extent,
        num_levels,
        flags,
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

unsafe fn memcpy_3d(
    p: *const cudaMemcpy3DParms,
) -> cudaError_t {
    let p = to_hip_memcpy_3d_params(*p);
    to_cuda(hipMemcpy3D(
        &p,
    ))
}

unsafe fn memcpy_3d_ptds(
    p: *const cudaMemcpy3DParms,
) -> cudaError_t {
    let p = to_hip_memcpy_3d_params(*p);
    to_cuda(hipMemcpy3D_spt(
        &p,
    ))
}

unsafe fn memcpy_3d_async(
    p: *const cudaMemcpy3DParms,
    stream: cudaStream_t,
) -> cudaError_t {
    let p = to_hip_memcpy_3d_params(*p);
    to_cuda(hipMemcpy3DAsync(
        &p,
        stream.cast(),
    ))
}

unsafe fn memcpy_3d_async_ptsz(
    p: *const cudaMemcpy3DParms,
    stream: cudaStream_t,
) -> cudaError_t {
    let p = to_hip_memcpy_3d_params(*p);
    to_cuda(hipMemcpy3DAsync_spt(
        &p,
        stream.cast(),
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

fn to_cuda_extent(extent: hipExtent) -> cudaExtent {
    cudaExtent {
        width: extent.width,
        height: extent.height,
        depth: extent.depth,
    }
}

fn to_hip_extent(extent: cudaExtent) -> hipExtent {
    hipExtent {
        width: extent.width,
        height: extent.height,
        depth: extent.depth,
    }
}

unsafe fn array_get_info(
    desc: *mut cudaChannelFormatDesc,
    extent: *mut cudaExtent,
    flags: *mut u32,
    array: cudaArray_t,
) -> cudaError_t {
    let p_desc = desc;
    let p_extent = extent;
    let mut desc = to_hip_channel_format_desc(*desc);
    let mut extent = to_hip_extent(*extent);
    let status = to_cuda(hipArrayGetInfo(
        &mut desc,
        &mut extent,
        flags,
        array.cast(),
    ));
    *p_desc = to_cuda_channel_format_desc(desc);
    *p_extent = to_cuda_extent(extent);
    status
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

unsafe fn memcpy_2d_array_to_array(
    _dst: cudaArray_t,
    _w_offset_dst: usize,
    _h_offset_dst: usize,
    _src: cudaArray_const_t,
    _w_offset_src: usize,
    _h_offset_src: usize,
    _width: usize,
    _height: usize,
    _kind: cudaMemcpyKind,
) -> cudaError_t {
    todo!()
}

unsafe fn memcpy_2d_array_to_array_ptds(
    _dst: cudaArray_t,
    _w_offset_dst: usize,
    _h_offset_dst: usize,
    _src: cudaArray_const_t,
    _w_offset_src: usize,
    _h_offset_src: usize,
    _width: usize,
    _height: usize,
    _kind: cudaMemcpyKind,
) -> cudaError_t {
    todo!()
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
    to_cuda(hipMemcpyAsync(
        dst,
        src,
        count,
        kind,
        stream.cast(),
    ))
}

unsafe fn memcpy_async_ptsz(
    dst: *mut ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpyAsync_spt(
        dst,
        src,
        count,
        kind,
        stream.cast(),
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
    to_cuda(hipMemcpyPeerAsync(
        dst,
        dst_device,
        src,
        src_device,
        count,
        stream.cast(),
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
    to_cuda(hipMemcpy2DAsync(
        dst,
        dpitch,
        src,
        spitch,
        width,
        height,
        kind,
        stream.cast(),
    ))
}

unsafe fn memcpy_2d_async_ptsz(
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
    to_cuda(hipMemcpy2DAsync_spt(
        dst,
        dpitch,
        src,
        spitch,
        width,
        height,
        kind,
        stream.cast(),
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
    to_cuda(hipMemcpy2DToArrayAsync(
        dst.cast(),
        w_offset,
        h_offset,
        src,
        spitch,
        width,
        height,
        kind,
        stream.cast(),
    ))
}

unsafe fn memcpy_2d_to_array_async_ptsz(
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
    to_cuda(hipMemcpy2DToArrayAsync_spt(
        dst.cast(),
        w_offset,
        h_offset,
        src,
        spitch,
        width,
        height,
        kind,
        stream.cast(),
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
    to_cuda(hipMemcpy2DFromArrayAsync(
        dst,
        dpitch,
        src.cast(),
        w_offset,
        h_offset,
        width,
        height,
        kind,
        stream.cast(),
    ))
}

unsafe fn memcpy_2d_from_array_async_ptsz(
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
    to_cuda(hipMemcpy2DFromArrayAsync_spt(
        dst,
        dpitch,
        src.cast(),
        w_offset,
        h_offset,
        width,
        height,
        kind,
        stream.cast(),
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
    to_cuda(hipMemcpyToSymbolAsync(
        symbol,
        src,
        count,
        offset,
        kind,
        stream.cast(),
    ))
}

unsafe fn memcpy_to_symbol_async_ptsz(
    symbol: *const ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpyToSymbolAsync_spt(
        symbol,
        src,
        count,
        offset,
        kind,
        stream.cast(),
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
    to_cuda(hipMemcpyFromSymbolAsync(
        dst,
        symbol,
        count,
        offset,
        kind,
        stream.cast(),
    ))
}

unsafe fn memcpy_from_symbol_async_ptsz(
    dst: *mut ::std::os::raw::c_void,
    symbol: *const ::std::os::raw::c_void,
    count: usize,
    offset: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpyFromSymbolAsync_spt(
        dst,
        symbol,
        count,
        offset,
        kind,
        stream.cast(),
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

unsafe fn memset_3d(
    pitched_dev_ptr: cudaPitchedPtr,
    value: i32,
    extent: cudaExtent,
) -> cudaError_t {
    let pitched_dev_ptr = to_hip_pitched_ptr(pitched_dev_ptr);
    let extent = to_hip_extent(extent);
    to_cuda(hipMemset3D(
        pitched_dev_ptr,
        value,
        extent,
    ))
}

unsafe fn memset_3d_ptds(
    pitched_dev_ptr: cudaPitchedPtr,
    value: i32,
    extent: cudaExtent,
) -> cudaError_t {
    let pitched_dev_ptr = to_hip_pitched_ptr(pitched_dev_ptr);
    let extent = to_hip_extent(extent);
    to_cuda(hipMemset3D_spt(
        pitched_dev_ptr,
        value,
        extent,
    ))
}

unsafe fn memset_async(
    dev_ptr: *mut ::std::os::raw::c_void,
    value: i32,
    count: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    to_cuda(hipMemsetAsync(
        dev_ptr,
        value,
        count,
        stream.cast(),
    ))
}

unsafe fn memset_async_ptsz(
    dev_ptr: *mut ::std::os::raw::c_void,
    value: i32,
    count: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    to_cuda(hipMemsetAsync_spt(
        dev_ptr,
        value,
        count,
        stream.cast(),
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
    to_cuda(hipMemset2DAsync(
        dev_ptr,
        pitch,
        value,
        width,
        height,
        stream.cast(),
    ))
}

unsafe fn memset_2d_async_ptsz(
    dev_ptr: *mut ::std::os::raw::c_void,
    pitch: usize,
    value: i32,
    width: usize,
    height: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    to_cuda(hipMemset2DAsync_spt(
        dev_ptr,
        pitch,
        value,
        width,
        height,
        stream.cast(),
    ))
}

unsafe fn memset_3d_async(
    pitched_dev_ptr: cudaPitchedPtr,
    value: i32,
    extent: cudaExtent,
    stream: cudaStream_t,
) -> cudaError_t {
    let pitched_dev_ptr = to_hip_pitched_ptr(pitched_dev_ptr);
    let extent = to_hip_extent(extent);
    to_cuda(hipMemset3DAsync(
        pitched_dev_ptr,
        value,
        extent,
        stream.cast(),
    ))
}

unsafe fn memset_3d_async_ptsz(
    pitched_dev_ptr: cudaPitchedPtr,
    value: i32,
    extent: cudaExtent,
    stream: cudaStream_t,
) -> cudaError_t {
    let pitched_dev_ptr = to_hip_pitched_ptr(pitched_dev_ptr);
    let extent = to_hip_extent(extent);
    to_cuda(hipMemset3DAsync_spt(
        pitched_dev_ptr,
        value,
        extent,
        stream.cast(),
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
    to_cuda(hipMemPrefetchAsync(
        dev_ptr,
        count,
        dst_device,
        stream.cast(),
    ))
}

unsafe fn mem_advise(
    dev_ptr: *const ::std::os::raw::c_void,
    count: usize,
    advice: cudaMemoryAdvise,
    device: i32,
) -> cudaError_t {
    let advice = to_hip_memory_advise(advice);
    to_cuda(hipMemAdvise(
        dev_ptr,
        count,
        advice,
        device,
    ))
}

unsafe fn mem_range_get_attribute(
    data: *mut ::std::os::raw::c_void,
    data_size: usize,
    attribute: cudaMemRangeAttribute,
    dev_ptr: *const ::std::os::raw::c_void,
    count: usize,
) -> cudaError_t {
    let attribute = to_hip_mem_range_attribute(attribute);
    to_cuda(hipMemRangeGetAttribute(
        data,
        data_size,
        attribute,
        dev_ptr,
        count,
    ))
}

unsafe fn mem_range_get_attributes(
    data: *mut *mut ::std::os::raw::c_void,
    data_sizes: *mut usize,
    attributes: *mut cudaMemRangeAttribute,
    num_attributes: usize,
    dev_ptr: *const ::std::os::raw::c_void,
    count: usize,
) -> cudaError_t {
    let ptr = attributes;
    let mut attributes = vec![];
    for i in 0..num_attributes {
        attributes.push(to_hip_mem_range_attribute(*ptr.add(i)));
    }
    let attributes = attributes.as_mut_ptr();
    let status = to_cuda(hipMemRangeGetAttributes(
        data,
        data_sizes,
        attributes,
        num_attributes,
        dev_ptr,
        count,
    ));
    for i in 0..num_attributes {
        (*ptr.add(i)) = to_cuda_mem_range_attribute(*attributes.add(i));
    }
    status
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

unsafe fn memcpy_array_to_array(
    _dst: cudaArray_t,
    _w_offset_dst: usize,
    _h_offset_dst: usize,
    _src: cudaArray_const_t,
    _w_offset_src: usize,
    _h_offset_src: usize,
    _count: usize,
    _kind: cudaMemcpyKind,
) -> cudaError_t {
    todo!()
}

unsafe fn memcpy_array_to_array_ptds(
    _dst: cudaArray_t,
    _w_offset_dst: usize,
    _h_offset_dst: usize,
    _src: cudaArray_const_t,
    _w_offset_src: usize,
    _h_offset_src: usize,
    _count: usize,
    _kind: cudaMemcpyKind,
) -> cudaError_t {
    todo!()
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
    to_cuda(hipMemcpy2DToArrayAsync(
        dst.cast(),
        w_offset,
        h_offset,
        src,
        count,
        w_offset,
        h_offset,
        kind,
        stream.cast(),
    ))
}

unsafe fn memcpy_to_array_async_ptsz(
    dst: cudaArray_t,
    w_offset: usize,
    h_offset: usize,
    src: *const ::std::os::raw::c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpy2DToArrayAsync_spt(
        dst.cast(),
        w_offset,
        h_offset,
        src,
        count,
        w_offset,
        h_offset,
        kind,
        stream.cast(),
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
    to_cuda(hipMemcpy2DFromArrayAsync(
        dst,
        count,
        src.cast(),
        w_offset,
        h_offset,
        w_offset,
        h_offset,
        kind, 
        stream.cast(),
    ))
}

unsafe fn memcpy_from_array_async_ptsz(
    dst: *mut ::std::os::raw::c_void,
    src: cudaArray_const_t,
    w_offset: usize,
    h_offset: usize,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let kind = to_hip_memcpy_kind(kind);
    to_cuda(hipMemcpy2DFromArrayAsync_spt(
        dst,
        count,
        src.cast(),
        w_offset,
        h_offset,
        w_offset,
        h_offset,
        kind, 
        stream.cast(),
    ))
}

unsafe fn malloc_async(
    dev_ptr: *mut *mut ::std::os::raw::c_void,
    size: usize,
    h_stream: cudaStream_t,
) -> cudaError_t {
    to_cuda(hipMallocAsync(
        dev_ptr,
        size,
        h_stream.cast(),
    ))
}

unsafe fn free_async(
    dev_ptr: *mut ::std::os::raw::c_void,
    h_stream: cudaStream_t,
) -> cudaError_t {
    to_cuda(hipFreeAsync(
        dev_ptr,
        h_stream.cast(),
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

unsafe fn mem_pool_set_access(
    mem_pool: cudaMemPool_t,
    desc_list: *const cudaMemAccessDesc,
    count: usize,
) -> cudaError_t {
    let ptr = desc_list;
    let mut desc_list = vec![];
    for i in 0..count {
        let desc = *ptr.add(i);
        desc_list.push(hipMemAccessDesc {
            location: to_hip_mem_location(desc.location),
            flags: to_hip_mem_access_flags(desc.flags),
        });
    }
    to_cuda(hipMemPoolSetAccess(
        mem_pool.cast(),
        desc_list.as_ptr(),
        count,
    ))
}

unsafe fn mem_pool_get_access(
    flags: *mut cudaMemAccessFlags,
    mem_pool: cudaMemPool_t,
    location: *mut cudaMemLocation,
) -> cudaError_t {
    let p_flags = flags;
    let p_location = location;
    let mut flags = to_hip_mem_access_flags(*flags);
    let mut location = to_hip_mem_location(*location);
    let status = to_cuda(hipMemPoolGetAccess(
        &mut flags,
        mem_pool.cast(),
        &mut location,
    ));
    *p_flags = to_cuda_mem_access_flags(flags);
    *p_location = to_cuda_mem_location(location);
    status
}

fn to_cuda_mem_pool_props(pool_props: hipMemPoolProps) -> cudaMemPoolProps {
    cudaMemPoolProps {
        allocType: to_cuda_mem_allocation_type(pool_props.allocType),
        handleTypes: to_cuda_mem_allocation_handle_type(pool_props.handleTypes),
        location: to_cuda_mem_location(pool_props.location),
        win32SecurityAttributes: pool_props.win32SecurityAttributes,
        reserved: [0; 64],
    }
}

fn to_hip_mem_pool_props(pool_props: cudaMemPoolProps) -> hipMemPoolProps {
    hipMemPoolProps {
        allocType: to_hip_mem_allocation_type(pool_props.allocType),
        handleTypes: to_hip_mem_allocation_handle_type(pool_props.handleTypes),
        location: to_hip_mem_location(pool_props.location),
        win32SecurityAttributes: pool_props.win32SecurityAttributes,
        reserved: [0; 64],
    }
}

unsafe fn mem_pool_create(
    mem_pool: *mut cudaMemPool_t,
    pool_props: *const cudaMemPoolProps,
) -> cudaError_t {
    let pool_props = to_hip_mem_pool_props(*pool_props);
    to_cuda(hipMemPoolCreate(
        mem_pool.cast(),
        &pool_props,
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
    to_cuda(hipMallocFromPoolAsync(
        ptr,
        size,
        mem_pool.cast(),
        stream.cast(),
    ))
}

unsafe fn mem_pool_export_to_shareable_handle(
    shareable_handle: *mut ::std::os::raw::c_void,
    mem_pool: cudaMemPool_t,
    handle_type: cudaMemAllocationHandleType,
    flags: u32,
) -> cudaError_t {
    let handle_type = to_hip_mem_allocation_handle_type(handle_type);
    to_cuda(hipMemPoolExportToShareableHandle(
        shareable_handle,
        mem_pool.cast(),
        handle_type,
        flags,
    ))
}

unsafe fn mem_pool_import_from_shareable_handle(
    mem_pool: *mut cudaMemPool_t,
    shareable_handle: *mut ::std::os::raw::c_void,
    handle_type: cudaMemAllocationHandleType,
    flags: u32,
) -> cudaError_t {
    let handle_type = to_hip_mem_allocation_handle_type(handle_type);
    to_cuda(hipMemPoolImportFromShareableHandle(
        mem_pool.cast(),
        shareable_handle,
        handle_type,
        flags,
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

unsafe fn pointer_get_attributes(
    attributes: *mut cudaPointerAttributes,
    ptr: *const ::std::os::raw::c_void,
) -> cudaError_t {
    let p_attributes = attributes;
    let attributes = *attributes;
    let mut attributes = hipPointerAttribute_t {
        __bindgen_anon_1: hipPointerAttribute_t__bindgen_ty_1 {
            memoryType: to_hip_memory_type(attributes.type_),
        },
        device: attributes.device,
        devicePointer: attributes.devicePointer,
        hostPointer: attributes.hostPointer,
        isManaged: 0,
        allocationFlags: 0,
    };
    let status = to_cuda(hipPointerGetAttributes(
        &mut attributes,
        ptr,
    ));
    *p_attributes = cudaPointerAttributes {
        type_: to_cuda_memory_type(attributes.__bindgen_anon_1.memoryType),
        device: attributes.device,
        devicePointer: attributes.devicePointer,
        hostPointer: attributes.hostPointer,
    };
    status
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
    to_cuda(hipGraphicsMapResources(
        count,
        resources.cast(),
        stream.cast(),
    ))
}

unsafe fn graphics_unmap_resources(
    count: i32,
    resources: *mut cudaGraphicsResource_t,
    stream: cudaStream_t,
) -> cudaError_t {
    to_cuda(hipGraphicsUnmapResources(
        count,
        resources.cast(),
        stream.cast(),
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

fn to_cuda_channel_format_desc(desc: hipChannelFormatDesc) -> cudaChannelFormatDesc {
    cudaChannelFormatDesc {
        x: desc.x,
        y: desc.y,
        z: desc.z,
        w: desc.w,
        f: to_cuda_channel_format_kind(desc.f),
    }
}

fn to_hip_channel_format_desc(desc: cudaChannelFormatDesc) -> hipChannelFormatDesc {
    hipChannelFormatDesc {
        x: desc.x,
        y: desc.y,
        z: desc.z,
        w: desc.w,
        f: to_hip_channel_format_kind(desc.f),
    }
}

fn to_cuda_texture_reference(tex: hip_runtime_api::textureReference) -> cudart::textureReference {
    let mut address_mode: [cudaTextureAddressMode; 3] = [cudaTextureAddressMode(0); 3];
    for i in 0..3 {
        address_mode[i] = to_cuda_address_mode(tex.addressMode[i]);
    }
    cudart::textureReference {
        normalized: tex.normalized,
        filterMode: to_cuda_filter_mode(tex.filterMode),
        addressMode: address_mode,
        channelDesc: to_cuda_channel_format_desc(tex.channelDesc),
        sRGB: tex.sRGB,
        maxAnisotropy: tex.maxAnisotropy,
        mipmapFilterMode: to_cuda_filter_mode(tex.mipmapFilterMode),
        mipmapLevelBias: tex.mipmapLevelBias,
        minMipmapLevelClamp: tex.minMipmapLevelClamp,
        maxMipmapLevelClamp: tex.maxMipmapLevelClamp,
        disableTrilinearOptimization: 0,
        __cudaReserved: [0; 14],
    }
}

fn to_hip_texture_reference(tex: cudart::textureReference) -> hip_runtime_api::textureReference {
    let tex_channel_desc = to_hip_channel_format_desc(tex.channelDesc);
    let mut address_mode: [hipTextureAddressMode; 3] = [hipTextureAddressMode(0); 3];
    for i in 0..3 {
        address_mode[i] = to_hip_address_mode(tex.addressMode[i]);
    }
    hip_runtime_api::textureReference {
        normalized: tex.normalized,
        readMode: hipTextureReadMode::hipReadModeElementType,
        filterMode: to_hip_filter_mode(tex.filterMode),
        addressMode: address_mode,
        channelDesc: tex_channel_desc,
        sRGB: tex.sRGB,
        maxAnisotropy: tex.maxAnisotropy,
        mipmapFilterMode: to_hip_filter_mode(tex.mipmapFilterMode),
        mipmapLevelBias: tex.mipmapLevelBias,
        minMipmapLevelClamp: tex.minMipmapLevelClamp,
        maxMipmapLevelClamp: tex.maxMipmapLevelClamp,
        textureObject: ptr::null_mut(),
        numChannels: 1,
        format: hipArray_Format::HIP_AD_FORMAT_FLOAT,
    }
}

unsafe fn bind_texture(
    offset: *mut usize,
    texref: *const cudart::textureReference,
    dev_ptr: *const ::std::os::raw::c_void,
    desc: *const cudaChannelFormatDesc,
    size: usize,
) -> cudaError_t {
    let tex = to_hip_texture_reference(*texref);
    let desc = to_hip_channel_format_desc(*desc);
    to_cuda(hipBindTexture(
        offset,
        &tex,
        dev_ptr,
        &desc,
        size,
    ))
}

unsafe fn bind_texture_2d(
    offset: *mut usize,
    texref: *const cudart::textureReference,
    dev_ptr: *const ::std::os::raw::c_void,
    desc: *const cudaChannelFormatDesc,
    width: usize,
    height: usize,
    pitch: usize,
) -> cudaError_t {
    let tex = to_hip_texture_reference(*texref);
    let desc = to_hip_channel_format_desc(*desc);
    to_cuda(hipBindTexture2D(
        offset,
        &tex,
        dev_ptr,
        &desc,
        width,
        height,
        pitch,
    ))
}

unsafe fn bind_texture_to_array(
    texref: *const cudart::textureReference,
    array: cudaArray_const_t,
    desc: *const cudaChannelFormatDesc,
) -> cudaError_t {
    let tex = to_hip_texture_reference(*texref);
    let desc = to_hip_channel_format_desc(*desc);
    to_cuda(hipBindTextureToArray(
        &tex,
        array.cast(),
        &desc,
    ))
}

unsafe fn bind_texture_to_mipmapped_array(
    texref: *const cudart::textureReference,
    mipmapped_array: cudaMipmappedArray_const_t,
    desc: *const cudaChannelFormatDesc,
) -> cudaError_t {
    let tex = to_hip_texture_reference(*texref);
    let desc = to_hip_channel_format_desc(*desc);
    to_cuda(hipBindTextureToMipmappedArray(
        &tex,
        mipmapped_array.cast(),
        &desc,
    ))
}

unsafe fn unbind_texture(
    texref: *const cudart::textureReference,
) -> cudaError_t {
    let tex = to_hip_texture_reference(*texref);
    to_cuda(hipUnbindTexture(
        &tex,
    ))
}

unsafe fn get_texture_alignment_offset(
    offset: *mut usize,
    texref: *const cudart::textureReference,
) -> cudaError_t {
    let tex = to_hip_texture_reference(*texref);
    to_cuda(hipGetTextureAlignmentOffset(
        offset,
        &tex,
    ))
}

unsafe fn get_texture_reference(
    texref: *mut *const cudart::textureReference,
    symbol: *const ::std::os::raw::c_void,
) -> cudaError_t {
    let tex = to_hip_texture_reference(**texref);
    let mut ptr = ptr::from_ref(&tex);
    let status = to_cuda(hipGetTextureReference(
        &mut ptr,
        symbol,
    ));
    let tex = to_cuda_texture_reference(*ptr);
    let ptr = ptr::from_ref(&tex);
    *texref = ptr;
    status
}

unsafe fn get_channel_desc(
    desc: *mut cudaChannelFormatDesc,
    array: cudaArray_const_t,
) -> cudaError_t {
    let p_desc = desc;
    let mut desc = to_hip_channel_format_desc(*desc);
    let status = to_cuda(hipGetChannelDesc(
        &mut desc,
        array.cast(),
    ));
    *p_desc = to_cuda_channel_format_desc(desc);
    status
}

unsafe fn to_cuda_resource_desc(desc: hipResourceDesc) -> cudaResourceDesc {
    let res_type = to_cuda_resource_type(desc.resType);
    cudaResourceDesc {
        resType: res_type,
        res: match res_type {
            cudaResourceType::cudaResourceTypeArray => cudaResourceDesc__bindgen_ty_1 {
                array: cudaResourceDesc__bindgen_ty_1__bindgen_ty_1 {
                    array: desc.res.array.array.cast(),
                },
            },
            cudaResourceType::cudaResourceTypeMipmappedArray => cudaResourceDesc__bindgen_ty_1 {
                mipmap: cudaResourceDesc__bindgen_ty_1__bindgen_ty_2 {
                    mipmap: desc.res.mipmap.mipmap.cast(),
                },
            },
            cudaResourceType::cudaResourceTypeLinear => cudaResourceDesc__bindgen_ty_1 {
                linear: cudaResourceDesc__bindgen_ty_1__bindgen_ty_3 {
                    devPtr: desc.res.linear.devPtr,
                    desc: to_cuda_channel_format_desc(desc.res.linear.desc),
                    sizeInBytes: desc.res.linear.sizeInBytes,
                },
            },
            cudaResourceType::cudaResourceTypePitch2D => cudaResourceDesc__bindgen_ty_1 {
                pitch2D: cudaResourceDesc__bindgen_ty_1__bindgen_ty_4 {
                    devPtr: desc.res.pitch2D.devPtr,
                    desc: to_cuda_channel_format_desc(desc.res.pitch2D.desc),
                    width: desc.res.pitch2D.width,
                    height: desc.res.pitch2D.height,
                    pitchInBytes: desc.res.pitch2D.pitchInBytes,
                },
            },
            _ => panic!(),
        },
    }
}

unsafe fn to_hip_resource_desc(desc: cudaResourceDesc) -> hipResourceDesc {
    let res_type = to_hip_resource_type(desc.resType);
    hipResourceDesc {
        resType: res_type,
        res: match res_type {
            hipResourceType::hipResourceTypeArray => hipResourceDesc__bindgen_ty_1 {
                array: hipResourceDesc__bindgen_ty_1__bindgen_ty_1 {
                    array: desc.res.array.array.cast(),
                },
            },
            hipResourceType::hipResourceTypeMipmappedArray => hipResourceDesc__bindgen_ty_1 {
                mipmap: hipResourceDesc__bindgen_ty_1__bindgen_ty_2 {
                    mipmap: desc.res.mipmap.mipmap.cast(),
                },
            },
            hipResourceType::hipResourceTypeLinear => hipResourceDesc__bindgen_ty_1 {
                linear: hipResourceDesc__bindgen_ty_1__bindgen_ty_3 {
                    devPtr: desc.res.linear.devPtr,
                    desc: to_hip_channel_format_desc(desc.res.linear.desc),
                    sizeInBytes: desc.res.linear.sizeInBytes,
                },
            },
            hipResourceType::hipResourceTypePitch2D => hipResourceDesc__bindgen_ty_1 {
                pitch2D: hipResourceDesc__bindgen_ty_1__bindgen_ty_4 {
                    devPtr: desc.res.pitch2D.devPtr,
                    desc: to_hip_channel_format_desc(desc.res.pitch2D.desc),
                    width: desc.res.pitch2D.width,
                    height: desc.res.pitch2D.height,
                    pitchInBytes: desc.res.pitch2D.pitchInBytes,
                },
            },
            _ => panic!(),
        },
    }
}

fn to_hip_texture_desc(desc: cudaTextureDesc) -> hipTextureDesc {
    let mut address_mode: [hipTextureAddressMode; 3] = [hipTextureAddressMode(0); 3];
    for i in 0..3 {
        address_mode[i] = to_hip_address_mode(desc.addressMode[i]);
    }
    hipTextureDesc {
        readMode: to_hip_read_mode(desc.readMode),
        filterMode: to_hip_filter_mode(desc.filterMode),
        addressMode: address_mode,
        sRGB: desc.sRGB,
        borderColor: desc.borderColor,
        normalizedCoords: desc.normalizedCoords,
        maxAnisotropy: desc.maxAnisotropy,
        mipmapFilterMode: to_hip_filter_mode(desc.mipmapFilterMode),
        mipmapLevelBias: desc.mipmapLevelBias,
        minMipmapLevelClamp: desc.minMipmapLevelClamp,
        maxMipmapLevelClamp: desc.maxMipmapLevelClamp,
    }
}

fn to_cuda_texture_desc(desc: hipTextureDesc) -> cudaTextureDesc {
    let mut address_mode: [cudaTextureAddressMode; 3] = [cudaTextureAddressMode(0); 3];
    for i in 0..3 {
        address_mode[i] = to_cuda_address_mode(desc.addressMode[i]);
    }
    cudaTextureDesc {
        addressMode: address_mode,
        filterMode: to_cuda_filter_mode(desc.filterMode),
        readMode: to_cuda_read_mode(desc.readMode),
        sRGB: desc.sRGB,
        borderColor: desc.borderColor,
        normalizedCoords: desc.normalizedCoords,
        maxAnisotropy: desc.maxAnisotropy,
        mipmapFilterMode: to_cuda_filter_mode(desc.mipmapFilterMode),
        mipmapLevelBias: desc.mipmapLevelBias,
        minMipmapLevelClamp: desc.minMipmapLevelClamp,
        maxMipmapLevelClamp: desc.maxMipmapLevelClamp,
        disableTrilinearOptimization: 0,
    }
}

fn to_hip_resource_view_desc(desc: cudaResourceViewDesc) -> hipResourceViewDesc {
    hipResourceViewDesc {
        format: to_hip_resource_view_format(desc.format),
        width: desc.width,
        height: desc.height,
        depth: desc.depth,
        firstMipmapLevel: desc.firstMipmapLevel,
        lastMipmapLevel: desc.lastMipmapLevel,
        firstLayer: desc.firstLayer,
        lastLayer: desc.lastLayer,
    }
}

unsafe fn create_texture_object(
    p_tex_object: *mut cudaTextureObject_t,
    p_res_desc: *const cudaResourceDesc,
    p_tex_desc: *const cudaTextureDesc,
    p_res_view_desc: *const cudaResourceViewDesc,
) -> cudaError_t {
    let p_res = to_hip_resource_desc(*p_res_desc);
    let p_tex = to_hip_texture_desc(*p_tex_desc);
    let p_res_view = to_hip_resource_view_desc(*p_res_view_desc);
    to_cuda(hipCreateTextureObject(
        p_tex_object.cast(),
        &p_res,
        &p_tex,
        &p_res_view,
    ))
}

unsafe fn destroy_texture_object(
    tex_object: cudaTextureObject_t,
) -> cudaError_t {
    to_cuda(hipDestroyTextureObject(
        tex_object as _,
    ))
}

unsafe fn get_texture_object_resource_desc(
    p_res_desc: *mut cudaResourceDesc,
    tex_object: cudaTextureObject_t,
) -> cudaError_t {
    let mut res_desc = to_hip_resource_desc(*p_res_desc);
    let status = to_cuda(hipGetTextureObjectResourceDesc(
        &mut res_desc,
        tex_object as _,
    ));
    *p_res_desc = to_cuda_resource_desc(res_desc);
    status
}

unsafe fn get_texture_object_texture_desc(
    p_tex_desc: *mut cudaTextureDesc,
    tex_object: cudaTextureObject_t,
) -> cudaError_t {
    let mut tex_desc = to_hip_texture_desc(*p_tex_desc);
    let status = to_cuda(hipGetTextureObjectTextureDesc(
        &mut tex_desc,
        tex_object as _,
    ));
    *p_tex_desc = to_cuda_texture_desc(tex_desc);
    status
}

unsafe fn get_texture_object_resource_view_desc(
    p_res_view_desc: *mut cudaResourceViewDesc,
    tex_object: cudaTextureObject_t,
) -> cudaError_t {
    let mut res_view_desc = to_hip_resource_view_desc(*p_res_view_desc);
    let status = to_cuda(hipGetTextureObjectResourceViewDesc(
        &mut res_view_desc,
        tex_object as _,
    ));
    *p_res_view_desc = cudaResourceViewDesc {
        format: to_cuda_resource_view_format(res_view_desc.format),
        width: res_view_desc.width,
        height: res_view_desc.height,
        depth: res_view_desc.depth,
        firstMipmapLevel: res_view_desc.firstMipmapLevel,
        lastMipmapLevel: res_view_desc.lastMipmapLevel,
        firstLayer: res_view_desc.firstLayer,
        lastLayer: res_view_desc.lastLayer,
    };
    status
}

unsafe fn create_surface_object(
    p_surf_object: *mut cudaSurfaceObject_t,
    p_res_desc: *const cudaResourceDesc,
) -> cudaError_t {
    let p_res = to_hip_resource_desc(*p_res_desc);
    to_cuda(hipCreateSurfaceObject(
        p_surf_object.cast(),
        &p_res,
    ))
}

unsafe fn destroy_surface_object(
    surf_object: cudaSurfaceObject_t,
) -> cudaError_t {
    to_cuda(hipDestroySurfaceObject(
        surf_object as _,
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

fn to_cuda_kernel_node_params(node_params: hipKernelNodeParams) -> cudaKernelNodeParams {
    cudaKernelNodeParams {
        func: node_params.func,
        gridDim: to_cuda_dim3(node_params.gridDim),
        blockDim: to_cuda_dim3(node_params.blockDim),
        sharedMemBytes: node_params.sharedMemBytes,
        kernelParams: node_params.kernelParams,
        extra: node_params.extra,
    }
}

fn to_hip_kernel_node_params(node_params: cudaKernelNodeParams) -> hipKernelNodeParams {
    hipKernelNodeParams {
        blockDim: to_hip_dim3(node_params.blockDim),
        extra: node_params.extra,
        func: node_params.func,
        gridDim: to_hip_dim3(node_params.gridDim),
        kernelParams: node_params.kernelParams,
        sharedMemBytes: node_params.sharedMemBytes,
    }
}

unsafe fn graph_add_kernel_node(
    p_graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    p_dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    p_node_params: *const cudaKernelNodeParams,
) -> cudaError_t {
    let p_node_params = to_hip_kernel_node_params(*p_node_params);
    to_cuda(hipGraphAddKernelNode(
        p_graph_node.cast(),
        graph.cast(),
        p_dependencies.cast(),
        num_dependencies,
        &p_node_params,
    ))
}

unsafe fn graph_kernel_node_get_params(
    node: cudaGraphNode_t,
    p_node_params: *mut cudaKernelNodeParams,
) -> cudaError_t {
    let mut node_params = to_hip_kernel_node_params(*p_node_params);
    let status = to_cuda(hipGraphKernelNodeGetParams(
        node.cast(),
        &mut node_params,
    ));
    *p_node_params = to_cuda_kernel_node_params(node_params);
    status
}

unsafe fn graph_kernel_node_set_params(
    node: cudaGraphNode_t,
    p_node_params: *const cudaKernelNodeParams,
) -> cudaError_t {
    let p_node_params = to_hip_kernel_node_params(*p_node_params);
    to_cuda(hipGraphKernelNodeSetParams(
        node.cast(),
        &p_node_params,
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

unsafe fn to_cuda_kernel_node_attr_value(value: hipKernelNodeAttrValue) -> cudaKernelNodeAttrValue {
    cudaKernelNodeAttrValue {
        cooperative: value.cooperative,
    }
}

unsafe fn to_hip_kernel_node_attr_value(value: cudaKernelNodeAttrValue) -> hipKernelNodeAttrValue {
    hipKernelNodeAttrValue {
        cooperative: value.cooperative,
    }
}

unsafe fn graph_kernel_node_get_attribute(
    h_node: cudaGraphNode_t,
    attr: cudaKernelNodeAttrID,
    value_out: *mut cudaKernelNodeAttrValue,
) -> cudaError_t {
    let attr = to_hip_kernel_node_attr_id(attr);
    let p_value_out = value_out;
    let mut value_out = to_hip_kernel_node_attr_value(*value_out);
    let status = to_cuda(hipGraphKernelNodeGetAttribute(
        h_node.cast(),
        attr,
        &mut value_out,
    ));
    *p_value_out = to_cuda_kernel_node_attr_value(value_out);
    status
}

unsafe fn graph_kernel_node_set_attribute(
    h_node: cudaGraphNode_t,
    attr: cudaKernelNodeAttrID,
    value: *const cudaKernelNodeAttrValue,
) -> cudaError_t {
    let attr = to_hip_kernel_node_attr_id(attr);
    let value = to_hip_kernel_node_attr_value(*value);
    to_cuda(hipGraphKernelNodeSetAttribute(
        h_node.cast(),
        attr,
        &value,
    ))
}

fn to_cuda_pitched_ptr(ptr: hipPitchedPtr) -> cudaPitchedPtr {
    cudaPitchedPtr {
        ptr: ptr.ptr,
        pitch: ptr.pitch,
        xsize: ptr.xsize,
        ysize: ptr.ysize,
    }
}

fn to_hip_pitched_ptr(ptr: cudaPitchedPtr) -> hipPitchedPtr {
    hipPitchedPtr {
        ptr: ptr.ptr,
        pitch: ptr.pitch,
        xsize: ptr.xsize,
        ysize: ptr.ysize,
    }
}

fn to_cuda_memcpy_3d_params(copy_params: hipMemcpy3DParms) -> cudaMemcpy3DParms {
    cudaMemcpy3DParms {
        srcArray: copy_params.srcArray.cast(),
        srcPos: to_cuda_pos(copy_params.srcPos),
        srcPtr: to_cuda_pitched_ptr(copy_params.srcPtr),
        dstArray: copy_params.dstArray.cast(),
        dstPos: to_cuda_pos(copy_params.dstPos),
        dstPtr: to_cuda_pitched_ptr(copy_params.srcPtr),
        extent: to_cuda_extent(copy_params.extent),
        kind: to_cuda_memcpy_kind(copy_params.kind),
    }
}

fn to_hip_memcpy_3d_params(copy_params: cudaMemcpy3DParms) -> hipMemcpy3DParms {
    hipMemcpy3DParms {
        srcArray: copy_params.srcArray.cast(),
        srcPos: to_hip_pos(copy_params.srcPos),
        srcPtr: to_hip_pitched_ptr(copy_params.srcPtr),
        dstArray: copy_params.dstArray.cast(),
        dstPos: to_hip_pos(copy_params.dstPos),
        dstPtr: to_hip_pitched_ptr(copy_params.srcPtr),
        extent: to_hip_extent(copy_params.extent),
        kind: to_hip_memcpy_kind(copy_params.kind),
    }
}

unsafe fn graph_add_memcpy_node(
    p_graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    p_dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    p_copy_params: *const cudaMemcpy3DParms,
) -> cudaError_t {
    let p_copy_params = to_hip_memcpy_3d_params(*p_copy_params);
    to_cuda(hipGraphAddMemcpyNode(
        p_graph_node.cast(),
        graph.cast(),
        p_dependencies.cast(),
        num_dependencies,
        &p_copy_params,
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

unsafe fn graph_memcpy_node_get_params(
    node: cudaGraphNode_t,
    p_node_params: *mut cudaMemcpy3DParms,
) -> cudaError_t {
    let mut node_params = to_hip_memcpy_3d_params(*p_node_params);
    let status = to_cuda(hipGraphMemcpyNodeGetParams(
        node.cast(),
        &mut node_params,
    ));
    *p_node_params = to_cuda_memcpy_3d_params(node_params);
    status
}

unsafe fn graph_memcpy_node_set_params(
    node: cudaGraphNode_t,
    p_node_params: *const cudaMemcpy3DParms,
) -> cudaError_t {
    let p_node_params = to_hip_memcpy_3d_params(*p_node_params);
    to_cuda(hipGraphMemcpyNodeSetParams(
        node.cast(),
        &p_node_params,
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

fn to_cuda_memset_params(memset_params: hipMemsetParams) -> cudaMemsetParams {
    cudaMemsetParams {
        dst: memset_params.dst,
        pitch: memset_params.pitch,
        value: memset_params.value,
        elementSize: memset_params.elementSize,
        width: memset_params.width,
        height: memset_params.height,
    }
}

fn to_hip_memset_params(memset_params: cudaMemsetParams) -> hipMemsetParams {
    hipMemsetParams {
        dst: memset_params.dst,
        elementSize: memset_params.elementSize,
        height: memset_params.height,
        pitch: memset_params.pitch,
        value: memset_params.value,
        width: memset_params.width,
    }
}

unsafe fn graph_add_memset_node(
    p_graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    p_dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    p_memset_params: *const cudaMemsetParams,
) -> cudaError_t {
    let p_memset_params = to_hip_memset_params(*p_memset_params);
    to_cuda(hipGraphAddMemsetNode(
        p_graph_node.cast(),
        graph.cast(),
        p_dependencies.cast(),
        num_dependencies,
        &p_memset_params,
    ))
}

unsafe fn graph_memset_node_get_params(
    node: cudaGraphNode_t,
    p_node_params: *mut cudaMemsetParams,
) -> cudaError_t {
    let mut memset_params = to_hip_memset_params(*p_node_params);
    let status = to_cuda(hipGraphMemsetNodeGetParams(
        node.cast(),
        &mut memset_params,
    ));
    *p_node_params = to_cuda_memset_params(memset_params);
    status
}

unsafe fn graph_memset_node_set_params(
    node: cudaGraphNode_t,
    p_node_params: *const cudaMemsetParams,
) -> cudaError_t {
    let p_node_params = to_hip_memset_params(*p_node_params);
    to_cuda(hipGraphMemsetNodeSetParams(
        node.cast(),
        &p_node_params,
    ))
}

fn to_cuda_host_node_params(node_params: hipHostNodeParams) -> cudaHostNodeParams {
    cudaHostNodeParams {
        fn_: node_params.fn_,
        userData: node_params.userData,
    }
}

fn to_hip_host_node_params(node_params: cudaHostNodeParams) -> hipHostNodeParams {
    hipHostNodeParams {
        fn_: node_params.fn_,
        userData: node_params.userData,
    }
}

unsafe fn graph_add_host_node(
    p_graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    p_dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    p_node_params: *const cudaHostNodeParams,
) -> cudaError_t {
    let p_node_params = to_hip_host_node_params(*p_node_params);
    to_cuda(hipGraphAddHostNode(
        p_graph_node.cast(),
        graph.cast(),
        p_dependencies.cast(),
        num_dependencies,
        &p_node_params,
    ))
}

unsafe fn graph_host_node_get_params(
    node: cudaGraphNode_t,
    p_node_params: *mut cudaHostNodeParams,
) -> cudaError_t {
    let mut node_params = to_hip_host_node_params(*p_node_params);
    let status = to_cuda(hipGraphHostNodeGetParams(
        node.cast(),
        &mut node_params,
    ));
    *p_node_params = to_cuda_host_node_params(node_params);
    status
}

unsafe fn graph_host_node_set_params(
    node: cudaGraphNode_t,
    p_node_params: *const cudaHostNodeParams,
) -> cudaError_t {
    let p_node_params = to_hip_host_node_params(*p_node_params);
    to_cuda(hipGraphHostNodeSetParams(
        node.cast(),
        &p_node_params,
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

fn to_cuda_mem_location(location: hipMemLocation) -> cudaMemLocation {
    cudaMemLocation {
        type_: to_cuda_mem_location_type(location.type_),
        id: location.id,
    }
}

fn to_hip_mem_location(location: cudaMemLocation) -> hipMemLocation {
    hipMemLocation {
        type_: to_hip_mem_location_type(location.type_),
        id: location.id,
    }
}

unsafe fn to_cuda_mem_alloc_node_params(node_params: hipMemAllocNodeParams) -> cudaMemAllocNodeParams {
    let mut access_descs = vec![];
    for i in 0..node_params.accessDescCount {
        let desc = *node_params.accessDescs.add(i);
        access_descs.push(cudaMemAccessDesc {
            location: to_cuda_mem_location(desc.location),
            flags: to_cuda_mem_access_flags(desc.flags),
        });
    }
    cudaMemAllocNodeParams {
        poolProps: to_cuda_mem_pool_props(node_params.poolProps),
        accessDescs: access_descs.as_ptr(),
        accessDescCount: node_params.accessDescCount,
        bytesize: node_params.bytesize,
        dptr: node_params.dptr,
    }
}

unsafe fn to_hip_mem_alloc_node_params(node_params: cudaMemAllocNodeParams) -> hipMemAllocNodeParams {
    let mut access_descs = vec![];
    for i in 0..node_params.accessDescCount {
        let desc = *node_params.accessDescs.add(i);
        access_descs.push(hipMemAccessDesc {
            location: to_hip_mem_location(desc.location),
            flags: to_hip_mem_access_flags(desc.flags),
        });
    }
    hipMemAllocNodeParams {
        poolProps: hipMemPoolProps {
            allocType: to_hip_mem_allocation_type(node_params.poolProps.allocType),
            handleTypes: to_hip_mem_allocation_handle_type(node_params.poolProps.handleTypes),
            location: to_hip_mem_location(node_params.poolProps.location),
            win32SecurityAttributes: node_params.poolProps.win32SecurityAttributes,
            reserved: [0; 64],
        },
        accessDescs: access_descs.as_ptr(),
        accessDescCount: node_params.accessDescCount,
        bytesize: node_params.bytesize,
        dptr: node_params.dptr,
    }
}

unsafe fn graph_add_mem_alloc_node(
    p_graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    p_dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    node_params: *mut cudaMemAllocNodeParams,
) -> cudaError_t {
    let mut node_params_d = to_hip_mem_alloc_node_params(*node_params);
    let status = to_cuda(hipGraphAddMemAllocNode(
        p_graph_node.cast(),
        graph.cast(),
        p_dependencies.cast(),
        num_dependencies,
        &mut node_params_d,
    ));
    *node_params = to_cuda_mem_alloc_node_params(node_params_d);
    status
}

unsafe fn graph_mem_alloc_node_get_params(
    node: cudaGraphNode_t,
    params_out: *mut cudaMemAllocNodeParams,
) -> cudaError_t {
    let mut params = to_hip_mem_alloc_node_params(*params_out);
    let status = to_cuda(hipGraphMemAllocNodeGetParams(
        node.cast(),
        &mut params,
    ));
    *params_out = to_cuda_mem_alloc_node_params(params);
    status
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

unsafe fn device_get_graph_mem_attribute(
    device: i32,
    attr: cudaGraphMemAttributeType,
    value: *mut ::std::os::raw::c_void,
) -> cudaError_t {
    let attr = to_hip_graph_mem_attribute_type(attr);
    to_cuda(hipDeviceGetGraphMemAttribute(
        device,
        attr,
        value,
    ))
}

unsafe fn device_set_graph_mem_attribute(
    device: i32,
    attr: cudaGraphMemAttributeType,
    value: *mut ::std::os::raw::c_void,
) -> cudaError_t {
    let attr = to_hip_graph_mem_attribute_type(attr);
    to_cuda(hipDeviceSetGraphMemAttribute(
        device,
        attr,
        value,
    ))
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

unsafe fn graph_node_get_type(
    node: cudaGraphNode_t,
    p_type: *mut cudaGraphNodeType,
) -> cudaError_t {
    let mut type_ = hipGraphNodeType(0);
    let status = to_cuda(hipGraphNodeGetType(
        node.cast(),
        &mut type_,
    ));
    *p_type = to_cuda_graph_node_type(type_);
    status
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

unsafe fn graph_exec_kernel_node_set_params(
    h_graph_exec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    p_node_params: *const cudaKernelNodeParams,
) -> cudaError_t {
    let p_node_params = to_hip_kernel_node_params(*p_node_params);
    to_cuda(hipGraphExecKernelNodeSetParams(
        h_graph_exec.cast(),
        node.cast(),
        &p_node_params,
    ))
}

unsafe fn graph_exec_memcpy_node_set_params(
    h_graph_exec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    p_node_params: *const cudaMemcpy3DParms,
) -> cudaError_t {
    let mut p_node_params = to_hip_memcpy_3d_params(*p_node_params);
    to_cuda(hipGraphExecMemcpyNodeSetParams(
        h_graph_exec.cast(),
        node.cast(),
        &mut p_node_params,
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

unsafe fn graph_exec_memset_node_set_params(
    h_graph_exec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    p_node_params: *const cudaMemsetParams,
) -> cudaError_t {
    let p_node_params = to_hip_memset_params(*p_node_params);
    to_cuda(hipGraphExecMemsetNodeSetParams(
        h_graph_exec.cast(),
        node.cast(),
        &p_node_params,
    ))
}

unsafe fn graph_exec_host_node_set_params(
    h_graph_exec: cudaGraphExec_t,
    node: cudaGraphNode_t,
    p_node_params: *const cudaHostNodeParams,
) -> cudaError_t {
    let p_node_params = to_hip_host_node_params(*p_node_params);
    to_cuda(hipGraphExecHostNodeSetParams(
        h_graph_exec.cast(),
        node.cast(),
        &p_node_params,
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

unsafe fn graph_exec_update(
    h_graph_exec: cudaGraphExec_t,
    h_graph: cudaGraph_t,
    h_error_node_out: *mut cudaGraphNode_t,
    update_result_out: *mut cudaGraphExecUpdateResult,
) -> cudaError_t {
    let p_update_result_out = update_result_out;
    let mut update_result_out = to_hip_graph_exec_update_result(*update_result_out);
    let status = to_cuda(hipGraphExecUpdate(
        h_graph_exec.cast(),
        h_graph.cast(),
        h_error_node_out.cast(),
        &mut update_result_out,
    ));
    *p_update_result_out = to_cuda_graph_exec_update_result(update_result_out);
    status
}

unsafe fn graph_upload(
    graph_exec: cudaGraphExec_t,
    stream: cudaStream_t,
) -> cudaError_t {
    to_cuda(hipGraphUpload(
        graph_exec.cast(),
        stream.cast(),
    ))
}

unsafe fn graph_launch(
    graph_exec: cudaGraphExec_t,
    stream: cudaStream_t,
) -> cudaError_t {
    to_cuda(hipGraphLaunch(
        graph_exec.cast(),
        stream.cast(),
    ))
}

unsafe fn graph_launch_ptsz(
    graph_exec: cudaGraphExec_t,
    stream: cudaStream_t,
) -> cudaError_t {
    to_cuda(hipGraphLaunch_spt(
        graph_exec.cast(),
        stream.cast(),
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
