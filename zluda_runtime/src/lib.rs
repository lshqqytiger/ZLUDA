mod cudart;
pub use cudart::*;

use hip_runtime_sys::*;
use std::ptr;

use std::backtrace::Backtrace;

#[cfg(debug_assertions)]
fn unsupported() -> cudaError_t {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
fn unsupported() -> cudaError_t {
    println!("{}", Backtrace::force_capture());
    panic!();
    cudaError::cudaErrorNotSupported
}

fn no_corresponding_function() -> cudaError_t {
    cudaError::cudaSuccess
}

fn to_cuda(status: hipError_t) -> cudaError_t {
    match status {
        hipError_t::hipSuccess => cudaError_t::cudaSuccess,
        hipError_t::hipErrorNotSupported => cudaError_t::cudaErrorNotSupported,
        err => panic!("[ZLUDA] HIP_RUNTIME failed: {}", err.0),
    }
}

fn to_hip(status: cudaError_t) -> hipError_t {
    match status {
        cudaError_t::cudaSuccess => hipError_t::hipSuccess,
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

fn to_cuda_graph_mem_attribute_type(graph_mem_attribute_type: hipGraphMemAttributeType) -> cudaGraphMemAttributeType {
    match graph_mem_attribute_type {
        _ => panic!()
    }
}

fn to_hip_graph_mem_attribute_type(graph_mem_attribute_type: cudaGraphMemAttributeType) -> hipGraphMemAttributeType {
    match graph_mem_attribute_type {
        _ => panic!()
    }
}

fn to_cuda_device_p2p_attr(device_p2p_attr: hipDeviceP2PAttr) -> cudaDeviceP2PAttr {
    match device_p2p_attr {
        _ => panic!()
    }
}

fn to_hip_device_p2p_attr(device_p2p_attr: cudaDeviceP2PAttr) -> hipDeviceP2PAttr {
    match device_p2p_attr {
        _ => panic!()
    }
}

fn to_cuda_func_attribute(func_attribute: hipFuncAttribute) -> cudaFuncAttribute {
    match func_attribute {
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
        _ => panic!()
    }
}

fn to_hip_memcpy_kind(memcpy_kind: cudaMemcpyKind) -> hipMemcpyKind {
    match memcpy_kind {
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
) -> ::std::os::raw::c_uint {
    let grid_dim = hip_runtime_api::dim3 {
        x: grid_dim.x,
        y: grid_dim.y,
        z: grid_dim.z,
    };
    let block_dim = hip_runtime_api::dim3 {
        x: block_dim.x,
        y: block_dim.y,
        z: block_dim.z,
    };
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
    ext: i32,
    size: usize,
    constant: i32,
    global: i32,
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
    dim: ::std::os::raw::c_int,
    ext: ::std::os::raw::c_int,
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
    let mut out_cache_confg = to_hip_func_cache(*p_cache_config);
    let status = to_cuda(hipDeviceGetCacheConfig(&mut out_cache_confg));
    *p_cache_config = to_cuda_func_cache(out_cache_confg);
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

unsafe fn event_record_with_flags(
    event: cudaEvent_t,
    stream: cudaStream_t,
    _flags: u32,
) -> cudaError_t {
    event_record(event, stream)
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

unsafe fn destroy_external_semaphore(
    ext_sem: cudaExternalSemaphore_t,
) -> cudaError_t {
    to_cuda(hipDestroyExternalSemaphore(
        ext_sem.cast(),
    ))
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
    let attr_d = *attr;
    let mut attr_hip = hipFuncAttributes {
        binaryVersion: attr_d.binaryVersion,
        cacheModeCA: attr_d.cacheModeCA,
        constSizeBytes: attr_d.constSizeBytes,
        localSizeBytes: attr_d.localSizeBytes,
        maxDynamicSharedSizeBytes: attr_d.maxDynamicSharedSizeBytes,
        maxThreadsPerBlock: attr_d.maxThreadsPerBlock,
        numRegs: attr_d.numRegs,
        preferredShmemCarveout: attr_d.preferredShmemCarveout,
        ptxVersion: attr_d.ptxVersion,
        sharedSizeBytes: attr_d.sharedSizeBytes,
    };
    let status = to_cuda(hipFuncGetAttributes(
        &mut attr_hip,
        func,
    ));
    *attr = cudaFuncAttributes {
        sharedSizeBytes: attr_hip.sharedSizeBytes,
        constSizeBytes: attr_hip.constSizeBytes,
        localSizeBytes: attr_hip.localSizeBytes,
        maxThreadsPerBlock: attr_hip.maxThreadsPerBlock,
        numRegs: attr_hip.numRegs,
        ptxVersion: attr_hip.ptxVersion,
        binaryVersion: attr_hip.binaryVersion,
        cacheModeCA: attr_hip.cacheModeCA,
        maxDynamicSharedSizeBytes: attr_hip.maxDynamicSharedSizeBytes,
        preferredShmemCarveout: attr_hip.preferredShmemCarveout,
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
    let mut desc_mut = to_hip_channel_format_desc(*desc);
    let extent_d = *extent;
    let mut extent_hip = to_hip_extent(extent_d);
    let status = to_cuda(hipArrayGetInfo(
        &mut desc_mut,
        &mut extent_hip,
        flags,
        array.cast(),
    ));
    *desc = to_cuda_channel_format_desc(desc_mut);
    *extent = to_cuda_extent(extent_hip);
    status
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

unsafe fn free_async(
    dev_ptr: *mut ::std::os::raw::c_void,
    h_stream: cudaStream_t,
) -> cudaError_t {
    to_cuda(hipFreeAsync(
        dev_ptr,
        h_stream.cast(),
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
    let tex_channel_desc = to_cuda_channel_format_desc(tex.channelDesc);
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
    let mut desc_mut = to_hip_channel_format_desc(*desc);
    let status = to_cuda(hipGetChannelDesc(
        &mut desc_mut,
        array.cast(),
    ));
    *desc = to_cuda_channel_format_desc(desc_mut);
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

unsafe fn create_texture_object(
    p_tex_object: *mut cudaTextureObject_t,
    p_res_desc: *const cudaResourceDesc,
    p_tex_desc: *const cudaTextureDesc,
    p_res_view_desc: *const cudaResourceViewDesc,
) -> cudaError_t {
    let p_res = to_hip_resource_desc(*p_res_desc);
    let p_tex = to_hip_texture_desc(*p_tex_desc);
    let p_res_view = *p_res_view_desc;
    let p_res_view = hipResourceViewDesc {
        format: to_hip_resource_view_format(p_res_view.format),
        width: p_res_view.width,
        height: p_res_view.height,
        depth: p_res_view.depth,
        firstMipmapLevel: p_res_view.firstMipmapLevel,
        lastMipmapLevel: p_res_view.lastMipmapLevel,
        firstLayer: p_res_view.firstLayer,
        lastLayer: p_res_view.lastLayer,
    };
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
    let mut p_res_mut = to_hip_resource_desc(*p_res_desc);
    let status = to_cuda(hipGetTextureObjectResourceDesc(
        &mut p_res_mut,
        tex_object as _,
    ));
    *p_res_desc = to_cuda_resource_desc(p_res_mut);
    status
}

unsafe fn get_texture_object_texture_desc(
    p_tex_desc: *mut cudaTextureDesc,
    tex_object: cudaTextureObject_t,
) -> cudaError_t {
    let mut p_tex = to_hip_texture_desc(*p_tex_desc);
    let status = to_cuda(hipGetTextureObjectTextureDesc(
        &mut p_tex,
        tex_object as _,
    ));
    *p_tex_desc = to_cuda_texture_desc(p_tex);
    status
}

unsafe fn get_texture_object_resource_view_desc(
    p_res_view_desc: *mut cudaResourceViewDesc,
    tex_object: cudaTextureObject_t,
) -> cudaError_t {
    let p_res_view = *p_res_view_desc;
    let mut p_res_view = hipResourceViewDesc {
        format: to_hip_resource_view_format(p_res_view.format),
        width: p_res_view.width,
        height: p_res_view.height,
        depth: p_res_view.depth,
        firstMipmapLevel: p_res_view.firstMipmapLevel,
        lastMipmapLevel: p_res_view.lastMipmapLevel,
        firstLayer: p_res_view.firstLayer,
        lastLayer: p_res_view.lastLayer,
    };
    let status = to_cuda(hipGetTextureObjectResourceViewDesc(
        &mut p_res_view,
        tex_object as _,
    ));
    *p_res_view_desc = cudaResourceViewDesc {
        format: to_cuda_resource_view_format(p_res_view.format),
        width: p_res_view.width,
        height: p_res_view.height,
        depth: p_res_view.depth,
        firstMipmapLevel: p_res_view.firstMipmapLevel,
        lastMipmapLevel: p_res_view.lastMipmapLevel,
        firstLayer: p_res_view.firstLayer,
        lastLayer: p_res_view.lastLayer,
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

unsafe fn graph_add_kernel_node(
    p_graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    p_dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    p_node_params: *const cudaKernelNodeParams,
) -> cudaError_t {
    let p_node_params = *p_node_params;
    let p_node_params = hipKernelNodeParams {
        blockDim: to_hip_dim3(p_node_params.blockDim),
        extra: p_node_params.extra,
        func: p_node_params.func,
        gridDim: to_hip_dim3(p_node_params.gridDim),
        kernelParams: p_node_params.kernelParams,
        sharedMemBytes: p_node_params.sharedMemBytes,
    };
    to_cuda(hipGraphAddKernelNode(
        p_graph_node.cast(),
        graph.cast(),
        p_dependencies.cast(),
        num_dependencies,
        &p_node_params,
    ))
}

fn to_hip_pitched_ptr(ptr: cudaPitchedPtr) -> hipPitchedPtr {
    hipPitchedPtr {
        ptr: ptr.ptr,
        pitch: ptr.pitch,
        xsize: ptr.xsize,
        ysize: ptr.ysize,
    }
}

unsafe fn graph_add_memcpy_node(
    p_graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    p_dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    p_copy_params: *const cudaMemcpy3DParms,
) -> cudaError_t {
    let p_copy_params = *p_copy_params;
    let p_copy_params = hipMemcpy3DParms {
        srcArray: p_copy_params.srcArray.cast(),
        srcPos: to_hip_pos(p_copy_params.srcPos),
        srcPtr: to_hip_pitched_ptr(p_copy_params.srcPtr),
        dstArray: p_copy_params.dstArray.cast(),
        dstPos: to_hip_pos(p_copy_params.dstPos),
        dstPtr: to_hip_pitched_ptr(p_copy_params.srcPtr),
        extent: to_hip_extent(p_copy_params.extent),
        kind: to_hip_memcpy_kind(p_copy_params.kind),
    };
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

unsafe fn graph_add_host_node(
    p_graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    p_dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    p_node_params: *const cudaHostNodeParams,
) -> cudaError_t {
    let p_node_params = *p_node_params;
    let p_node_params = hipHostNodeParams {
        fn_: p_node_params.fn_,
        userData: p_node_params.userData,
    };
    to_cuda(hipGraphAddHostNode(
        p_graph_node.cast(),
        graph.cast(),
        p_dependencies.cast(),
        num_dependencies,
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

unsafe fn graph_add_mem_alloc_node(
    p_graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    p_dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    node_params: *mut cudaMemAllocNodeParams,
) -> cudaError_t {
    let node_params_d = *node_params;
    let mut access_descs_hip = vec![];
    for i in 0..node_params_d.accessDescCount {
        let desc = *node_params_d.accessDescs.add(i);
        access_descs_hip.push(hipMemAccessDesc {
            location: to_hip_mem_location(desc.location),
            flags: to_hip_mem_access_flags(desc.flags),
        });
    }
    let mut node_params_d = hipMemAllocNodeParams {
        poolProps: hipMemPoolProps {
            allocType: to_hip_mem_allocation_type(node_params_d.poolProps.allocType),
            handleTypes: to_hip_mem_allocation_handle_type(node_params_d.poolProps.handleTypes),
            location: to_hip_mem_location(node_params_d.poolProps.location),
            win32SecurityAttributes: node_params_d.poolProps.win32SecurityAttributes,
            reserved: [0; 64],
        },
        accessDescs: access_descs_hip.as_ptr(),
        accessDescCount: node_params_d.accessDescCount,
        bytesize: node_params_d.bytesize,
        dptr: node_params_d.dptr,
    };
    let status = to_cuda(hipGraphAddMemAllocNode(
        p_graph_node.cast(),
        graph.cast(),
        p_dependencies.cast(),
        num_dependencies,
        &mut node_params_d,
    ));
    let mut access_descs_cuda = vec![];
    for i in 0..node_params_d.accessDescCount {
        let desc = *node_params_d.accessDescs.add(i);
        access_descs_cuda.push(cudaMemAccessDesc {
            location: to_cuda_mem_location(desc.location),
            flags: to_cuda_mem_access_flags(desc.flags),
        });
    }
    *node_params = cudaMemAllocNodeParams {
        poolProps: cudaMemPoolProps {
            allocType: to_cuda_mem_allocation_type(node_params_d.poolProps.allocType),
            handleTypes: to_cuda_mem_allocation_handle_type(node_params_d.poolProps.handleTypes),
            location: to_cuda_mem_location(node_params_d.poolProps.location),
            win32SecurityAttributes: node_params_d.poolProps.win32SecurityAttributes,
            reserved: [0; 64],
        },
        accessDescs: access_descs_cuda.as_ptr(),
        accessDescCount: node_params_d.accessDescCount,
        bytesize: node_params_d.bytesize,
        dptr: node_params_d.dptr,
    };
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
