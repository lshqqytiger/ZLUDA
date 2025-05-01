#![allow(warnings)]
mod cudart;
pub use cudart::*;
mod extra;
pub use extra::*;
mod decl;

use hip_runtime_sys::*;
use lazy_static::lazy_static;
use rustc_hash::FxHashMap;
use std::{mem, ptr, sync::Mutex};

#[cfg(debug_assertions)]
fn unsupported() -> cudaError_t {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
fn unsupported() -> cudaError_t {
    cudaError_t::cudaErrorNotSupported
}

decl!(cudaLaunchKernel_ptsz);
decl!(cudaMemcpy3D_ptds);
decl!(cudaMemcpy2D_ptds);
decl!(cudaMemcpyToSymbol_ptds);
decl!(cudaGLUnregisterBufferObject);
decl!(cudaGLUnmapBufferObject);
decl!(cudaGLMapBufferObject);
decl!(cudaGLRegisterBufferObject);

#[no_mangle]
pub extern "system" fn cudaProfilerInitialize(
    configFile: *const ::std::os::raw::c_char,
    outputFile: *const ::std::os::raw::c_char,
    outputMode: cudaOutputMode_t,
) -> cudaError_t {
    unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cudaProfilerStart() -> cudaError_t {
    unsupported()
}

#[no_mangle]
pub unsafe extern "system" fn cudaProfilerStop() -> cudaError_t {
    unsupported()
}

#[repr(C)]
#[derive(Copy, Clone)]
struct uint3([::std::os::raw::c_uint; 3]);

#[derive(PartialEq, Eq, Hash)]
struct HostFunction(*const std::ffi::c_void);

unsafe impl Send for HostFunction {}

impl Into<HostFunction> for *const std::ffi::c_void {
    fn into(self) -> HostFunction {
        HostFunction(self)
    }
}

impl Into<HostFunction> for *const ::std::os::raw::c_char {
    fn into(self) -> HostFunction {
        HostFunction(self.cast())
    }
}

struct DeviceFunction(*mut std::ffi::c_void);

unsafe impl Send for DeviceFunction {}

impl Into<DeviceFunction> for *mut std::ffi::c_void {
    fn into(self) -> DeviceFunction {
        DeviceFunction(self)
    }
}

lazy_static! {
    static ref FUNCTION_TABLE: Mutex<FxHashMap<HostFunction, DeviceFunction>> =
        Mutex::new(FxHashMap::default());
}

#[no_mangle]
pub extern "system" fn __cudaInitModule(
    fatCubinHandle: *mut *mut std::ffi::c_void,
) -> ::std::os::raw::c_char {
    unimplemented!()
}

#[no_mangle]
pub unsafe extern "system" fn __cudaPopCallConfiguration(
    gridDim: *mut cudart::dim3,
    blockDim: *mut cudart::dim3,
    sharedMem: *mut usize,
    stream: *mut cudaStream_t,
) -> cudaError_t {
    to_cuda(__hipPopCallConfiguration(
        gridDim.cast(),
        blockDim.cast(),
        sharedMem,
        stream.cast(),
    ))
}

#[no_mangle]
pub unsafe extern "system" fn __cudaPushCallConfiguration(
    gridDim: cudart::dim3,
    blockDim: cudart::dim3,
    sharedMem: usize,
    stream: cudaStream_t,
) -> ::std::os::raw::c_uint {
    __hipPushCallConfiguration(
        mem::transmute(gridDim),
        mem::transmute(blockDim),
        sharedMem,
        stream.cast(),
    )
    .0 as _
}

#[no_mangle]
pub unsafe extern "system" fn __cudaRegisterFatBinary(
    fatCubin: *mut std::ffi::c_void,
) -> *mut *mut std::ffi::c_void {
    let lib = hip_common::zluda_ext::get_cuda_library().unwrap();
    let cu_module_load_data = lib
        .get::<unsafe extern "C" fn(
            module: *mut *mut std::ffi::c_void,
            image: *const std::ffi::c_void,
        ) -> cuda_types::CUresult>(b"cuModuleLoadData\0")
        .unwrap();
    let mut module = mem::zeroed();
    cu_module_load_data(&mut module, fatCubin);
    Box::into_raw(Box::new(module))
}

#[no_mangle]
pub extern "system" fn __cudaRegisterFatBinaryEnd(
    fatCubinHandle: *mut *mut std::ffi::c_void,
) -> () {
}

#[no_mangle]
pub unsafe extern "system" fn __cudaRegisterFunction(
    fatCubinHandle: *mut *mut std::ffi::c_void,
    hostFun: *const ::std::os::raw::c_char,
    deviceFun: *mut ::std::os::raw::c_char,
    deviceName: *const ::std::os::raw::c_char,
    thread_limit: ::std::os::raw::c_int,
    tid: *mut uint3,
    bid: *mut uint3,
    bDim: *mut cudart::dim3,
    gDim: *mut cudart::dim3,
    wSize: *mut ::std::os::raw::c_int,
) -> () {
    let lib = hip_common::zluda_ext::get_cuda_library().unwrap();
    let cu_module_get_function = lib
        .get::<unsafe extern "C" fn(
            hfunc: *mut *mut std::ffi::c_void,
            hmod: *mut std::ffi::c_void,
            name: *const ::std::os::raw::c_char,
        ) -> cuda_types::CUresult>(b"cuModuleGetFunction\0")
        .unwrap();
    let mut func = mem::zeroed();
    cu_module_get_function(&mut func, *fatCubinHandle, deviceName);
    let table = &mut *FUNCTION_TABLE.lock().unwrap();
    table.insert(hostFun.into(), func.into());
}

#[no_mangle]
pub extern "system" fn __cudaRegisterHostVar(
    fatCubinHandle: *mut *mut std::ffi::c_void,
    deviceName: *const ::std::os::raw::c_char,
    hostVar: *mut ::std::os::raw::c_char,
    size: usize,
) -> () {
    #[cfg(not(debug_assertions))]
    return ();
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn __cudaRegisterManagedVar(
    fatCubinHandle: *mut *mut std::ffi::c_void,
    hostVarPtrAddress: *mut *mut std::ffi::c_void,
    deviceAddress: *mut ::std::os::raw::c_char,
    deviceName: *const ::std::os::raw::c_char,
    ext: ::std::os::raw::c_int,
    size: usize,
    constant: ::std::os::raw::c_int,
    global: ::std::os::raw::c_int,
) -> () {
    #[cfg(not(debug_assertions))]
    return ();
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn __cudaRegisterSurface(
    fatCubinHandle: *mut *mut std::ffi::c_void,
    hostVar: *const std::ffi::c_void,
    deviceAddress: *const *mut std::ffi::c_void,
    deviceName: *const ::std::os::raw::c_char,
    dim: ::std::os::raw::c_int,
    ext: ::std::os::raw::c_int,
) -> () {
    #[cfg(not(debug_assertions))]
    return ();
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn __cudaRegisterTexture(
    fatCubinHandle: *mut *mut std::ffi::c_void,
    hostVar: *const std::ffi::c_void,
    deviceAddress: *const *mut std::ffi::c_void,
    deviceName: *const ::std::os::raw::c_char,
    dim: ::std::os::raw::c_int,
    norm: ::std::os::raw::c_int,
    ext: ::std::os::raw::c_int,
) -> () {
}

#[no_mangle]
pub unsafe extern "system" fn __cudaRegisterVar(
    fatCubinHandle: *mut *mut std::ffi::c_void,
    hostVar: *mut ::std::os::raw::c_char,
    deviceAddress: *mut ::std::os::raw::c_char,
    deviceName: *const ::std::os::raw::c_char,
    ext: ::std::os::raw::c_int,
    size: usize,
    constant: ::std::os::raw::c_int,
    global: ::std::os::raw::c_int,
) -> () {
}

#[no_mangle]
pub unsafe extern "system" fn __cudaUnregisterFatBinary(
    fatCubinHandle: *mut *mut std::ffi::c_void,
) -> () {
    drop(Box::from_raw(fatCubinHandle));
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
        cudaError_t::cudaErrorInitializationError => hipError_t::hipErrorInitializationError,
        cudaError_t::cudaErrorDeviceUninitialized => hipError_t::hipErrorNotInitialized,
        cudaError_t::cudaErrorInvalidResourceHandle => hipError_t::hipErrorInvalidResourceHandle,
        cudaError_t::cudaErrorNotSupported => hipError_t::hipErrorNotSupported,
        err => panic!("[ZLUDA] HIP Runtime failed: {}", err.0),
    }
}

unsafe fn to_stream(stream: cudaStream_t) -> hipStream_t {
    let lib = hip_common::zluda_ext::get_cuda_library().unwrap();
    let zluda_get_hip_object = lib
        .get::<unsafe extern "C" fn(
            cuda_object: *mut std::ffi::c_void,
            kind: hip_common::zluda_ext::CudaObjectKind,
        ) -> hip_common::zluda_ext::CudaResult<*const std::ffi::c_void>>(
            b"zluda_get_hip_object\0"
        )
        .unwrap();
    let result = zluda_get_hip_object(stream.cast(), hip_common::zluda_ext::CudaObjectKind::Stream);
    if result.return_code != cuda_types::CUresult::CUDA_SUCCESS {
        panic!("[ZLUDA] Invalid CUDA stream.");
    }
    result.value as _
}

fn memcpy_kind(kind: cudaMemcpyKind) -> hipMemcpyKind {
    match kind {
        cudaMemcpyKind::cudaMemcpyHostToHost => hipMemcpyKind::hipMemcpyHostToHost,
        cudaMemcpyKind::cudaMemcpyHostToDevice => hipMemcpyKind::hipMemcpyHostToDevice,
        cudaMemcpyKind::cudaMemcpyDeviceToHost => hipMemcpyKind::hipMemcpyDeviceToHost,
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => hipMemcpyKind::hipMemcpyDeviceToDevice,
        cudaMemcpyKind::cudaMemcpyDefault => hipMemcpyKind::hipMemcpyDefault,
        _ => panic!(),
    }
}

fn to_cuda_stream_capture_status(status: hipStreamCaptureStatus) -> cudaStreamCaptureStatus {
    match status {
        hipStreamCaptureStatus::hipStreamCaptureStatusNone => {
            cudaStreamCaptureStatus::cudaStreamCaptureStatusNone
        }
        hipStreamCaptureStatus::hipStreamCaptureStatusActive => {
            cudaStreamCaptureStatus::cudaStreamCaptureStatusActive
        }
        hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated => {
            cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated
        }
        _ => panic!(),
    }
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

unsafe fn get_last_error() -> cudaError_t {
    to_cuda(hipGetLastError())
}

unsafe fn get_error_string(error: cudaError_t) -> *const ::std::os::raw::c_char {
    hipGetErrorString(to_hip(error))
}

unsafe fn get_device_count(count: *mut i32) -> cudaError_t {
    to_cuda(hipGetDeviceCount(count))
}

unsafe fn get_device_properties(prop: *mut cudaDeviceProp, device: i32) -> cudaError_t {
    if prop == ptr::null_mut() {
        return cudaError_t::cudaErrorInvalidValue;
    }
    let mut hip_props = mem::zeroed();
    let status = hipGetDeviceProperties!(&mut hip_props, device);
    if status != hipError_t::hipSuccess {
        return to_cuda(status);
    }
    (*prop).maxThreadsPerBlock = hip_props.maxThreadsPerBlock;
    (*prop).maxThreadsDim = hip_props.maxThreadsDim;
    (*prop).maxGridSize = hip_props.maxGridSize;
    (*prop).totalConstMem = usize::min(hip_props.totalConstMem, i32::MAX as usize);
    (*prop).warpSize = hip_props.warpSize;
    (*prop).memPitch = usize::min(hip_props.memPitch, i32::MAX as usize);
    (*prop).regsPerBlock = hip_props.regsPerBlock;
    (*prop).clockRate = hip_props.clockRate;
    (*prop).textureAlignment = usize::min(hip_props.textureAlignment, i32::MAX as usize);
    if hip_props.warpSize != 32 {
        (*prop).maxThreadsPerBlock /= 2;
        (*prop).maxThreadsDim[0] /= 2;
        (*prop).maxThreadsDim[1] /= 2;
        (*prop).maxThreadsDim[2] /= 2;
        (*prop).maxGridSize[0] /= 2;
        (*prop).maxGridSize[1] /= 2;
        (*prop).maxGridSize[2] /= 2;
    }
    cudaError_t::cudaSuccess
}

unsafe fn set_device(device: i32) -> cudaError_t {
    to_cuda(hipSetDevice(device))
}

unsafe fn get_device(device: *mut i32) -> cudaError_t {
    to_cuda(hipGetDevice(device))
}

unsafe fn stream_create_with_flags(p_stream: *mut *mut CUstream_st, flags: u32) -> cudaError_t {
    let mut least = 0;
    let mut greatest = 0;
    let status = hipDeviceGetStreamPriorityRange(&mut least, &mut greatest);
    if status != hipError_t::hipSuccess {
        return to_cuda(status);
    }
    stream_create_with_priority(p_stream, flags, least)
}

unsafe fn stream_create_with_priority(
    p_stream: *mut *mut CUstream_st,
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
    cudaError_t(cu_stream_create_with_priority(p_stream.cast(), flags, priority).0)
}

unsafe fn stream_synchronize(stream: *mut CUstream_st) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipStreamSynchronize(stream))
}

unsafe fn stream_is_capturing(
    stream: *mut CUstream_st,
    p_capture_status: *mut cudaStreamCaptureStatus,
) -> cudaError_t {
    let stream = to_stream(stream);
    let mut capture_status = mem::zeroed();
    let status = to_cuda(hipStreamIsCapturing(stream, &mut capture_status));
    *p_capture_status = to_cuda_stream_capture_status(capture_status);
    status
}

unsafe fn event_create_with_flags(event: *mut *mut CUevent_st, flags: u32) -> cudaError_t {
    to_cuda(hipEventCreateWithFlags(event.cast(), flags))
}

unsafe fn event_record(event: *mut CUevent_st, stream: *mut CUstream_st) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipEventRecord(event.cast(), stream.cast()))
}

unsafe fn event_synchronize(event: *mut CUevent_st) -> cudaError_t {
    to_cuda(hipEventSynchronize(event.cast()))
}

unsafe fn launch_kernel(
    func: *const std::ffi::c_void,
    grid_dim: cudart::dim3,
    block_dim: cudart::dim3,
    args: *mut *mut std::ffi::c_void,
    shared_mem: usize,
    stream: *mut CUstream_st,
) -> cudaError_t {
    let lib = hip_common::zluda_ext::get_cuda_library().unwrap();
    let cu_launch_kernel = lib
        .get::<unsafe extern "C" fn(
            f: *mut std::ffi::c_void,
            gridDimX: ::std::os::raw::c_uint,
            gridDimY: ::std::os::raw::c_uint,
            gridDimZ: ::std::os::raw::c_uint,
            blockDimX: ::std::os::raw::c_uint,
            blockDimY: ::std::os::raw::c_uint,
            blockDimZ: ::std::os::raw::c_uint,
            sharedMemBytes: ::std::os::raw::c_uint,
            hStream: *mut std::ffi::c_void,
            kernelParams: *mut *mut ::std::os::raw::c_void,
            extra: *mut *mut ::std::os::raw::c_void,
        ) -> cuda_types::CUresult>(b"cuLaunchKernel\0")
        .unwrap();
    let table = &*FUNCTION_TABLE.lock().unwrap();
    if let Some(func) = table.get(&func.into()) {
        return cudaError_t(
            cu_launch_kernel(
                func.0,
                grid_dim.x,
                grid_dim.y,
                grid_dim.z,
                block_dim.x,
                block_dim.y,
                block_dim.z,
                shared_mem as _,
                stream.cast(),
                args,
                ptr::null_mut(),
            )
            .0,
        );
    }
    cudaError_t::cudaErrorInvalidDeviceFunction
}

unsafe fn malloc(dev_ptr: *mut *mut std::ffi::c_void, size: usize) -> cudaError_t {
    to_cuda(hipMalloc(dev_ptr, size))
}

unsafe fn malloc_host(ptr: *mut *mut std::ffi::c_void, size: usize) -> cudaError_t {
    to_cuda(hipMallocHost(ptr, size))
}

unsafe fn free(dev_ptr: *mut std::ffi::c_void) -> cudaError_t {
    to_cuda(hipFree(dev_ptr))
}

unsafe fn mem_get_info(free: *mut usize, total: *mut usize) -> cudaError_t {
    to_cuda(hipMemGetInfo(free, total))
}

unsafe fn memcpy(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    let kind = memcpy_kind(kind);
    to_cuda(hipMemcpy(dst, src, count, kind))
}

unsafe fn memcpy_async(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: *mut CUstream_st,
) -> cudaError_t {
    let kind = memcpy_kind(kind);
    let stream = to_stream(stream);
    to_cuda(hipMemcpyAsync(dst, src, count, kind, stream))
}
