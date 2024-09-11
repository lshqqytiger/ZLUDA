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
    tid: *mut uint3,
    bid: *mut uint3,
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
        tid.cast(),
        bid.cast(),
        b_dim.cast(),
        g_dim.cast(),
        w_size,
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

unsafe fn get_device_count(count: *mut i32) -> cudaError_t {
    to_cuda(hipGetDeviceCount(count))
}

unsafe fn get_device(device: *mut i32) -> cudaError_t {
    to_cuda(hipGetDevice(device))
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
    cudaError_t((cu_stream_create_with_priority)(p_stream.cast(), flags, priority).0)
}

unsafe fn stream_synchronize(stream: cudaStream_t) -> cudaError_t {
    let stream = to_stream(stream);
    to_cuda(hipStreamSynchronize(stream))
}

unsafe fn stream_is_capturing(
    stream: cudaStream_t,
    p_capture_status: *mut cudaStreamCaptureStatus,
) -> cudaError_t {
    let stream = to_stream(stream);
    let mut capture_status = mem::zeroed();
    let status = to_cuda(hipStreamIsCapturing(stream, &mut capture_status));
    *p_capture_status = to_cuda_stream_capture_status(capture_status);
    status
}

unsafe fn malloc(dev_ptr: *mut *mut ::std::os::raw::c_void, size: usize) -> cudaError_t {
    to_cuda(hipMalloc(dev_ptr, size))
}

unsafe fn free(dev_ptr: *mut ::std::os::raw::c_void) -> cudaError_t {
    to_cuda(hipFree(dev_ptr))
}

unsafe fn mem_get_info(free: *mut usize, total: *mut usize) -> cudaError_t {
    to_cuda(hipMemGetInfo(free, total))
}

unsafe fn memcpy_async(
    dst: *mut ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    let kind = memcpy_kind(kind);
    let stream = to_stream(stream);
    to_cuda(hipMemcpyAsync(dst, src, count, kind, stream))
}
