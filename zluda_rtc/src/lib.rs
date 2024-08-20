#![allow(warnings)]
mod nvrtc;
pub use nvrtc::*;

use hiprtc_sys::*;

#[cfg(debug_assertions)]
fn unsupported() -> nvrtcResult {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
fn unsupported() -> nvrtcResult {
    nvrtcResult::NVRTC_ERROR_INTERNAL_ERROR
}

const NVRTC_VERSION_MAJOR: i32 = 11;
const NVRTC_VERSION_MINOR: i32 = 7;
const SUPPORTED_OPTIONS: [&'static str; 2] = ["--std", "-default-device"];

fn to_nvrtc(status: hiprtc_sys::hiprtcResult) -> nvrtcResult {
    match status {
        hiprtc_sys::hiprtcResult::HIPRTC_SUCCESS => nvrtcResult::NVRTC_SUCCESS,
        hiprtc_sys::hiprtcResult::HIPRTC_ERROR_INVALID_PROGRAM => nvrtcResult::NVRTC_ERROR_INVALID_PROGRAM,
        hiprtc_sys::hiprtcResult::HIPRTC_ERROR_COMPILATION => nvrtcResult::NVRTC_ERROR_COMPILATION,
        hiprtc_sys::hiprtcResult::HIPRTC_ERROR_INTERNAL_ERROR => nvrtcResult::NVRTC_ERROR_INTERNAL_ERROR,
        err => panic!("[ZLUDA] HIPRTC failed: {}", err.0),
    }
}

fn to_hiprtc(status: nvrtcResult) -> hiprtc_sys::hiprtcResult {
    match status {
        nvrtcResult::NVRTC_SUCCESS => hiprtc_sys::hiprtcResult::HIPRTC_SUCCESS,
        nvrtcResult::NVRTC_ERROR_INVALID_PROGRAM => hiprtc_sys::hiprtcResult::HIPRTC_ERROR_INVALID_PROGRAM,
        nvrtcResult::NVRTC_ERROR_COMPILATION => hiprtc_sys::hiprtcResult::HIPRTC_ERROR_COMPILATION,
        nvrtcResult::NVRTC_ERROR_INTERNAL_ERROR => hiprtc_sys::hiprtcResult::HIPRTC_ERROR_INTERNAL_ERROR,
        err => panic!("[ZLUDA] HIPRTC failed: {}", err.0),
    }
}

unsafe fn get_error_string(result: nvrtcResult) -> *const ::std::os::raw::c_char {
    hiprtcGetErrorString(to_hiprtc(result))
}

unsafe fn version(
    major: *mut i32,
    minor: *mut i32,
) -> nvrtcResult {
    *major = NVRTC_VERSION_MAJOR;
    *minor = NVRTC_VERSION_MINOR;
    nvrtcResult::NVRTC_SUCCESS
}

unsafe fn create_program(
    prog: *mut nvrtcProgram,
    src: *const std::ffi::c_char,
    name: *const std::ffi::c_char,
    num_headers: i32,
    headers: *const *const std::ffi::c_char,
    include_names: *const *const std::ffi::c_char,
) -> nvrtcResult {
    to_nvrtc(hiprtcCreateProgram(
        prog.cast(),
        src,
        name,
        num_headers,
        headers.cast_mut(),
        include_names.cast_mut(),
    ))
}

unsafe fn destroy_program(
    prog: *mut nvrtcProgram,
) -> nvrtcResult {
    to_nvrtc(hiprtcDestroyProgram(prog.cast()))
}

unsafe fn compile_program(
    prog: nvrtcProgram,
    num_options: i32,
    options: *const *const std::ffi::c_char,
) -> nvrtcResult {
    let mut arguments: Vec<*const std::ffi::c_char> = Vec::new();
    for i in 0..num_options {
        let option_string = std::ffi::CStr::from_ptr(*options.offset(i as _)).to_str().unwrap();
        let option: Vec<&str> = option_string.split("=").collect();
        if SUPPORTED_OPTIONS.contains(&option[0]) {
            let cstr = std::ffi::CString::new(option_string).unwrap();
            arguments.push(cstr.as_ptr());
        }
    }
    // TODO
    to_nvrtc(hiprtcCompileProgram(
        prog.cast(),
        arguments.len() as _,
        arguments.as_mut_ptr(),
    ))
}

unsafe fn get_code_size(prog: nvrtcProgram, code_size_ret: *mut usize) -> nvrtcResult {
    to_nvrtc(hiprtcGetCodeSize(prog.cast(), code_size_ret))
}

unsafe fn get_code(prog: nvrtcProgram, code: *mut std::ffi::c_char) -> nvrtcResult {
    to_nvrtc(hiprtcGetCode(prog.cast(), code))
}

unsafe fn get_lowered_name(
    prog: nvrtcProgram,
    name_expression: *const std::ffi::c_char,
    lowered_name: *mut *const std::ffi::c_char,
) -> nvrtcResult {
    to_nvrtc(hiprtcGetLoweredName(prog.cast(), name_expression, lowered_name))
}
