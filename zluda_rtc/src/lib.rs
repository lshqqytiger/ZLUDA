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

fn to_nvrtc(status: hiprtc_sys::hiprtcResult) -> nvrtcResult {
    match status {
        hiprtc_sys::hiprtcResult::HIPRTC_SUCCESS => nvrtcResult::NVRTC_SUCCESS,
        err => panic!("[ZLUDA] HIPRTC failed: {}", err.0),
    }
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
    to_nvrtc(hiprtcCompileProgram(prog.cast(), num_options, options.cast_mut()))
}
