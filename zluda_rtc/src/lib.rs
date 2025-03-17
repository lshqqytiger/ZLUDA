#[allow(warnings)]
mod nvrtc;
pub use nvrtc::*;

use lazy_static::lazy_static;
use std::{env, ffi::c_char, ptr, result, sync::Mutex};

macro_rules! call {
    ($expr:expr) => {
        #[allow(unused_unsafe)]
        {
            let result = unsafe { $expr };
            if result != nvrtcResult::NVRTC_SUCCESS {
                return Err(result);
            }
        }
    };
}

lazy_static! {
    static ref NVRTC: Mutex<Nvrtc> = Mutex::new(Nvrtc::load());
}

trait Then<T> {
    fn then<F: FnOnce(T)>(self, f: F) -> nvrtcResult;
}

impl<T> Into<nvrtcResult> for result::Result<T, nvrtcResult> {
    fn into(self) -> nvrtcResult {
        match self {
            Ok(_) => nvrtcResult::NVRTC_SUCCESS,
            Err(e) => e,
        }
    }
}

impl<T> Then<T> for result::Result<T, nvrtcResult> {
    fn then<F: FnOnce(T)>(self, f: F) -> nvrtcResult {
        match self {
            Ok(ok) => {
                f(ok);
                nvrtcResult::NVRTC_SUCCESS
            }
            Err(e) => e,
        }
    }
}

struct Nvrtc(Option<LibNvrtc>);

unsafe impl Send for Nvrtc {}

impl Nvrtc {
    pub fn load() -> Self {
        Nvrtc(unsafe { Self::load_library() }.ok())
    }

    unsafe fn load_library() -> Result<LibNvrtc, libloading::Error> {
        LibNvrtc::new(env::var("ZLUDA_NVRTC_LIB").unwrap_or("nvrtc_cuda.dll".into()))
    }

    fn get(&self) -> Result<&LibNvrtc, nvrtcResult> {
        if let Some(nvrtc) = &self.0 {
            Ok(nvrtc)
        } else {
            Err(nvrtcResult::NVRTC_ERROR_INTERNAL_ERROR)
        }
    }

    pub fn get_error_string(&self, result: nvrtcResult) -> *const c_char {
        if let Ok(nvrtc) = self.get() {
            unsafe { nvrtc.nvrtcGetErrorString(result) }
        } else {
            ptr::null()
        }
    }

    pub fn create_program(
        &self,
        src: *const c_char,
        name: *const c_char,
        num_headers: i32,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> Result<nvrtcProgram, nvrtcResult> {
        let mut prog = ptr::null_mut();
        call!(self.get()?.nvrtcCreateProgram(
            &mut prog,
            src,
            name,
            num_headers,
            headers,
            include_names
        ));
        Ok(prog)
    }

    pub fn destroy_program(&self, prog: *mut nvrtcProgram) -> Result<(), nvrtcResult> {
        call!(self.get()?.nvrtcDestroyProgram(prog));
        Ok(())
    }

    pub fn compile_program(
        &self,
        prog: nvrtcProgram,
        num_options: i32,
        options_: *const *const c_char,
    ) -> Result<(), nvrtcResult> {
        let mut options = Vec::<*const c_char>::new();
        for i in 0..num_options {
            let option =
                unsafe { std::ffi::CStr::from_ptr(*options_.add(i as _)) }.to_string_lossy();
            if option.starts_with("--gpu-architecture") {
                options.push(b"--gpu-architecture=sm_86\0".as_ptr() as _);
                continue;
            }
            options.push(option.as_ptr() as _);
        }

        let nvrtc = self.get()?;
        call!(nvrtc.nvrtcCompileProgram(prog, options.len() as _, options.as_ptr()));

        Ok(())
    }

    pub fn get_ptx_size(&self, prog: nvrtcProgram) -> Result<usize, nvrtcResult> {
        let nvrtc = self.get()?;
        let mut size = 0;
        call!(nvrtc.nvrtcGetPTXSize(prog, &mut size));
        Ok(size)
    }

    pub fn get_ptx(&self, prog: nvrtcProgram, code: *mut c_char) -> Result<(), nvrtcResult> {
        let nvrtc = self.get()?;
        call!(nvrtc.nvrtcGetPTX(prog, code));
        Ok(())
    }

    pub fn get_program_log_size(&self, prog: nvrtcProgram) -> Result<usize, nvrtcResult> {
        let nvrtc = self.get()?;
        let mut size = 0;
        call!(nvrtc.nvrtcGetProgramLogSize(prog, &mut size));
        Ok(size)
    }

    pub fn get_program_log(&self, prog: nvrtcProgram, log: *mut c_char) -> Result<(), nvrtcResult> {
        let nvrtc = self.get()?;
        call!(nvrtc.nvrtcGetProgramLog(prog, log));
        Ok(())
    }
}

#[cfg(debug_assertions)]
fn unsupported() -> nvrtcResult {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
fn unsupported() -> nvrtcResult {
    nvrtcResult::NVRTC_ERROR_INTERNAL_ERROR
}

const NVRTC_VERSION_MAJOR: i32 = 12;
const NVRTC_VERSION_MINOR: i32 = 2;

fn get_error_string(result: nvrtcResult) -> *const c_char {
    let nvrtc_mutex = &*NVRTC;
    let nvrtc = &*nvrtc_mutex.lock().unwrap();
    nvrtc.get_error_string(result)
}

unsafe fn version(major: *mut i32, minor: *mut i32) -> nvrtcResult {
    *major = NVRTC_VERSION_MAJOR;
    *minor = NVRTC_VERSION_MINOR;
    nvrtcResult::NVRTC_SUCCESS
}

fn create_program(
    prog: *mut nvrtcProgram,
    src: *const c_char,
    name: *const c_char,
    num_headers: i32,
    headers: *const *const c_char,
    include_names: *const *const c_char,
) -> nvrtcResult {
    let nvrtc_mutex = &*NVRTC;
    let nvrtc = &*nvrtc_mutex.lock().unwrap();
    nvrtc
        .create_program(src, name, num_headers, headers, include_names)
        .then(|program| unsafe {
            *prog = program;
        })
}

fn destroy_program(prog: *mut nvrtcProgram) -> nvrtcResult {
    let nvrtc_mutex = &*NVRTC;
    let nvrtc = &*nvrtc_mutex.lock().unwrap();
    nvrtc.destroy_program(prog).into()
}

fn compile_program(
    prog: nvrtcProgram,
    num_options: i32,
    options: *const *const c_char,
) -> nvrtcResult {
    let nvrtc_mutex = &*NVRTC;
    let nvrtc = &*nvrtc_mutex.lock().unwrap();
    nvrtc.compile_program(prog, num_options, options).into()
}

unsafe fn get_ptx_size(prog: nvrtcProgram, code_size_ret: *mut usize) -> nvrtcResult {
    let nvrtc_mutex = &*NVRTC;
    let nvrtc = &*nvrtc_mutex.lock().unwrap();
    nvrtc.get_ptx_size(prog).then(|size| unsafe {
        *code_size_ret = size;
    })
}

unsafe fn get_ptx(prog: nvrtcProgram, code: *mut c_char) -> nvrtcResult {
    let nvrtc_mutex = &*NVRTC;
    let nvrtc = &*nvrtc_mutex.lock().unwrap();
    nvrtc.get_ptx(prog, code).into()
}

unsafe fn get_program_log_size(prog: nvrtcProgram, log_size_ret: *mut usize) -> nvrtcResult {
    let nvrtc_mutex = &*NVRTC;
    let nvrtc = &*nvrtc_mutex.lock().unwrap();
    nvrtc.get_program_log_size(prog).then(|size| unsafe {
        *log_size_ret = size;
    })
}

unsafe fn get_program_log(prog: nvrtcProgram, log: *mut c_char) -> nvrtcResult {
    let nvrtc_mutex = &*NVRTC;
    let nvrtc = &*nvrtc_mutex.lock().unwrap();
    nvrtc.get_program_log(prog, log).into()
}

unsafe fn get_cubin_size(prog: nvrtcProgram, cubin_size_ret: *mut usize) -> nvrtcResult {
    get_ptx_size(prog, cubin_size_ret)
}

unsafe fn get_cubin(prog: nvrtcProgram, cubin: *mut c_char) -> nvrtcResult {
    /* We return PTX code instead of ELF binary here
    because it may be passed to cuModuleLoadData,
    then it can be treated as CUmoduleContent::RawText.
    So nvcuda.dll will translate PTX and compile LLVM-IR. */
    get_ptx(prog, cubin)
}
