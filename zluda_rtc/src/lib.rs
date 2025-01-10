#![allow(warnings)]
mod nvrtc;
use std::{env, ffi::c_char, ptr, result, sync::Mutex};

use lazy_static::lazy_static;
pub use nvrtc::*;

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
        options: *const *const c_char,
    ) -> Result<Box<[c_char]>, nvrtcResult> {
        let nvrtc = self.get()?;
        call!(nvrtc.nvrtcCompileProgram(prog, num_options, options));
        let mut size = 0;
        call!(nvrtc.nvrtcGetPTXSize(prog, &mut size));
        let mut ptx = {
            let ptx = Box::<[c_char]>::new_uninit_slice(size);
            unsafe { ptx.assume_init() }
        };
        call!(nvrtc.nvrtcGetPTX(prog, ptx.as_mut_ptr()));
        Ok(ptx)
    }

    pub fn get_program_log_size(&self, prog: nvrtcProgram) -> Result<usize, nvrtcResult> {
        let nvrtc = self.get()?;
        let mut log_size_ret = 0;
        call!(nvrtc.nvrtcGetProgramLogSize(prog, &mut log_size_ret));
        Ok(log_size_ret)
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

#[repr(C)]
struct Program {
    base: nvrtcProgram,
    ptx: Option<Box<[c_char]>>,
}

impl Program {
    fn new(base: nvrtcProgram) -> Self {
        Program { base, ptx: None }
    }

    unsafe fn from<'a>(ptr: nvrtcProgram) -> Option<&'a mut Program> {
        (ptr as *mut Program).as_mut()
    }

    fn set_ptx(&mut self, ptx: Box<[c_char]>) {
        self.ptx = Some(ptx);
    }
}

trait IntoBox {
    unsafe fn into_box(self) -> Box<Program>;
}

impl IntoBox for *mut nvrtcProgram {
    unsafe fn into_box(self) -> Box<Program> {
        Box::from_raw(*(self as *mut *mut Program))
    }
}

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
        .then(|program| {
            let program = Box::into_raw(Box::new(Program::new(program)));
            unsafe {
                *(prog as *mut *mut Program) = program;
            }
        })
}

fn destroy_program(prog: *mut nvrtcProgram) -> nvrtcResult {
    let nvrtc_mutex = &*NVRTC;
    let nvrtc = &*nvrtc_mutex.lock().unwrap();

    let mut prog = unsafe { prog.into_box() };
    let result = nvrtc.destroy_program(&mut prog.base).into();
    drop(prog);
    result
}

fn compile_program(
    prog: nvrtcProgram,
    num_options: i32,
    options: *const *const c_char,
) -> nvrtcResult {
    let nvrtc_mutex = &*NVRTC;
    let nvrtc = &*nvrtc_mutex.lock().unwrap();

    let prog = unsafe { Program::from(prog) };
    if prog.is_none() {
        return nvrtcResult::NVRTC_ERROR_INVALID_PROGRAM;
    }
    let prog = prog.unwrap();

    nvrtc
        .compile_program(prog.base, num_options, options)
        .then(|ptx| {
            prog.set_ptx(ptx);
        })
}

unsafe fn get_ptx_size(prog: nvrtcProgram, code_size_ret: *mut usize) -> nvrtcResult {
    let prog = Program::from(prog);
    if let Some(prog) = prog {
        if let Some(ptx) = &prog.ptx {
            *code_size_ret = ptx.len();
            return nvrtcResult::NVRTC_SUCCESS;
        }
    }
    nvrtcResult::NVRTC_ERROR_INVALID_PROGRAM
}

unsafe fn get_ptx(prog: nvrtcProgram, code: *mut c_char) -> nvrtcResult {
    let prog = Program::from(prog);
    if let Some(prog) = prog {
        if let Some(ptx) = &prog.ptx {
            for (i, &c) in ptx.iter().enumerate() {
                *code.add(i) = c;
            }
            return nvrtcResult::NVRTC_SUCCESS;
        }
    }
    nvrtcResult::NVRTC_ERROR_INVALID_PROGRAM
}

fn get_program_log_size(prog: nvrtcProgram, log_size_ret: *mut usize) -> nvrtcResult {
    let nvrtc_mutex = &*NVRTC;
    let nvrtc = &*nvrtc_mutex.lock().unwrap();
    nvrtc.get_program_log_size(prog).then(|size| unsafe {
        *log_size_ret = size;
    })
}
