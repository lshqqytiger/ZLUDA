use super::hipfix;
use crate::hip_call_cuda;
use cuda_types::*;
use hip_runtime_sys::*;
use lazy_static::lazy_static;
use rustc_hash::FxHashSet;
use std::{ptr, sync::Mutex};

#[repr(C)]
#[derive(PartialEq, Eq, Hash)]
pub(crate) struct TextureObject(hipTextureObject_t);

unsafe impl Send for TextureObject {}

lazy_static! {
    static ref POOL: Mutex<FxHashSet<TextureObject>> = Mutex::new(FxHashSet::default());
}

pub(crate) unsafe fn create(
    p_tex_object: *mut hipTextureObject_t,
    p_res_desc: *const CUDA_RESOURCE_DESC,
    p_tex_desc: *const HIP_TEXTURE_DESC,
    p_res_view_desc: *const HIP_RESOURCE_VIEW_DESC,
) -> Result<(), CUresult> {
    if p_res_desc == ptr::null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let result = hipfix::array::with_resource_desc(
        p_res_desc,
        p_res_view_desc,
        |p_res_desc, p_res_view_desc| {
            hip_call_cuda!(hipTexObjectCreate(
                p_tex_object,
                p_res_desc,
                p_tex_desc,
                p_res_view_desc
            ));
            Ok(())
        },
    )?;
    let pool = &mut *POOL.lock().unwrap();
    pool.insert(TextureObject(*p_tex_object));
    result
}

// cuTexObjectDestroy() returns CUDA_SUCCESS when it is called
// multiple times with the same texture object while calling
// hipTexObjectDestroy() with invalid texture object is not allowed
// and leads to double free.
pub(crate) unsafe fn destroy(tex_object: hipTextureObject_t) -> Result<(), CUresult> {
    let pool = &mut *POOL.lock().unwrap();
    let tex_object = TextureObject(tex_object);
    if pool.contains(&tex_object) {
        pool.remove(&tex_object);
    }
    Ok(())
}
