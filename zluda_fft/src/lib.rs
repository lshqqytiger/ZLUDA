#[allow(warnings)]
mod cufft;
pub use cufft::*;

#[allow(warnings)]
mod cufftxt;
pub use cufftxt::*;

use cuda_types::*;
use hipfft_sys::*;
use lazy_static::lazy_static;
use slab::Slab;
use std::{mem, ptr, sync::Mutex};

#[cfg(debug_assertions)]
pub(crate) fn unsupported() -> cufftResult {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
pub(crate) fn unsupported() -> cufftResult {
    cufftResult::CUFFT_NOT_SUPPORTED
}

#[no_mangle]
pub extern "system" fn cufftLeaveCS() {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn cufftEnterCS() {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn cufftMakePlanGuru64() {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn cufftXtMakePlanGuru64() {
    unimplemented!()
}

lazy_static! {
    static ref PLANS: Mutex<Slab<Plan>> = Mutex::new(Slab::new());
}

struct Plan {
    handle: hipfftHandle,
    xt_typ: Option<cufftType_t>,
}
unsafe impl Send for Plan {}

unsafe fn create(handle: *mut cufftHandle) -> cufftResult {
    let mut hip_handle = unsafe { mem::zeroed() };
    let error = hipfftCreate(&mut hip_handle);
    if error != hipfftResult::HIPFFT_SUCCESS {
        return cufftResult::CUFFT_INTERNAL_ERROR;
    }
    let plan_key = {
        let mut plans = PLANS.lock().unwrap();
        plans.insert(Plan {
            handle: hip_handle,
            xt_typ: None,
        })
    };
    *handle = plan_key as i32;
    cufftResult::CUFFT_SUCCESS
}

fn destroy(plan: i32) -> cufftResult_t {
    let mut plans = PLANS.lock().unwrap();
    plans.remove(plan as usize);
    cufftResult::CUFFT_SUCCESS
}

unsafe fn make_plan_many_64(
    plan: i32,
    rank: i32,
    n: *mut i64,
    inembed: *mut i64,
    istride: i64,
    idist: i64,
    onembed: *mut i64,
    ostride: i64,
    odist: i64,
    type_: cufftType,
    batch: i64,
    work_size: *mut usize,
) -> cufftResult_t {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let type_ = cuda_type(type_);
    to_cuda(hipfftMakePlanMany64(
        hip_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch,
        work_size,
    ))
}

fn cuda_type(type_: cufftType) -> hipfftType_t {
    match type_ {
        cufftType::CUFFT_R2C => hipfftType_t::HIPFFT_R2C,
        cufftType::CUFFT_C2R => hipfftType_t::HIPFFT_C2R,
        cufftType::CUFFT_C2C => hipfftType_t::HIPFFT_C2C,
        cufftType::CUFFT_D2Z => hipfftType_t::HIPFFT_D2Z,
        cufftType::CUFFT_Z2D => hipfftType_t::HIPFFT_Z2D,
        cufftType::CUFFT_Z2Z => hipfftType_t::HIPFFT_Z2Z,
        _ => panic!(),
    }
}

fn xt_type(input: cudaDataType, output: cudaDataType) -> cufftType {
    match (input, output) {
        (cudaDataType::CUDA_R_32F, cudaDataType::CUDA_C_32F) => cufftType::CUFFT_R2C,
        (cudaDataType::CUDA_C_32F, cudaDataType::CUDA_R_32F) => cufftType::CUFFT_C2R,
        (cudaDataType::CUDA_C_32F, cudaDataType::CUDA_C_32F) => cufftType::CUFFT_C2C,
        _ => panic!(
            "[ZLUDA] Unknown type combination: ({}, {})",
            input.0, output.0
        ),
    }
}

fn get_hip_plan(plan: cufftHandle) -> Result<hipfftHandle, cufftResult_t> {
    let plans = PLANS.lock().unwrap();
    plans
        .get(plan as usize)
        .map(|p| p.handle)
        .ok_or(cufftResult_t::CUFFT_INVALID_PLAN)
}

fn to_cuda(result: hipfftResult) -> cufftResult_t {
    match result {
        hipfftResult::HIPFFT_SUCCESS => cufftResult_t::CUFFT_SUCCESS,
        _ => cufftResult_t::CUFFT_INTERNAL_ERROR,
    }
}

unsafe fn make_plan_many(
    plan: i32,
    rank: i32,
    n: *mut i32,
    inembed: *mut i32,
    istride: i32,
    idist: i32,
    onembed: *mut i32,
    ostride: i32,
    odist: i32,
    type_: cufftType,
    batch: i32,
    work_size: *mut usize,
) -> cufftResult_t {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let type_ = cuda_type(type_);
    to_cuda(hipfftMakePlanMany(
        hip_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch,
        work_size,
    ))
}

unsafe fn plan_many(
    plan: *mut i32,
    rank: i32,
    n: *mut i32,
    inembed: *mut i32,
    istride: i32,
    idist: i32,
    onembed: *mut i32,
    ostride: i32,
    odist: i32,
    type_: cufftType,
    batch: i32,
) -> cufftResult_t {
    let type_ = cuda_type(type_);
    let mut hip_plan = mem::zeroed();
    let result = to_cuda(hipfftPlanMany(
        &mut hip_plan,
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type_,
        batch,
    ));
    if result != cufftResult_t::CUFFT_SUCCESS {
        return result;
    }
    let plan_key = {
        let mut plans = PLANS.lock().unwrap();
        plans.insert(Plan {
            handle: hip_plan,
            xt_typ: None,
        })
    };
    *plan = plan_key as i32;
    result
}

unsafe fn set_stream(plan: i32, stream: *mut cufft::CUstream_st) -> cufftResult_t {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let lib = hip_common::zluda_ext::get_cuda_library().unwrap();
    let cu_get_export_table = lib
        .get::<unsafe extern "C" fn(
            ppExportTable: *mut *const ::std::os::raw::c_void,
            pExportTableId: *const CUuuid,
        ) -> CUresult>(b"cuGetExportTable\0")
        .unwrap();
    let mut export_table = ptr::null();
    let error = (cu_get_export_table)(&mut export_table, &zluda_dark_api::ZludaExt::GUID);
    assert_eq!(error, CUresult::CUDA_SUCCESS);
    let zluda_ext = zluda_dark_api::ZludaExt::new(export_table);
    let stream: Result<_, _> = zluda_ext.get_hip_stream(stream as _).into();
    to_cuda(hipfftSetStream(hip_plan, stream.unwrap() as _))
}

unsafe fn set_work_area(plan: i32, work_area: *mut ::std::os::raw::c_void) -> cufftResult_t {
    let plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftSetWorkArea(plan, work_area))
}

unsafe fn set_auto_allocation(plan: i32, auto_allocate: i32) -> cufftResult_t {
    let plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftSetAutoAllocation(plan, auto_allocate))
}

unsafe fn exec_c2c(
    plan: i32,
    idata: *mut cufft::float2,
    odata: *mut cufft::float2,
    direction: i32,
) -> cufftResult_t {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftExecC2C(
        hip_plan,
        idata.cast(),
        odata.cast(),
        direction,
    ))
}

unsafe fn exec_z2z(
    plan: i32,
    idata: *mut cufft::double2,
    odata: *mut cufft::double2,
    direction: i32,
) -> cufftResult {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftExecZ2Z(
        hip_plan,
        idata.cast(),
        odata.cast(),
        direction,
    ))
}

unsafe fn exec_r2c(plan: i32, idata: *mut f32, odata: *mut cufft::float2) -> cufftResult {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftExecR2C(hip_plan, idata, odata.cast()))
}

unsafe fn exec_c2r(plan: i32, idata: *mut cufft::float2, odata: *mut f32) -> cufftResult {
    let hip_plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    to_cuda(hipfftExecC2R(hip_plan, idata.cast(), odata))
}

unsafe fn plan_3d(plan: *mut i32, nx: i32, ny: i32, nz: i32, type_: cufftType) -> cufftResult {
    let type_ = cuda_type(type_);
    let mut hip_plan = mem::zeroed();
    let result = to_cuda(hipfftPlan3d(&mut hip_plan, nx, ny, nz, type_));
    if result != cufftResult_t::CUFFT_SUCCESS {
        return result;
    }
    let plan_key = {
        let mut plans = PLANS.lock().unwrap();
        plans.insert(Plan {
            handle: hip_plan,
            xt_typ: None,
        })
    };
    *plan = plan_key as i32;
    result
}

fn set_xt_type(plan: cufftHandle, typ: cufftType_t) -> Result<(), cufftResult_t> {
    let mut plans = PLANS.lock().unwrap();
    plans
        .get_mut(plan as usize)
        .map(|x| {
            x.xt_typ = Some(typ);
        })
        .ok_or(cufftResult_t::CUFFT_INVALID_PLAN)
}

fn get_xt_type(plan: cufftHandle) -> Result<cufftType_t, cufftResult_t> {
    let plans = PLANS.lock().unwrap();
    plans
        .get(plan as usize)
        .map(|x| x.xt_typ)
        .flatten()
        .ok_or(cufftResult_t::CUFFT_INVALID_PLAN)
}

unsafe fn xt_make_plan_many(
    plan: i32,
    rank: i32,
    n: *mut i64,
    inembed: *mut i64,
    istride: i64,
    idist: i64,
    inputtype: cudaDataType,
    onembed: *mut i64,
    ostride: i64,
    odist: i64,
    outputtype: cudaDataType,
    batch: i64,
    work_size: *mut usize,
    _executiontype: cudaDataType,
) -> cufftResult_t {
    let typ = xt_type(inputtype, outputtype);
    if let Err(result) = set_xt_type(plan, typ) {
        return result;
    }
    let plan = match get_hip_plan(plan) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let typ = cuda_type(typ);
    to_cuda(hipfftMakePlanMany64(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, typ, batch, work_size,
    ))
}

unsafe fn xt_exec(
    plan: i32,
    input: *mut ::std::os::raw::c_void,
    output: *mut ::std::os::raw::c_void,
    direction: i32,
) -> cufftResult_t {
    let typ = match get_xt_type(plan) {
        Ok(t) => t,
        Err(e) => return e,
    };
    match typ {
        cufftType_t::CUFFT_R2C => exec_r2c(plan, input.cast(), output.cast()),
        cufftType_t::CUFFT_C2R => exec_c2r(plan, input.cast(), output.cast()),
        cufftType_t::CUFFT_C2C => exec_c2c(plan, input.cast(), output.cast(), direction),
        cufftType_t::CUFFT_Z2Z => exec_z2z(plan, input.cast(), output.cast(), direction),
        _ => unimplemented!(),
    }
}
