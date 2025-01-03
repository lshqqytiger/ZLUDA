#![allow(warnings)]
mod cublaslt;
mod trap;
use std::{alloc, ptr};

pub use cublaslt::*;

use hipblaslt_sys::*;

#[cfg(debug_assertions)]
pub(crate) fn unsupported() -> cublasStatus_t {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
pub(crate) fn unsupported() -> cublasStatus_t {
    cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
}

// Not in the headers, but exported by library and used (by cuBLAS)
// These traps allow us to load the original cuBLAS library
// and ZLUDA simultaneously.
install_trap!(cublasLtLegacyGemmUtilizationDDD);
install_trap!(cublasLtLegacyGemmUtilizationCCC);
install_trap!(cublasLtLegacyGemmUtilizationZZZ);
install_trap!(cublasLtLegacyGemmDDD);
install_trap!(cublasLtLegacyGemmCCC);
install_trap!(cublasLtLegacyGemmZZZ);
install_trap!(cublasLtLegacyGemmTST);
install_trap!(cublasLtLegacyGemmTSS);
install_trap!(cublasLtLegacyGemmSSS);
install_trap!(cublasLtLegacyGemmHSS);
install_trap!(cublasLtLegacyGemmHSH);
install_trap!(cublasLtLegacyGemmHHH);
install_trap!(cublasLtLegacyGemmBSS);
install_trap!(cublasLtLegacyGemmBII);
install_trap!(cublasLtLegacyGemmACC);
install_trap!(cublasLt_for_cublas_TST);
install_trap!(cublasLt_for_cublas_TSS);
install_trap!(cublasLt_for_cublas_ZZZ);
install_trap!(cublasLt_for_cublas_HSS);
install_trap!(cublasLt_for_cublas_HSH);
install_trap!(cublasLt_for_cublas_BSS);
install_trap!(cublasLt_for_cublas_BII);
install_trap!(cublasLt_for_cublas_CCC);
install_trap!(cublasLt_for_cublas_SSS);
install_trap!(cublasLt_for_cublas_HHH);
install_trap!(cublasLt_for_cublas_DDD);
install_trap!(cublasLtZZZMatmulAlgoInit);
install_trap!(cublasLtZZZMatmulAlgoGetHeuristic);
install_trap!(cublasLtZZZMatmul);
install_trap!(cublasLtTSTMatmulAlgoInit);
install_trap!(cublasLtTSTMatmulAlgoGetHeuristic);
install_trap!(cublasLtTSTMatmul);
install_trap!(cublasLtTSSMatmulAlgoInit);
install_trap!(cublasLtTSSMatmulAlgoGetHeuristic);
install_trap!(cublasLtTSSMatmul);
install_trap!(cublasLtSSSMatmulAlgoInit);
install_trap!(cublasLtSSSMatmulAlgoGetHeuristic);
install_trap!(cublasLtSSSMatmul);
install_trap!(cublasLtHSSMatmulAlgoInit);
install_trap!(cublasLtHSSMatmulAlgoGetHeuristic);
install_trap!(cublasLtHSSMatmul);
install_trap!(cublasLtHSHMatmulAlgoInit);
install_trap!(cublasLtHSHMatmulAlgoGetHeuristic);
install_trap!(cublasLtHSHMatmul);
install_trap!(cublasLtHHHMatmulAlgoInit);
install_trap!(cublasLtHHHMatmulAlgoGetHeuristic);
install_trap!(cublasLtHHHMatmul);
install_trap!(cublasLtDDDMatmulAlgoInit);
install_trap!(cublasLtDDDMatmulAlgoGetHeuristic);
install_trap!(cublasLtDDDMatmul);
install_trap!(cublasLtCCCMatmulAlgoInit);
install_trap!(cublasLtCCCMatmulAlgoGetHeuristic);
install_trap!(cublasLtCCCMatmul);
install_trap!(cublasLtBIIMatmulAlgoInit);
install_trap!(cublasLtBIIMatmulAlgoGetHeuristic);
install_trap!(cublasLtBIIMatmul);
install_trap!(cublasLtBSSMatmulAlgoInit);
install_trap!(cublasLtBSSMatmulAlgoGetHeuristic);
install_trap!(cublasLtBSSMatmul);
install_trap!(cublasLtACCMatmulAlgoInit);
install_trap!(cublasLtACCMatmulAlgoGetHeuristic);
install_trap!(cublasLtACCMatmul);
install_trap!(cublasLtCtxInit);
install_trap!(cublasLtShutdownCtx);

unsafe fn create(handle: *mut cublasLtHandle_t) -> cublasStatus_t {
    to_cuda(hipblasLtCreate(handle.cast()))
}

fn to_cuda(result: hipblasStatus_t) -> cublasStatus_t {
    match result {
        hipblasStatus_t::HIPBLAS_STATUS_SUCCESS => cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        hipblasStatus_t::HIPBLAS_STATUS_INVALID_VALUE => {
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE
        }
        _ => panic!("[ZLUDA] hipBLASLt failed: {}", result.0),
    }
}

fn get_version() -> usize {
    111103
}

unsafe fn matmul(
    light_handle: *mut cublasLtContext,
    compute_desc: *mut cublasLtMatmulDescOpaque_t,
    alpha: *const std::ffi::c_void,
    a: *const std::ffi::c_void,
    adesc: *mut cublasLtMatrixLayoutOpaque_t,
    b: *const std::ffi::c_void,
    bdesc: *mut cublasLtMatrixLayoutOpaque_t,
    beta: *const std::ffi::c_void,
    c: *const std::ffi::c_void,
    cdesc: *mut cublasLtMatrixLayoutOpaque_t,
    d: *mut std::ffi::c_void,
    ddesc: *mut cublasLtMatrixLayoutOpaque_t,
    algo: *const cublasLtMatmulAlgo_t,
    workspace: *mut std::ffi::c_void,
    workspace_size_in_bytes: usize,
    stream: *mut CUstream_st,
) -> cublasStatus_t {
    let stream = to_stream(stream);
    to_cuda(hipblasLtMatmul(
        light_handle.cast(),
        compute_desc.cast(),
        alpha,
        a,
        adesc.cast(),
        b,
        bdesc.cast(),
        beta,
        c,
        cdesc.cast(),
        d,
        ddesc.cast(),
        algo.cast(),
        workspace,
        workspace_size_in_bytes,
        stream,
    ))
}

unsafe fn to_stream(stream: cudaStream_t) -> hipStream_t {
    use cuda_types::*;
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
    let maybe_hip_stream: Result<_, _> = zluda_ext.get_hip_stream(stream as _).into();
    maybe_hip_stream.unwrap() as _
}

unsafe fn matmul_algo_get_heuristic(
    light_handle: *mut cublasLtContext,
    operation_desc: *mut cublasLtMatmulDescOpaque_t,
    adesc: *mut cublasLtMatrixLayoutOpaque_t,
    bdesc: *mut cublasLtMatrixLayoutOpaque_t,
    cdesc: *mut cublasLtMatrixLayoutOpaque_t,
    ddesc: *mut cublasLtMatrixLayoutOpaque_t,
    preference: *mut cublasLtMatmulPreferenceOpaque_t,
    requested_algo_count: i32,
    heuristic_results_array: *mut cublasLtMatmulHeuristicResult_t,
    return_algo_count: *mut i32,
) -> cublasStatus_t {
    to_cuda(hipblasLtMatmulAlgoGetHeuristic(
        light_handle.cast(),
        operation_desc.cast(),
        adesc.cast(),
        bdesc.cast(),
        cdesc.cast(),
        ddesc.cast(),
        preference.cast(),
        requested_algo_count,
        heuristic_results_array.cast(),
        return_algo_count,
    ))
}

unsafe fn matmul_desc_create(
    matmul_desc: *mut *mut cublasLtMatmulDescOpaque_t,
    compute_type: cublasComputeType_t,
    scale_type: cudaDataType_t,
) -> cublasStatus_t {
    if compute_type != cublasComputeType_t::CUBLAS_COMPUTE_32F {
        return cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED;
    }
    let scale_type = data_type(scale_type);
    to_cuda(hipblasLtMatmulDescCreate(
        matmul_desc.cast(),
        hipblasComputeType_t::HIPBLAS_COMPUTE_32F,
        scale_type,
    ))
}

fn data_type(data_type: cudaDataType_t) -> hipDataType {
    match data_type {
        cudaDataType_t::CUDA_R_32F => hipDataType::HIP_R_32F,
        cudaDataType_t::CUDA_R_64F => hipDataType::HIP_R_64F,
        cudaDataType_t::CUDA_R_16F => hipDataType::HIP_R_16F,
        cudaDataType_t::CUDA_R_8I => hipDataType::HIP_R_8I,
        cudaDataType_t::CUDA_R_8U => hipDataType::HIP_R_8U,
        cudaDataType_t::CUDA_R_32I => hipDataType::HIP_R_32I,
        cudaDataType_t::CUDA_R_32U => hipDataType::HIP_R_32U,
        cudaDataType_t::CUDA_R_16BF => hipDataType::HIP_R_16BF,
        cudaDataType_t::CUDA_R_8F_E4M3 => hipDataType::HIP_R_8F_E4M3_FNUZ,
        cudaDataType_t::CUDA_R_8F_E5M2 => hipDataType::HIP_R_8F_E5M2_FNUZ,
        _ => panic!(),
    }
}

struct VoidPointer {
    layout: Option<alloc::Layout>,
    base: *mut u8,
    droppable: bool,
}

impl VoidPointer {
    fn new<T>(v: T) -> Self {
        let layout = alloc::Layout::new::<T>();
        let base = unsafe {
            let ptr = alloc::alloc(layout);
            *(ptr as *mut T) = v;
            ptr
        };
        VoidPointer {
            layout: Some(layout),
            base,
            droppable: true,
        }
    }

    fn from_raw(raw: *const std::ffi::c_void) -> Self {
        VoidPointer {
            layout: None,
            base: raw as *mut u8,
            droppable: false,
        }
    }

    fn as_raw(&self) -> *const std::ffi::c_void {
        self.base.cast()
    }

    fn try_drop(self) -> bool {
        if !self.droppable {
            return false;
        }
        unsafe {
            std::alloc::dealloc(self.base, self.layout.unwrap());
        }
        true
    }
}

unsafe fn matmul_desc_set_attribute(
    matmul_desc: *mut cublasLtMatmulDescOpaque_t,
    attr: cublasLtMatmulDescAttributes_t,
    buf: *const std::ffi::c_void,
    size_in_bytes: usize,
) -> cublasStatus_t {
    let attr = to_attrib(attr);
    let buf = transform_attrib(attr, buf);
    let result = to_cuda(hipblasLtMatmulDescSetAttribute(
        matmul_desc.cast(),
        attr,
        buf.as_raw(),
        size_in_bytes,
    ));
    buf.try_drop();
    result
}

fn to_attrib(attr: cublasLtMatmulDescAttributes_t) -> hipblasLtMatmulDescAttributes_t {
    match attr {
        cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA => {
            hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_TRANSA
        }
        cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB => {
            hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_TRANSB
        }
        cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE => {
            hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_EPILOGUE
        }
        cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER => {
            hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_BIAS_POINTER
        }
        cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE => {
            hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE
        }
        cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_D_SCALE_POINTER => {
            hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER
        }
        _ => panic!(),
    }
}

unsafe fn transform_attrib(
    attr: hipblasLtMatmulDescAttributes_t,
    buf: *const std::ffi::c_void,
) -> VoidPointer {
    match attr {
        hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_TRANSA
        | hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_TRANSB => {
            VoidPointer::new(to_operation(*(buf as *const cublasOperation_t)))
        }
        hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_EPILOGUE
        | hipblasLtMatmulDescAttributes_t::HIPBLASLT_MATMUL_DESC_BIAS_POINTER => {
            VoidPointer::from_raw(buf)
        }
        _ => panic!("[ZLUDA] Don't know how to transform attribute({}).", attr.0),
    }
}

fn to_operation(operation: cublasOperation_t) -> hipblasOperation_t {
    match operation {
        cublasOperation_t::CUBLAS_OP_N => hipblasOperation_t::HIPBLAS_OP_N,
        cublasOperation_t::CUBLAS_OP_T => hipblasOperation_t::HIPBLAS_OP_T,
        cublasOperation_t::CUBLAS_OP_C => hipblasOperation_t::HIPBLAS_OP_C,
        _ => panic!(),
    }
}

unsafe fn matrix_layout_create(
    mat_layout: *mut *mut cublasLtMatrixLayoutOpaque_t,
    type_: cudaDataType_t,
    rows: u64,
    cols: u64,
    ld: i64,
) -> cublasStatus_t {
    let type_ = data_type(type_);
    to_cuda(hipblasLtMatrixLayoutCreate(
        mat_layout.cast(),
        type_,
        rows,
        cols,
        ld,
    ))
}

unsafe fn matmul_desc_destroy(matmul_desc: *mut cublasLtMatmulDescOpaque_t) -> cublasStatus_t {
    to_cuda(hipblasLtMatmulDescDestroy(matmul_desc.cast()))
}

unsafe fn matmul_desc_get_attribute(
    matmul_desc: *mut cublasLtMatmulDescOpaque_t,
    attr: cublasLtMatmulDescAttributes_t,
    buf: *mut std::ffi::c_void,
    size_in_bytes: usize,
    size_written: *mut usize,
) -> cublasStatus_t {
    let attr = to_attrib(attr);
    to_cuda(hipblasLtMatmulDescGetAttribute(
        matmul_desc.cast(),
        attr,
        buf,
        size_in_bytes,
        size_written,
    ))
}

unsafe fn matmul_preference_create(
    pref: *mut *mut cublasLtMatmulPreferenceOpaque_t,
) -> cublasStatus_t {
    to_cuda(hipblasLtMatmulPreferenceCreate(pref.cast()))
}

unsafe fn matmul_preference_destroy(pref: *mut cublasLtMatmulPreferenceOpaque_t) -> cublasStatus_t {
    to_cuda(hipblasLtMatmulPreferenceDestroy(pref.cast()))
}

unsafe fn matmul_preference_set_attribute(
    pref: *mut cublasLtMatmulPreferenceOpaque_t,
    attr: cublasLtMatmulPreferenceAttributes_t,
    buf: *const std::ffi::c_void,
    size_in_bytes: usize,
) -> cublasStatus_t {
    if matches!(
        attr,
        cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES
            | cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES
            | cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES
            | cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES
    ) {
        return cublasStatus_t::CUBLAS_STATUS_SUCCESS;
    }
    let attr = to_preference_attrib(attr);
    to_cuda(hipblasLtMatmulPreferenceSetAttribute(
        pref.cast(),
        attr,
        buf,
        size_in_bytes,
    ))
}

fn to_preference_attrib(
    attr: cublasLtMatmulPreferenceAttributes_t,
) -> hipblasLtMatmulPreferenceAttributes_t {
    match attr {
        cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_SEARCH_MODE => {
            hipblasLtMatmulPreferenceAttributes_t::HIPBLASLT_MATMUL_PREF_SEARCH_MODE
        }
        cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES => {
            hipblasLtMatmulPreferenceAttributes_t::HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
        }
        _ => panic!("{}", attr.0),
    }
}

unsafe fn matrix_layout_destroy(mat_layout: *mut cublasLtMatrixLayoutOpaque_t) -> cublasStatus_t {
    to_cuda(hipblasLtMatrixLayoutDestroy(mat_layout.cast()))
}

unsafe fn matrix_layout_set_attribute(
    mat_layout: *mut cublasLtMatrixLayoutOpaque_t,
    attr: cublasLtMatrixLayoutAttribute_t,
    buf: *const std::ffi::c_void,
    size_in_bytes: usize,
) -> cublasStatus_t {
    let attr = to_matrix_attrib(attr);
    to_cuda(hipblasLtMatrixLayoutSetAttribute(
        mat_layout.cast(),
        attr,
        buf,
        size_in_bytes,
    ))
}

fn to_matrix_attrib(attr: cublasLtMatrixLayoutAttribute_t) -> hipblasLtMatrixLayoutAttribute_t {
    match attr {
        cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT => {
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT
        }
        cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET => {
            hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET
        }
        _ => panic!(),
    }
}
