#[allow(warnings)]
mod cudnn;
#[allow(warnings)]
pub use cudnn::*;

use cuda_types::{CUresult, CUuuid};
use hip_runtime_sys::*;
use lazy_static::lazy_static;
use miopen_sys::*;
use std::{mem, ptr, sync::Mutex};

lazy_static! {
    static ref LAST_ERROR: Mutex<Option<miopenStatus_t>> = Mutex::new(None);
}

macro_rules! call {
    ($expr:expr) => {{
        let result = $expr;
        if result != miopen_sys::miopenStatus_t::miopenStatusSuccess {
            if let Ok(mut error) = LAST_ERROR.lock() {
                *error = Some(result);
            }
        }
        to_cudnn(result)
    }};
}

macro_rules! asserted_call {
    ($expr:expr) => {{
        let result = call!($expr);
        if result != cudnn::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return result;
        }
    }};
}

#[cfg(debug_assertions)]
fn unsupported() -> cudnnStatus_t {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
fn unsupported() -> cudnnStatus_t {
    if let Ok(mut error) = LAST_ERROR.lock() {
        *error = Some(miopenStatus_t::miopenStatusNotImplemented);
    }
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

fn to_cudnn(status: miopen_sys::miopenStatus_t) -> cudnnStatus_t {
    match status {
        miopen_sys::miopenStatus_t::miopenStatusSuccess => cudnnStatus_t::CUDNN_STATUS_SUCCESS,
        miopen_sys::miopenStatus_t::miopenStatusNotInitialized => {
            cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED
        }
        miopen_sys::miopenStatus_t::miopenStatusInvalidValue => {
            cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE
        }
        miopen_sys::miopenStatus_t::miopenStatusBadParm => cudnnStatus_t::CUDNN_STATUS_BAD_PARAM,
        miopen_sys::miopenStatus_t::miopenStatusNotImplemented => {
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
        }
        miopen_sys::miopenStatus_t::miopenStatusUnknownError => {
            cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR
        }
        miopen_sys::miopenStatus_t::miopenStatusUnsupportedOp => {
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
        }
        err => panic!("[ZLUDA] MIOpen failed: {}", err.0),
    }
}

unsafe fn get_error_string(status: cudnnStatus_t) -> *const ::std::os::raw::c_char {
    miopenGetErrorString(match status {
        cudnnStatus_t::CUDNN_STATUS_SUCCESS => miopen_sys::miopenStatus_t::miopenStatusSuccess,
        cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED => {
            miopen_sys::miopenStatus_t::miopenStatusNotInitialized
        }
        cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE => {
            miopen_sys::miopenStatus_t::miopenStatusInvalidValue
        }
        cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => miopen_sys::miopenStatus_t::miopenStatusBadParm,
        cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => {
            miopen_sys::miopenStatus_t::miopenStatusNotImplemented
        }
        cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR => {
            miopen_sys::miopenStatus_t::miopenStatusUnknownError
        }
        err => panic!("[ZLUDA] MIOpen failed: {}", err.0),
    })
}

unsafe fn get_last_error_string(message: *mut ::std::os::raw::c_char, max_size: usize) {
    if let Some(last_error) = LAST_ERROR.lock().ok().and_then(|x| *x) {
        let ptr = miopenGetErrorString(last_error);
        for i in 0..max_size {
            *message.add(i) = *ptr.add(i);
        }
    }
}

unsafe fn get_property(prop: libraryPropertyType, value: *mut i32) -> cudnnStatus_t {
    *value = match prop {
        libraryPropertyType_t::MAJOR_VERSION => 9,
        libraryPropertyType_t::MINOR_VERSION => 1,
        libraryPropertyType_t::PATCH_LEVEL => 0,
        _ => panic!(),
    };
    cudnnStatus_t::CUDNN_STATUS_SUCCESS
}

unsafe fn create(handle: *mut cudnnHandle_t) -> cudnnStatus_t {
    call!(miopenCreate(handle as _))
}

unsafe fn cudnn_create_tensor_descriptor(
    tensor_desc: *mut cudnnTensorDescriptor_t,
) -> cudnnStatus_t {
    call!(miopenCreateTensorDescriptor(tensor_desc as _))
}

unsafe fn cudnn_create_activation_descriptor(
    activation_desc: *mut cudnnActivationDescriptor_t,
) -> cudnnStatus_t {
    call!(miopenCreateActivationDescriptor(activation_desc as _))
}

unsafe fn cudnn_create_convolution_descriptor(
    conv_desc: *mut cudnnConvolutionDescriptor_t,
) -> cudnnStatus_t {
    call!(miopenCreateConvolutionDescriptor(conv_desc as _))
}

unsafe fn cudnn_create_filter_descriptor(
    filter_desc: *mut cudnnFilterDescriptor_t,
) -> cudnnStatus_t {
    call!(miopenCreateTensorDescriptor(filter_desc as _))
}

unsafe fn cudnn_create_lrn_descriptor(norm_desc: *mut cudnnLRNDescriptor_t) -> cudnnStatus_t {
    call!(miopenCreateLRNDescriptor(norm_desc as _))
}

unsafe fn cudnn_create_pooling_descriptor(
    pooling_desc: *mut cudnnPoolingDescriptor_t,
) -> cudnnStatus_t {
    call!(miopenCreatePoolingDescriptor(pooling_desc as _))
}

unsafe fn set_tensor_nd_decriptor(
    tensor_desc: *mut cudnnTensorStruct,
    data_type: cudnnDataType_t,
    nb_dims: i32,
    dim_a: *const i32,
    stride_a: *const i32,
) -> cudnnStatus_t {
    let data_type = to_data_type(data_type);
    call!(miopenSetTensorDescriptor(
        tensor_desc as _,
        data_type,
        nb_dims,
        dim_a as _,
        stride_a as _,
    ))
}

fn to_data_type(type_: cudnnDataType_t) -> miopenDataType_t {
    match type_ {
        cudnnDataType_t::CUDNN_DATA_FLOAT => miopenDataType_t::miopenFloat,
        cudnnDataType_t::CUDNN_DATA_DOUBLE => miopenDataType_t::miopenDouble,
        cudnnDataType_t::CUDNN_DATA_HALF => miopenDataType_t::miopenHalf,
        cudnnDataType_t::CUDNN_DATA_BFLOAT16 => miopenDataType_t::miopenBFloat16,
        _ => todo!(),
    }
}

unsafe fn set_filter_nd_descriptor(
    filter_desc: cudnnFilterDescriptor_t,
    data_type: cudnnDataType_t,
    _format: cudnnTensorFormat_t,
    nb_dims: i32,
    filter_dim_a: *const i32,
) -> cudnnStatus_t {
    let data_type = to_data_type(data_type);
    call!(miopenSetTensorDescriptor(
        filter_desc as _,
        data_type,
        nb_dims,
        filter_dim_a as _,
        ptr::null_mut(),
    ))
}

unsafe fn set_convolution_nd_descriptor(
    conv_desc: cudnnConvolutionDescriptor_t,
    array_length: i32,
    pad_a: *const i32,
    filter_stride_a: *const i32,
    dilation_a: *const i32,
    mode: cudnnConvolutionMode_t,
    _compute_type: cudnnDataType_t,
) -> cudnnStatus_t {
    if array_length != 2 {
        todo!()
    }
    let pad_h = *pad_a.add(0);
    let pad_w = *pad_a.add(1);
    let u = *filter_stride_a.add(0);
    let v = *filter_stride_a.add(1);
    let d_h = *dilation_a.add(0);
    let d_w = *dilation_a.add(1);
    let mode = to_conv_mode(mode);
    call!(miopenInitConvolutionDescriptor(
        conv_desc as _,
        mode,
        pad_h,
        pad_w,
        u,
        v,
        d_h,
        d_w,
    ))
}

fn to_conv_mode(mode: cudnnConvolutionMode_t) -> miopenConvolutionMode_t {
    match mode {
        cudnnConvolutionMode_t::CUDNN_CONVOLUTION => miopenConvolutionMode_t::miopenTranspose,
        cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION => {
            miopenConvolutionMode_t::miopenConvolution
        }
        _ => panic!(),
    }
}

unsafe fn get_convolution_nd_forward_output_dim(
    conv_desc: cudnnConvolutionDescriptor_t,
    input_tensor_desc: cudnnTensorDescriptor_t,
    filter_desc: cudnnFilterDescriptor_t,
    mut nb_dims: i32,
    tensor_ouput_dim_a: *mut i32,
) -> cudnnStatus_t {
    call!(miopenGetConvolutionNdForwardOutputDim(
        conv_desc as _,
        input_tensor_desc as _,
        filter_desc as _,
        &mut nb_dims as *mut _,
        tensor_ouput_dim_a,
    ))
}

unsafe fn find_convolution_forward_algorithm(
    handle: cudnnHandle_t,
    x_desc: cudnnTensorDescriptor_t,
    w_desc: cudnnFilterDescriptor_t,
    conv_desc: cudnnConvolutionDescriptor_t,
    y_desc: cudnnTensorDescriptor_t,
    requested_algo_count: i32,
    returned_algo_count: *mut i32,
    perf_results: *mut cudnnConvolutionFwdAlgoPerf_t,
) -> cudnnStatus_t {
    let mut result = vec![mem::zeroed(); requested_algo_count as usize];
    let mut x_size = 0;
    asserted_call! { miopenGetTensorNumBytes(x_desc as _, &mut x_size) };
    let mut x = mem::zeroed();
    let error = hipMalloc(&mut x, x_size);
    if error != hipError_t::hipSuccess {
        return cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED;
    }
    let mut w_size = 0;
    asserted_call! { miopenGetTensorNumBytes(w_desc as _, &mut w_size) };
    let mut w = mem::zeroed();
    let error = hipMalloc(&mut w, w_size);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let mut y_size = 0;
    asserted_call! { miopenGetTensorNumBytes(y_desc as _, &mut y_size) };
    let mut y = mem::zeroed();
    let error = hipMalloc(&mut y, y_size);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let mut workspace_size = 0;
    asserted_call! { miopenConvolutionForwardGetWorkSpaceSize(handle as _, w_desc as _, x_desc as _, conv_desc as _, y_desc as _, &mut workspace_size) };
    let mut workspace = mem::zeroed();
    let error = hipMalloc(&mut workspace, workspace_size);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let status = call!(miopenFindConvolutionForwardAlgorithm(
        handle as _,
        x_desc as _,
        x,
        w_desc as _,
        w,
        conv_desc as _,
        y_desc as _,
        y,
        requested_algo_count,
        returned_algo_count,
        result.as_mut_ptr(),
        workspace,
        workspace_size,
        true,
    ));
    let error = hipFree(x);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let error = hipFree(w);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let error = hipFree(y);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let error = hipFree(workspace);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    for i in 0..result.len() {
        let result = result[i];
        *perf_results.add(i) = algoperf_to_cudnn(result);
    }
    status
}

unsafe fn find_convolution_forward_algorithm_ex(
    handle: *mut cudnnContext,
    x_desc: *mut cudnnTensorStruct,
    x: *const std::ffi::c_void,
    w_desc: *mut cudnnFilterStruct,
    w: *const std::ffi::c_void,
    conv_desc: *mut cudnnConvolutionStruct,
    y_desc: *mut cudnnTensorStruct,
    y: *mut std::ffi::c_void,
    requested_algo_count: i32,
    returned_algo_count: *mut i32,
    perf_results: *mut cudnnConvolutionFwdAlgoPerfStruct,
    work_space: *mut std::ffi::c_void,
    work_space_size_in_bytes: usize,
) -> cudnnStatus_t {
    let mut result = vec![mem::zeroed(); requested_algo_count as usize];
    let error = call!(miopenFindConvolutionForwardAlgorithm(
        handle as _,
        x_desc as _,
        x,
        w_desc as _,
        w,
        conv_desc as _,
        y_desc as _,
        y,
        requested_algo_count,
        returned_algo_count,
        result.as_mut_ptr(),
        work_space,
        work_space_size_in_bytes,
        true,
    ));
    for i in 0..result.len() {
        let result = result[i];
        *perf_results.add(i) = algoperf_to_cudnn(result);
    }
    error
}

unsafe fn algoperf_to_cudnn(result: miopenConvAlgoPerf_t) -> cudnnConvolutionFwdAlgoPerf_t {
    let algo = algo_to_cudnn(result);
    cudnnConvolutionFwdAlgoPerf_t {
        algo,
        status: cudnnStatus_t::CUDNN_STATUS_SUCCESS,
        time: result.time,
        memory: result.memory,
        determinism: cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
        mathType: cudnnMathType_t::CUDNN_DEFAULT_MATH,
        reserved: mem::zeroed(),
    }
}

unsafe fn algo_to_cudnn(result: miopenConvAlgoPerf_t) -> cudnnConvolutionFwdAlgo_t {
    match result.__bindgen_anon_1.fwd_algo {
        miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoGEMM => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM
        }
        miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoDirect => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
        }
        miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoFFT => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT
        }
        miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoWinograd => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
        }
        miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoImplicitGEMM => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
        }
        _ => cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    }
}

pub unsafe fn get_convolution_forward_workspace_size(
    handle: *mut cudnnContext,
    x_desc: *mut cudnnTensorStruct,
    w_desc: *mut cudnnFilterStruct,
    conv_desc: *mut cudnnConvolutionStruct,
    y_desc: *mut cudnnTensorStruct,
    _algo: cudnnConvolutionFwdAlgo_t,
    size_in_bytes: *mut usize,
) -> cudnnStatus_t {
    call!(miopenConvolutionForwardGetWorkSpaceSize(
        handle as _,
        w_desc as _,
        x_desc as _,
        conv_desc as _,
        y_desc as _,
        size_in_bytes,
    ))
}

unsafe fn convolution_forward(
    handle: *mut cudnnContext,
    alpha: *const std::ffi::c_void,
    x_desc: *mut cudnnTensorStruct,
    x: *const std::ffi::c_void,
    w_desc: *mut cudnnFilterStruct,
    w: *const std::ffi::c_void,
    conv_desc: *mut cudnnConvolutionStruct,
    algo: cudnnConvolutionFwdAlgo_t,
    work_space: *mut std::ffi::c_void,
    work_space_size_in_bytes: usize,
    beta: *const std::ffi::c_void,
    y_desc: *mut cudnnTensorStruct,
    y: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    let mut algo = algo_from_cudnn(algo);
    // In cuDNN it is possible to find algorithm for sizes X and then pass the algo
    // for sizes Y. On miOpen this fails
    let mut perf_results = vec![mem::zeroed(); 32];
    let mut algo_count = 0;
    asserted_call!(miopenFindConvolutionForwardAlgorithm(
        handle as _,
        x_desc as _,
        x,
        w_desc as _,
        w,
        conv_desc as _,
        y_desc as _,
        y,
        32,
        &mut algo_count,
        perf_results.as_mut_ptr(),
        work_space,
        work_space_size_in_bytes,
        true,
    ));
    if algo_count == 0 {
        panic!()
    }
    if let None = perf_results[..algo_count as usize]
        .iter()
        .find(|result| result.__bindgen_anon_1.fwd_algo == algo)
    {
        algo = perf_results[0].__bindgen_anon_1.fwd_algo;
    }
    call!(miopenConvolutionForward(
        handle as _,
        alpha,
        x_desc as _,
        x,
        w_desc as _,
        w,
        conv_desc as _,
        algo,
        beta,
        y_desc as _,
        y,
        work_space,
        work_space_size_in_bytes,
    ))
}

fn algo_from_cudnn(algo: cudnnConvolutionFwdAlgo_t) -> miopenConvFwdAlgorithm_t {
    match algo {
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM => {
            miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoImplicitGEMM
        }
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM => {
            miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoGEMM
        }
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM => {
            miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoGEMM
        }
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT => {
            miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoDirect
        }
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT => {
            miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoFFT
        }
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING => {
            miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoFFT
        }
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD => {
            miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoWinograd
        }
        cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED => {
            miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoWinograd
        }
        _ => miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoGEMM,
    }
}

unsafe fn add_tensor(
    handle: *mut cudnnContext,
    alpha: *const std::ffi::c_void,
    a_desc: *mut cudnnTensorStruct,
    a: *const std::ffi::c_void,
    beta: *const std::ffi::c_void,
    c_desc: *mut cudnnTensorStruct,
    c: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    // CUDA tensor A might be 1 in some dimensions
    // MIOpen tensors A and C must be the same
    let zero = 0f64;
    call!(miopenOpTensor(
        handle as _,
        miopenTensorOp_t::miopenTensorOpAdd,
        alpha,
        c_desc as _,
        c,
        beta,
        a_desc as _,
        a,
        &zero as *const _ as _,
        c_desc as _,
        c,
    ))
}

unsafe fn set_pooling_nd_descriptor(
    pooling_desc: *mut cudnnPoolingStruct,
    mode: cudnnPoolingMode_t,
    _maxpooling_nan_opt: cudnnNanPropagation_t,
    nb_dims: i32,
    window_dim_a: *const i32,
    padding_a: *const i32,
    stride_a: *const i32,
) -> cudnnStatus_t {
    let mode = pooling_from_cudnn(mode);
    call!(miopenSetNdPoolingDescriptor(
        pooling_desc as _,
        mode,
        nb_dims,
        window_dim_a as _,
        padding_a as _,
        stride_a as _,
    ))
}

fn pooling_from_cudnn(mode: cudnnPoolingMode_t) -> miopenPoolingMode_t {
    match mode {
        cudnnPoolingMode_t::CUDNN_POOLING_MAX => miopenPoolingMode_t::miopenPoolingMax,
        cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING => {
            miopenPoolingMode_t::miopenPoolingAverageInclusive
        }
        cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING => {
            miopenPoolingMode_t::miopenPoolingAverage
        }
        _ => todo!(),
    }
}

unsafe fn get_pooling_nd_forward_output_dim(
    pooling_desc: *mut cudnnPoolingStruct,
    input_tensor_desc: *mut cudnnTensorStruct,
    nb_dims: i32,
    output_tensor_dim_a: *mut i32,
) -> cudnnStatus_t {
    if nb_dims != 4 {
        todo!()
    }
    call!(miopenGetPoolingForwardOutputDim(
        pooling_desc as _,
        input_tensor_desc as _,
        output_tensor_dim_a.add(0),
        output_tensor_dim_a.add(1),
        output_tensor_dim_a.add(2),
        output_tensor_dim_a.add(3),
    ))
}

unsafe fn pooling_forward(
    handle: *mut cudnnContext,
    pooling_desc: *mut cudnnPoolingStruct,
    alpha: *const std::ffi::c_void,
    x_desc: *mut cudnnTensorStruct,
    x: *const std::ffi::c_void,
    beta: *const std::ffi::c_void,
    y_desc: *mut cudnnTensorStruct,
    y: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    let mut workspace_size = 0;
    asserted_call! { miopenPoolingGetWorkSpaceSize(y_desc as _, &mut workspace_size) };
    let mut workspace = mem::zeroed();
    let error = hipMalloc(&mut workspace, workspace_size);
    if error != hipError_t::hipSuccess {
        return cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED;
    }
    // TODO: Only alpha=1 and beta=0 is supported
    let result = call!(miopenPoolingForward(
        handle as _,
        pooling_desc as _,
        alpha,
        x_desc as _,
        x,
        beta,
        y_desc as _,
        y,
        false,
        workspace,
        workspace_size,
    ));
    let error = hipFree(workspace);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    result
}

unsafe fn set_activation_descriptor(
    activation_desc: *mut cudnnActivationStruct,
    mode: cudnnActivationMode_t,
    _relu_nan_opt: cudnnNanPropagation_t,
    coef: f64,
) -> cudnnStatus_t {
    let mode = activation_mode(mode);
    call!(miopenSetActivationDescriptor(
        activation_desc as _,
        mode,
        coef,
        0.0,
        0.0,
    ))
}

fn activation_mode(mode: cudnnActivationMode_t) -> miopenActivationMode_t {
    match mode {
        cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID => {
            miopenActivationMode_t::miopenActivationLOGISTIC
        }
        cudnnActivationMode_t::CUDNN_ACTIVATION_RELU => {
            miopenActivationMode_t::miopenActivationRELU
        }
        cudnnActivationMode_t::CUDNN_ACTIVATION_TANH => {
            miopenActivationMode_t::miopenActivationTANH
        }
        cudnnActivationMode_t::CUDNN_ACTIVATION_CLIPPED_RELU => {
            miopenActivationMode_t::miopenActivationCLIPPEDRELU
        }
        cudnnActivationMode_t::CUDNN_ACTIVATION_ELU => miopenActivationMode_t::miopenActivationELU,
        cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY => {
            miopenActivationMode_t::miopenActivationPASTHRU
        }
        _ => panic!(),
    }
}

unsafe fn activation_forward(
    handle: *mut cudnnContext,
    activation_desc: *mut cudnnActivationStruct,
    alpha: *const std::ffi::c_void,
    x_desc: *mut cudnnTensorStruct,
    x: *const std::ffi::c_void,
    beta: *const std::ffi::c_void,
    y_desc: *mut cudnnTensorStruct,
    y: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    call!(miopenActivationForward(
        handle as _,
        activation_desc as _,
        alpha,
        x_desc as _,
        x,
        beta,
        y_desc as _,
        y,
    ))
}

unsafe fn set_lrn_descriptor(
    norm_desc: *mut cudnnLRNStruct,
    lrn_n: u32,
    lrn_alpha: f64,
    lrn_beta: f64,
    lrn_k: f64,
) -> cudnnStatus_t {
    call!(miopenSetLRNDescriptor(
        norm_desc as _,
        miopenLRNMode_t::miopenLRNCrossChannel, // ???
        lrn_n,
        lrn_alpha,
        lrn_beta,
        lrn_k,
    ))
}

unsafe fn lrn_cross_channel_forward(
    handle: *mut cudnnContext,
    norm_desc: *mut cudnnLRNStruct,
    _lrn_mode: cudnnLRNMode_t,
    alpha: *const std::ffi::c_void,
    x_desc: *mut cudnnTensorStruct,
    x: *const std::ffi::c_void,
    beta: *const std::ffi::c_void,
    y_desc: *mut cudnnTensorStruct,
    y: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    call!(miopenLRNForward(
        handle as _,
        norm_desc as _,
        alpha,
        x_desc as _,
        x,
        beta,
        y_desc as _,
        y,
        false,
        ptr::null_mut(),
    ))
}

unsafe fn softmax_forward(
    handle: *mut cudnnContext,
    algo: cudnnSoftmaxAlgorithm_t,
    mode: cudnnSoftmaxMode_t,
    alpha: *const std::ffi::c_void,
    x_desc: *mut cudnnTensorStruct,
    x: *const std::ffi::c_void,
    beta: *const std::ffi::c_void,
    y_desc: *mut cudnnTensorStruct,
    y: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    let algo = softmax_algo(algo);
    let mode = softmax_mode(mode);
    call!(miopenSoftmaxForward_V2(
        handle as _,
        alpha,
        x_desc as _,
        x,
        beta,
        y_desc as _,
        y,
        algo,
        mode,
    ))
}

fn softmax_algo(algo: cudnnSoftmaxAlgorithm_t) -> miopenSoftmaxAlgorithm_t {
    match algo {
        cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE => {
            miopenSoftmaxAlgorithm_t::MIOPEN_SOFTMAX_ACCURATE
        }
        cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST => {
            miopenSoftmaxAlgorithm_t::MIOPEN_SOFTMAX_FAST
        }
        cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_LOG => miopenSoftmaxAlgorithm_t::MIOPEN_SOFTMAX_LOG,
        _ => panic!(),
    }
}

fn softmax_mode(mode: cudnnSoftmaxMode_t) -> miopenSoftmaxMode_t {
    match mode {
        cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL => {
            miopenSoftmaxMode_t::MIOPEN_SOFTMAX_MODE_CHANNEL
        }
        cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE => {
            miopenSoftmaxMode_t::MIOPEN_SOFTMAX_MODE_INSTANCE
        }
        _ => panic!(),
    }
}

unsafe fn destroy(handle: *mut cudnnContext) -> cudnnStatus_t {
    call!(miopenDestroy(handle as _))
}

unsafe fn destroy_activation_descriptor(
    activation_desc: *mut cudnnActivationStruct,
) -> cudnnStatus_t {
    call!(miopenDestroyActivationDescriptor(activation_desc as _))
}

unsafe fn destroy_convolution_descriptor(conv_desc: *mut cudnnConvolutionStruct) -> cudnnStatus_t {
    call!(miopenDestroyConvolutionDescriptor(conv_desc as _))
}

unsafe fn destroy_filter_descriptor(filter_desc: *mut cudnnFilterStruct) -> cudnnStatus_t {
    call!(miopenDestroyTensorDescriptor(filter_desc as _))
}

unsafe fn destroy_lrn_descriptor(lrn_desc: *mut cudnnLRNStruct) -> cudnnStatus_t {
    call!(miopenDestroyLRNDescriptor(lrn_desc as _))
}

unsafe fn destroy_pooling_descriptor(pooling_desc: *mut cudnnPoolingStruct) -> cudnnStatus_t {
    call!(miopenDestroyPoolingDescriptor(pooling_desc as _))
}

unsafe fn destroy_tensor_descriptor(tensor_desc: *mut cudnnTensorStruct) -> cudnnStatus_t {
    call!(miopenDestroyTensorDescriptor(tensor_desc as _))
}

unsafe fn set_tensor_4d_descriptor_ex(
    tensor_desc: *mut cudnnTensorStruct,
    data_type: cudnnDataType_t,
    n: i32,
    c: i32,
    h: i32,
    w: i32,
    n_stride: i32,
    c_stride: i32,
    h_stride: i32,
    w_stride: i32,
) -> cudnnStatus_t {
    let data_type = to_data_type(data_type);
    call!(miopenSet4dTensorDescriptorEx(
        tensor_desc as _,
        data_type,
        n,
        c,
        h,
        w,
        n_stride,
        c_stride,
        h_stride,
        w_stride,
    ))
}

unsafe fn transform_tensor(
    handle: *mut cudnnContext,
    alpha: *const std::ffi::c_void,
    x_desc: *mut cudnnTensorStruct,
    x: *const std::ffi::c_void,
    beta: *const std::ffi::c_void,
    y_desc: *mut cudnnTensorStruct,
    y: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    call!(miopenTransformTensor(
        handle as _,
        alpha,
        x_desc as _,
        x,
        beta,
        y_desc as _,
        y,
    ))
}

unsafe fn set_stream(handle: cudnnHandle_t, stream_id: *mut CUstream_st) -> cudnnStatus_t {
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
    let stream: Result<_, _> = zluda_ext.get_hip_stream(stream_id as _).into();
    call!(miopenSetStream(handle.cast(), stream.unwrap() as _))
}

fn set_convolution_math_type(
    _conv_desc: cudnnConvolutionDescriptor_t,
    _math_type: cudnnMathType_t,
) -> cudnnStatus_t {
    //TODO: implement
    cudnnStatus_t::CUDNN_STATUS_SUCCESS
}

unsafe fn set_convolution_group_count(
    conv_desc: *mut cudnnConvolutionStruct,
    group_count: i32,
) -> cudnnStatus_t {
    //TODO: implement
    call!(miopenSetConvolutionGroupCount(conv_desc as _, group_count))
}

unsafe fn get_convolution_backward_data_algorithm_max_count(
    _handle: *mut cudnnContext,
    count: *mut i32,
) -> cudnnStatus_t {
    *count = 1;
    cudnnStatus_t::CUDNN_STATUS_SUCCESS
}

unsafe fn get_convolution_backward_data_algorithm_v7(
    handle: *mut cudnnContext,
    w_desc: *mut cudnnFilterStruct,
    dy_desc: *mut cudnnTensorStruct,
    conv_desc: *mut cudnnConvolutionStruct,
    dx_desc: *mut cudnnTensorStruct,
    requested_algo_count: i32,
    returned_algo_count: *mut i32,
    perf_results: *mut cudnnConvolutionBwdDataAlgoPerf_t,
    memory_limit_in_bytes: usize,
) -> cudnnStatus_t {
    let mut work_space_size = 0;
    let mut dy_size = 0;
    asserted_call! { miopenGetTensorNumBytes(dy_desc as _, &mut dy_size) };
    let mut dy = mem::zeroed();
    let error = hipMalloc(&mut dy, dy_size);
    if error != hipError_t::hipSuccess {
        return cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED;
    }
    let mut w_size = 0;
    asserted_call! { miopenGetTensorNumBytes(w_desc as _, &mut w_size) };
    let mut w = mem::zeroed();
    let error = hipMalloc(&mut w, w_size);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let mut dx_size = 0;
    asserted_call! { miopenGetTensorNumBytes(dx_desc as _, &mut dx_size) };
    let mut dx = mem::zeroed();
    let error = hipMalloc(&mut dx, dx_size);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let error = miopenConvolutionBackwardDataGetWorkSpaceSize(
        handle as _,
        dy_desc as _,
        w_desc as _,
        conv_desc as _,
        dx_desc as _,
        &mut work_space_size,
    );
    work_space_size = work_space_size.min(memory_limit_in_bytes);
    if error != miopenStatus_t::miopenStatusSuccess {
        panic!("")
    }
    let mut work_space = mem::zeroed();
    if hipMalloc(&mut work_space, work_space_size) != hipError_t::hipSuccess {
        panic!("")
    }
    let mut miopen_perf_results = vec![mem::zeroed(); requested_algo_count as usize];
    let result = call!(miopenFindConvolutionBackwardDataAlgorithm(
        handle as _,
        dy_desc as _,
        dy,
        w_desc as _,
        w,
        conv_desc as _,
        dx_desc as _,
        dx,
        requested_algo_count,
        returned_algo_count,
        miopen_perf_results.as_mut_ptr(),
        work_space,
        work_space_size,
        true,
    ));
    let error = hipFree(dy);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let error = hipFree(w);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let error = hipFree(dx);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    let error = hipFree(work_space);
    if error != hipError_t::hipSuccess {
        panic!("{:?}", error);
    }
    for i in 0..*returned_algo_count {
        *perf_results.add(i as usize) = convert_bwd_algo(miopen_perf_results[i as usize]);
    }
    result
}

unsafe fn convert_bwd_algo(result: miopenConvAlgoPerf_t) -> cudnnConvolutionBwdDataAlgoPerf_t {
    let algo = bwd_data_algo_to_cudnn(result.__bindgen_anon_1.bwd_data_algo);
    cudnnConvolutionBwdDataAlgoPerf_t {
        algo,
        status: cudnnStatus_t::CUDNN_STATUS_SUCCESS,
        time: result.time,
        memory: result.memory,
        determinism: cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
        mathType: cudnnMathType_t::CUDNN_DEFAULT_MATH,
        reserved: mem::zeroed(),
    }
}

fn bwd_data_algo_to_cudnn(algo: miopenConvBwdDataAlgorithm_t) -> cudnnConvolutionBwdDataAlgo_t {
    match algo {
        miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoGEMM => {
            cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
        }
        miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoDirect => {
            cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
        }
        miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoWinograd => {
            cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD
        }
        miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoFFT => {
            cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT
        }
        miopenConvBwdDataAlgorithm_t::miopenTransposeBwdDataAlgoGEMM => {
            cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
        }
        miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoImplicitGEMM => {
            cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
        }
        _ => panic!(),
    }
}

fn bwd_data_algo_from_cudnn(algo: cudnnConvolutionBwdDataAlgo_t) -> miopenConvBwdDataAlgorithm_t {
    match algo {
        cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 => {
            miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoGEMM
        }
        cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 => {
            miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoDirect
        }
        cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT => {
            miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoFFT
        }
        cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING => {
            miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoFFT
        }
        cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD => {
            miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoWinograd
        }
        cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED => {
            miopenConvBwdDataAlgorithm_t::miopenConvolutionBwdDataAlgoWinograd
        }
        _ => panic!(),
    }
}

unsafe fn get_convolution_backward_data_workspace_size(
    handle: *mut cudnnContext,
    w_desc: *mut cudnnFilterStruct,
    dy_desc: *mut cudnnTensorStruct,
    conv_desc: *mut cudnnConvolutionStruct,
    dx_desc: *mut cudnnTensorStruct,
    _algo: cudnnConvolutionBwdDataAlgo_t,
    size_in_bytes: *mut usize,
) -> cudnnStatus_t {
    call!(miopenConvolutionBackwardDataGetWorkSpaceSize(
        handle as _,
        dy_desc as _,
        w_desc as _,
        conv_desc as _,
        dx_desc as _,
        size_in_bytes,
    ))
}

unsafe fn convolution_backward_data(
    handle: *mut cudnnContext,
    alpha: *const std::ffi::c_void,
    w_desc: *mut cudnnFilterStruct,
    w: *const std::ffi::c_void,
    dy_desc: *mut cudnnTensorStruct,
    dy: *const std::ffi::c_void,
    conv_desc: *mut cudnnConvolutionStruct,
    algo: cudnnConvolutionBwdDataAlgo_t,
    work_space: *mut std::ffi::c_void,
    work_space_size_in_bytes: usize,
    beta: *const std::ffi::c_void,
    dx_desc: *mut cudnnTensorStruct,
    dx: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    let algo = bwd_data_algo_from_cudnn(algo);
    call!(miopenConvolutionBackwardData(
        handle as _,
        alpha,
        dy_desc as _,
        dy,
        w_desc as _,
        w,
        conv_desc as _,
        algo,
        beta,
        dx_desc as _,
        dx,
        work_space,
        work_space_size_in_bytes,
    ))
}

unsafe fn get_stream(handle: *mut cudnnContext, stream_id: *mut cudaStream_t) -> cudnnStatus_t {
    call!(miopenGetStream(handle as _, stream_id as _))
}

impl cudnnBackendHeurMode_t {
    pub const CUDNN_HEUR_MODE_INSTANT: cudnnBackendHeurMode_t = cudnnBackendHeurMode_t(0);
}
impl cudnnBackendHeurMode_t {
    pub const CUDNN_HEUR_MODE_B: cudnnBackendHeurMode_t = cudnnBackendHeurMode_t(1);
}
impl cudnnBackendHeurMode_t {
    pub const CUDNN_HEUR_MODE_FALLBACK: cudnnBackendHeurMode_t = cudnnBackendHeurMode_t(2);
}
impl cudnnBackendHeurMode_t {
    pub const CUDNN_HEUR_MODE_A: cudnnBackendHeurMode_t = cudnnBackendHeurMode_t(3);
}
impl cudnnBackendHeurMode_t {
    pub const CUDNN_HEUR_MODES_COUNT: cudnnBackendHeurMode_t = cudnnBackendHeurMode_t(4);
}
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct cudnnBackendHeurMode_t(pub ::std::os::raw::c_uint);

impl miopenBackendHeurMode_t {
    pub const MIOPEN_HEUR_MODE_INSTANT: miopenBackendHeurMode_t = miopenBackendHeurMode_t(0);
}
impl miopenBackendHeurMode_t {
    pub const MIOPEN_HEUR_MODE_B: miopenBackendHeurMode_t = miopenBackendHeurMode_t(1);
}
impl miopenBackendHeurMode_t {
    pub const MIOPEN_HEUR_MODE_FALLBACK: miopenBackendHeurMode_t = miopenBackendHeurMode_t(2);
}
impl miopenBackendHeurMode_t {
    pub const MIOPEN_HEUR_MODE_A: miopenBackendHeurMode_t = miopenBackendHeurMode_t(3);
}
impl miopenBackendHeurMode_t {
    pub const MIOPEN_HEUR_MODES_COUNT: miopenBackendHeurMode_t = miopenBackendHeurMode_t(4);
}
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct miopenBackendHeurMode_t(pub ::std::os::raw::c_uint);

trait FromCuda<T: Sized>: Sized {
    fn from_cuda(t: T) -> Self {
        unsafe { mem::transmute_copy(&t) }
    }
}

impl FromCuda<cudnnBackendHeurMode_t> for miopenBackendHeurMode_t {}
impl FromCuda<cudnnBackendAttributeName_t> for miopenBackendAttributeName_t {}
impl FromCuda<cudnnBackendAttributeType_t> for miopenBackendAttributeType_t {}

#[derive(PartialEq, Eq)]
#[repr(u8)]
enum BackendDescriptorKind {
    /// Created by ZLUDA
    Owned = 0,
    /// Created by MIOpen internally
    System = 1,
    /// Persisting descriptor  
    /// This kind of descriptors will never be destroyed.  
    /// Used for hacking reasons.
    Persisting = 2,
}

const ZLUDA_DESCRIPTOR_MAGIC: ::std::os::raw::c_uint = 0x1B950F42;
#[repr(C)]
struct BackendDescriptor {
    magic: ::std::os::raw::c_uint,
    kind: BackendDescriptorKind,
    internal: miopenBackendDescriptor_t,
}

impl BackendDescriptor {
    fn new(internal: miopenBackendDescriptor_t) -> BackendDescriptor {
        BackendDescriptor {
            magic: ZLUDA_DESCRIPTOR_MAGIC,
            kind: BackendDescriptorKind::Owned,
            internal,
        }
    }

    fn dummy() -> BackendDescriptor {
        BackendDescriptor::persisting(ptr::null_mut())
    }

    fn persisting(internal: miopenBackendDescriptor_t) -> BackendDescriptor {
        BackendDescriptor {
            magic: ZLUDA_DESCRIPTOR_MAGIC,
            kind: BackendDescriptorKind::Persisting,
            internal,
        }
    }

    fn release(self) -> *mut BackendDescriptor {
        Box::into_raw(Box::new(self))
    }

    unsafe fn try_retrieve(raw: *mut BackendDescriptor) -> Option<Box<BackendDescriptor>> {
        if (*raw).magic != ZLUDA_DESCRIPTOR_MAGIC {
            return None;
        }
        Some(Box::from_raw(raw))
    }

    unsafe fn try_from<'a>(raw: miopenBackendDescriptor_t) -> Option<&'a mut BackendDescriptor> {
        let raw = raw as *mut BackendDescriptor;
        if (*raw).magic != ZLUDA_DESCRIPTOR_MAGIC {
            return None;
        }
        raw.as_mut()
    }
}

fn to_backend_descriptor_type(
    descriptor_type: cudnnBackendDescriptorType_t,
) -> miopenBackendDescriptorType_t {
    match descriptor_type {
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR => {
            miopenBackendDescriptorType_t::MIOPEN_BACKEND_CONVOLUTION_DESCRIPTOR
        }
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINE_DESCRIPTOR => {
            miopenBackendDescriptorType_t::MIOPEN_BACKEND_ENGINE_DESCRIPTOR
        }
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINECFG_DESCRIPTOR => {
            miopenBackendDescriptorType_t::MIOPEN_BACKEND_ENGINECFG_DESCRIPTOR
        }
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR => {
            miopenBackendDescriptorType_t::MIOPEN_BACKEND_ENGINEHEUR_DESCRIPTOR
        }
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR => {
            miopenBackendDescriptorType_t::MIOPEN_BACKEND_EXECUTION_PLAN_DESCRIPTOR
        }
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR => {
            miopenBackendDescriptorType_t::MIOPEN_BACKEND_KNOB_CHOICE_DESCRIPTOR
        }
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR => {
            miopenBackendDescriptorType_t::MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR
        }
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR => {
            miopenBackendDescriptorType_t::MIOPEN_BACKEND_OPERATIONGRAPH_DESCRIPTOR
        }
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR => {
            miopenBackendDescriptorType_t::MIOPEN_BACKEND_VARIANT_PACK_DESCRIPTOR
        }
        cudnnBackendDescriptorType_t::CUDNN_BACKEND_TENSOR_DESCRIPTOR => {
            miopenBackendDescriptorType_t::MIOPEN_BACKEND_TENSOR_DESCRIPTOR
        }
        _ => panic!("[ZLUDA] Unknown descriptor type: {}", descriptor_type.0),
    }
}

unsafe fn backend_create_descriptor(
    descriptor_type: cudnnBackendDescriptorType_t,
    descriptor: *mut cudnnBackendDescriptor_t,
) -> cudnnStatus_t {
    let descriptor_type = to_backend_descriptor_type(descriptor_type);

    if descriptor_type == miopenBackendDescriptorType_t::MIOPEN_BACKEND_KNOB_CHOICE_DESCRIPTOR {
        *descriptor = BackendDescriptor::dummy().release() as _;
        return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
    }

    let result = call!(miopenBackendCreateDescriptor(
        descriptor_type,
        descriptor.cast(),
    ));

    if descriptor_type == miopenBackendDescriptorType_t::MIOPEN_BACKEND_OPERATIONGRAPH_DESCRIPTOR {
        // We cannot destroy OperationGraph descriptor.
        // Once OperationGraph descriptor is destroyed,
        // the vectors that contains the information of
        // the convolution to be forwarded is unloaded
        // simultaneously from the memory.
        // Indeed, this leads to memory leak.
        // However, there is no other clear way to
        // make this work.
        *descriptor = BackendDescriptor::persisting((*descriptor).cast()).release() as _;
    } else {
        // MIOpen behaves differently from cuDNN.
        // cuDNN "updates" the contents (members in C) of the descriptors.
        // However, MIOpen "replaces" the descriptors with their internal descriptors.
        // Therefore, we wrap MIOpen backend descriptor inside our own BackendDescriptor
        // to get full control and distinction of the descriptor.
        // e.g.
        // https://github.com/NVIDIA/cudnn-frontend/blob/5040925e9450c399a66240b485b38564226e1212/include/cudnn_frontend_Heuristics.h#L95
        *descriptor = BackendDescriptor::new((*descriptor).cast()).release() as _;
    }

    result
}

unsafe fn backend_destroy_descriptor(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t {
    if let Some(descriptor) = BackendDescriptor::try_retrieve(descriptor.cast()) {
        if matches!(
            descriptor.kind,
            BackendDescriptorKind::System | BackendDescriptorKind::Persisting
        ) {
            // Cannot destroy the descriptor as it is created from MIOpen internally.
            // Persisting descriptors will not be destroyed. (it may be dummy or OperationGraph descriptor)
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        }
        return call!(miopenBackendDestroyDescriptor(descriptor.internal));
    }
    cudnnStatus_t::CUDNN_STATUS_BAD_PARAM
}

unsafe fn backend_finalize(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t {
    if let Some(descriptor) = BackendDescriptor::try_from(descriptor.cast()) {
        return call!(miopenBackendFinalize(descriptor.internal));
    }
    cudnnStatus_t::CUDNN_STATUS_BAD_PARAM
}

unsafe fn backend_set_attribute_impl(
    descriptor: miopenBackendDescriptor_t,
    attribute_name: miopenBackendAttributeName_t,
    attribute_type: miopenBackendAttributeType_t,
    element_count: i64,
    array_of_elements: *mut ::std::os::raw::c_void,
) -> miopenStatus_t {
    if attribute_name == miopenBackendAttributeName_t::MIOPEN_ATTR_TENSOR_BYTE_ALIGNMENT {
        return miopenStatus_t::miopenStatusSuccess;
    }

    match attribute_type {
        miopenBackendAttributeType_t::MIOPEN_TYPE_DATA_TYPE => {
            if element_count != 1 {
                panic!("[ZLUDA] Unexpected value: element_count={}", element_count)
            }
            let data_type = to_data_type(*array_of_elements.cast());
            miopenBackendSetAttribute(
                descriptor,
                attribute_name,
                attribute_type,
                element_count,
                &raw const data_type as _,
            )
        }
        miopenBackendAttributeType_t::MIOPEN_TYPE_CONVOLUTION_MODE => {
            if element_count != 1 {
                panic!("[ZLUDA] Unexpected value: element_count={}", element_count)
            }
            let conv_type = to_conv_mode(*array_of_elements.cast());
            miopenBackendSetAttribute(
                descriptor,
                attribute_name,
                attribute_type,
                element_count,
                &raw const conv_type as _,
            )
        }
        miopenBackendAttributeType_t::MIOPEN_TYPE_HEUR_MODE => {
            if element_count != 1 {
                panic!("[ZLUDA] Unexpected value: element_count={}", element_count)
            }
            let heur_mode: miopenBackendHeurMode_t = *array_of_elements.cast();
            miopenBackendSetAttribute(
                descriptor,
                attribute_name,
                attribute_type,
                element_count,
                &raw const heur_mode as _,
            )
        }
        miopenBackendAttributeType_t::MIOPEN_TYPE_BACKEND_DESCRIPTOR => {
            let count = element_count as usize;
            let mut elements = Vec::with_capacity(count);
            for i in 0..count {
                if let Some(descriptor) = BackendDescriptor::try_from(
                    *(array_of_elements as *mut miopenBackendDescriptor_t).add(i),
                ) {
                    elements.push(descriptor.internal);
                } else {
                    return miopenStatus_t::miopenStatusBadParm;
                }
            }
            miopenBackendSetAttribute(
                descriptor,
                attribute_name,
                attribute_type,
                element_count,
                elements.as_mut_ptr() as _,
            )
        }
        miopenBackendAttributeType_t::MIOPEN_TYPE_HANDLE
        | miopenBackendAttributeType_t::MIOPEN_TYPE_INT64
        | miopenBackendAttributeType_t::MIOPEN_TYPE_FLOAT
        | miopenBackendAttributeType_t::MIOPEN_TYPE_DOUBLE
        | miopenBackendAttributeType_t::MIOPEN_TYPE_VOID_PTR => miopenBackendSetAttribute(
            descriptor,
            attribute_name,
            attribute_type,
            element_count,
            array_of_elements,
        ),
        _ => panic!(
            "[ZLUDA] Unknown backend attribute type: {}",
            attribute_type.0
        ),
    }
}

unsafe fn backend_set_attribute(
    descriptor: cudnnBackendDescriptor_t,
    attribute_name: cudnnBackendAttributeName_t,
    attribute_type: cudnnBackendAttributeType_t,
    element_count: i64,
    array_of_elements: *const ::std::os::raw::c_void,
) -> cudnnStatus_t {
    if let Some(descriptor) = BackendDescriptor::try_from(descriptor.cast()) {
        let attribute_name = miopenBackendAttributeName_t::from_cuda(attribute_name);
        let attribute_type = miopenBackendAttributeType_t::from_cuda(attribute_type);

        return call!(backend_set_attribute_impl(
            descriptor.internal,
            attribute_name,
            attribute_type,
            element_count,
            array_of_elements.cast_mut()
        ));
    }

    cudnnStatus_t::CUDNN_STATUS_BAD_PARAM
}

unsafe fn backend_get_attribute(
    descriptor: cudnnBackendDescriptor_t,
    attribute_name: cudnnBackendAttributeName_t,
    attribute_type: cudnnBackendAttributeType_t,
    requested_element_count: i64,
    element_count: *mut i64,
    array_of_elements: *mut ::std::os::raw::c_void,
) -> cudnnStatus_t {
    if let Some(descriptor) = BackendDescriptor::try_from(descriptor.cast()) {
        let attribute_name = miopenBackendAttributeName_t::from_cuda(attribute_name);
        let attribute_type = miopenBackendAttributeType_t::from_cuda(attribute_type);

        // cuDNN frontend
        if requested_element_count == 0
            && attribute_name == miopenBackendAttributeName_t::MIOPEN_ATTR_ENGINEHEUR_RESULTS
        {
            let mut array_of_elements = mem::zeroed::<miopenBackendDescriptor_t>();
            return call!(miopenBackendGetAttribute(
                descriptor.internal,
                attribute_name,
                attribute_type,
                1,
                element_count,
                &raw mut array_of_elements as _,
            ));
        }

        if attribute_type == miopenBackendAttributeType_t::MIOPEN_TYPE_BACKEND_DESCRIPTOR {
            let mut descriptors =
                vec![mem::zeroed::<miopenBackendDescriptor_t>(); requested_element_count as usize];
            asserted_call!(miopenBackendGetAttribute(
                descriptor.internal,
                attribute_name,
                attribute_type,
                requested_element_count,
                element_count,
                descriptors.as_mut_ptr().cast(),
            ));

            for i in 0..(*element_count as usize) {
                if let Some(descriptor) = BackendDescriptor::try_from(
                    *(array_of_elements as *mut miopenBackendDescriptor_t).add(i),
                ) {
                    asserted_call!(miopenBackendDestroyDescriptor(descriptor.internal));
                    descriptor.kind = BackendDescriptorKind::System;
                    descriptor.internal = descriptors[i];
                    continue;
                }

                return cudnnStatus_t::CUDNN_STATUS_BAD_PARAM;
            }

            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        }

        // cuDNN frontend
        if element_count == ptr::null_mut()
            && attribute_name
                == miopenBackendAttributeName_t::MIOPEN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE
        {
            assert_eq!(requested_element_count, 1);

            let mut element_count = 0i64;
            return call!(miopenBackendGetAttribute(
                descriptor.internal,
                attribute_name,
                attribute_type,
                requested_element_count,
                &mut element_count,
                array_of_elements
            ));
        }

        // cuDNN frontend
        if array_of_elements == ptr::null_mut()
            && matches!(
                attribute_type,
                miopenBackendAttributeType_t::MIOPEN_TYPE_NUMERICAL_NOTE
                    | miopenBackendAttributeType_t::MIOPEN_TYPE_BEHAVIOR_NOTE
            )
        {
            let mut array_of_elements = mem::zeroed::<*mut ::std::os::raw::c_void>();
            return call!(miopenBackendGetAttribute(
                descriptor.internal,
                attribute_name,
                attribute_type,
                1,
                element_count,
                &raw mut array_of_elements as _,
            ));
        }

        return call!(miopenBackendGetAttribute(
            descriptor.internal,
            attribute_name,
            attribute_type,
            requested_element_count,
            element_count,
            array_of_elements
        ));
    }

    cudnnStatus_t::CUDNN_STATUS_BAD_PARAM
}

unsafe fn backend_execute(
    handle: cudnnHandle_t,
    execution_plan: cudnnBackendDescriptor_t,
    variant_pack: cudnnBackendDescriptor_t,
) -> cudnnStatus_t {
    let execution_plan = BackendDescriptor::try_from(execution_plan.cast());
    let variant_pack = BackendDescriptor::try_from(variant_pack.cast());
    if execution_plan.is_none() || variant_pack.is_none() {
        return cudnnStatus_t::CUDNN_STATUS_BAD_PARAM;
    }
    call!(miopenBackendExecute(
        handle.cast(),
        execution_plan.unwrap().internal,
        variant_pack.unwrap().internal,
    ))
}
