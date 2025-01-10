#[repr(C)]
#[repr(align(8))]
#[derive(Copy, Clone)]
pub struct float2 {
    pub x: f32,
    pub y: f32,
}
#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone)]
pub struct double2 {
    pub x: f64,
    pub y: f64,
}
pub type cuFloatComplex = float2;
pub type cuDoubleComplex = double2;
pub type cuComplex = cuFloatComplex;
impl cublasStatus_t {
    pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = cublasStatus_t(0);
}
impl cublasStatus_t {
    pub const CUBLAS_STATUS_NOT_INITIALIZED: cublasStatus_t = cublasStatus_t(1);
}
impl cublasStatus_t {
    pub const CUBLAS_STATUS_ALLOC_FAILED: cublasStatus_t = cublasStatus_t(3);
}
impl cublasStatus_t {
    pub const CUBLAS_STATUS_INVALID_VALUE: cublasStatus_t = cublasStatus_t(7);
}
impl cublasStatus_t {
    pub const CUBLAS_STATUS_ARCH_MISMATCH: cublasStatus_t = cublasStatus_t(8);
}
impl cublasStatus_t {
    pub const CUBLAS_STATUS_MAPPING_ERROR: cublasStatus_t = cublasStatus_t(11);
}
impl cublasStatus_t {
    pub const CUBLAS_STATUS_EXECUTION_FAILED: cublasStatus_t = cublasStatus_t(13);
}
impl cublasStatus_t {
    pub const CUBLAS_STATUS_INTERNAL_ERROR: cublasStatus_t = cublasStatus_t(14);
}
impl cublasStatus_t {
    pub const CUBLAS_STATUS_NOT_SUPPORTED: cublasStatus_t = cublasStatus_t(15);
}
impl cublasStatus_t {
    pub const CUBLAS_STATUS_LICENSE_ERROR: cublasStatus_t = cublasStatus_t(16);
}
#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct cublasStatus_t(pub ::std::os::raw::c_int);
impl cublasFillMode_t {
    pub const CUBLAS_FILL_MODE_LOWER: cublasFillMode_t = cublasFillMode_t(0);
}
impl cublasFillMode_t {
    pub const CUBLAS_FILL_MODE_UPPER: cublasFillMode_t = cublasFillMode_t(1);
}
impl cublasFillMode_t {
    pub const CUBLAS_FILL_MODE_FULL: cublasFillMode_t = cublasFillMode_t(2);
}
#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct cublasFillMode_t(pub ::std::os::raw::c_int);
impl cublasDiagType_t {
    pub const CUBLAS_DIAG_NON_UNIT: cublasDiagType_t = cublasDiagType_t(0);
}
impl cublasDiagType_t {
    pub const CUBLAS_DIAG_UNIT: cublasDiagType_t = cublasDiagType_t(1);
}
#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct cublasDiagType_t(pub ::std::os::raw::c_int);
impl cublasSideMode_t {
    pub const CUBLAS_SIDE_LEFT: cublasSideMode_t = cublasSideMode_t(0);
}
impl cublasSideMode_t {
    pub const CUBLAS_SIDE_RIGHT: cublasSideMode_t = cublasSideMode_t(1);
}
#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct cublasSideMode_t(pub ::std::os::raw::c_int);
impl cublasOperation_t {
    pub const CUBLAS_OP_N: cublasOperation_t = cublasOperation_t(0);
}
impl cublasOperation_t {
    pub const CUBLAS_OP_T: cublasOperation_t = cublasOperation_t(1);
}
impl cublasOperation_t {
    pub const CUBLAS_OP_C: cublasOperation_t = cublasOperation_t(2);
}
impl cublasOperation_t {
    pub const CUBLAS_OP_HERMITAN: cublasOperation_t = cublasOperation_t(2);
}
impl cublasOperation_t {
    pub const CUBLAS_OP_CONJG: cublasOperation_t = cublasOperation_t(3);
}
#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct cublasOperation_t(pub ::std::os::raw::c_int);
