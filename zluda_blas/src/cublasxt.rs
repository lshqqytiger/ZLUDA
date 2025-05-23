use crate::common::*;

/* automatically generated by rust-bindgen 0.69.4 */

#[repr(C)]
#[derive(Copy, Clone)]
pub struct cublasXtContext {
    _unused: [u8; 0],
}
pub type cublasXtHandle_t = *mut cublasXtContext;

#[no_mangle]
pub extern "system" fn cublasXtCreate(handle: *mut cublasXtHandle_t) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtDestroy(handle: cublasXtHandle_t) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtGetNumBoards(
    nbDevices: ::std::os::raw::c_int,
    deviceId: *mut ::std::os::raw::c_int,
    nbBoards: *mut ::std::os::raw::c_int,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtMaxBoards(
    nbGpuBoards: *mut ::std::os::raw::c_int,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtDeviceSelect(
    handle: cublasXtHandle_t,
    nbDevices: ::std::os::raw::c_int,
    deviceId: *mut ::std::os::raw::c_int,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtSetBlockDim(
    handle: cublasXtHandle_t,
    blockDim: ::std::os::raw::c_int,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtGetBlockDim(
    handle: cublasXtHandle_t,
    blockDim: *mut ::std::os::raw::c_int,
) -> cublasStatus_t {
    crate::unsupported()
}
impl cublasXtPinnedMemMode_t {
    pub const CUBLASXT_PINNING_DISABLED: cublasXtPinnedMemMode_t = cublasXtPinnedMemMode_t(0);
}
impl cublasXtPinnedMemMode_t {
    pub const CUBLASXT_PINNING_ENABLED: cublasXtPinnedMemMode_t = cublasXtPinnedMemMode_t(1);
}
#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct cublasXtPinnedMemMode_t(pub ::std::os::raw::c_int);

#[no_mangle]
pub extern "system" fn cublasXtGetPinningMemMode(
    handle: cublasXtHandle_t,
    mode: *mut cublasXtPinnedMemMode_t,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtSetPinningMemMode(
    handle: cublasXtHandle_t,
    mode: cublasXtPinnedMemMode_t,
) -> cublasStatus_t {
    crate::unsupported()
}
impl cublasXtOpType_t {
    pub const CUBLASXT_FLOAT: cublasXtOpType_t = cublasXtOpType_t(0);
}
impl cublasXtOpType_t {
    pub const CUBLASXT_DOUBLE: cublasXtOpType_t = cublasXtOpType_t(1);
}
impl cublasXtOpType_t {
    pub const CUBLASXT_COMPLEX: cublasXtOpType_t = cublasXtOpType_t(2);
}
impl cublasXtOpType_t {
    pub const CUBLASXT_DOUBLECOMPLEX: cublasXtOpType_t = cublasXtOpType_t(3);
}
#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct cublasXtOpType_t(pub ::std::os::raw::c_int);
impl cublasXtBlasOp_t {
    pub const CUBLASXT_GEMM: cublasXtBlasOp_t = cublasXtBlasOp_t(0);
}
impl cublasXtBlasOp_t {
    pub const CUBLASXT_SYRK: cublasXtBlasOp_t = cublasXtBlasOp_t(1);
}
impl cublasXtBlasOp_t {
    pub const CUBLASXT_HERK: cublasXtBlasOp_t = cublasXtBlasOp_t(2);
}
impl cublasXtBlasOp_t {
    pub const CUBLASXT_SYMM: cublasXtBlasOp_t = cublasXtBlasOp_t(3);
}
impl cublasXtBlasOp_t {
    pub const CUBLASXT_HEMM: cublasXtBlasOp_t = cublasXtBlasOp_t(4);
}
impl cublasXtBlasOp_t {
    pub const CUBLASXT_TRSM: cublasXtBlasOp_t = cublasXtBlasOp_t(5);
}
impl cublasXtBlasOp_t {
    pub const CUBLASXT_SYR2K: cublasXtBlasOp_t = cublasXtBlasOp_t(6);
}
impl cublasXtBlasOp_t {
    pub const CUBLASXT_HER2K: cublasXtBlasOp_t = cublasXtBlasOp_t(7);
}
impl cublasXtBlasOp_t {
    pub const CUBLASXT_SPMM: cublasXtBlasOp_t = cublasXtBlasOp_t(8);
}
impl cublasXtBlasOp_t {
    pub const CUBLASXT_SYRKX: cublasXtBlasOp_t = cublasXtBlasOp_t(9);
}
impl cublasXtBlasOp_t {
    pub const CUBLASXT_HERKX: cublasXtBlasOp_t = cublasXtBlasOp_t(10);
}
impl cublasXtBlasOp_t {
    pub const CUBLASXT_TRMM: cublasXtBlasOp_t = cublasXtBlasOp_t(11);
}
impl cublasXtBlasOp_t {
    pub const CUBLASXT_ROUTINE_MAX: cublasXtBlasOp_t = cublasXtBlasOp_t(12);
}
#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct cublasXtBlasOp_t(pub ::std::os::raw::c_int);

#[no_mangle]
pub extern "system" fn cublasXtSetCpuRoutine(
    handle: cublasXtHandle_t,
    blasOp: cublasXtBlasOp_t,
    type_: cublasXtOpType_t,
    blasFunctor: *mut ::std::os::raw::c_void,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtSetCpuRatio(
    handle: cublasXtHandle_t,
    blasOp: cublasXtBlasOp_t,
    type_: cublasXtOpType_t,
    ratio: f32,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtSgemm(
    handle: cublasXtHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f32,
    A: *const f32,
    lda: usize,
    B: *const f32,
    ldb: usize,
    beta: *const f32,
    C: *mut f32,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtDgemm(
    handle: cublasXtHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const f64,
    A: *const f64,
    lda: usize,
    B: *const f64,
    ldb: usize,
    beta: *const f64,
    C: *mut f64,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtCgemm(
    handle: cublasXtHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: usize,
    B: *const cuComplex,
    ldb: usize,
    beta: *const cuComplex,
    C: *mut cuComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtZgemm(
    handle: cublasXtHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: usize,
    n: usize,
    k: usize,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: usize,
    B: *const cuDoubleComplex,
    ldb: usize,
    beta: *const cuDoubleComplex,
    C: *mut cuDoubleComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtSsyrk(
    handle: cublasXtHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: usize,
    k: usize,
    alpha: *const f32,
    A: *const f32,
    lda: usize,
    beta: *const f32,
    C: *mut f32,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtDsyrk(
    handle: cublasXtHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: usize,
    k: usize,
    alpha: *const f64,
    A: *const f64,
    lda: usize,
    beta: *const f64,
    C: *mut f64,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtCsyrk(
    handle: cublasXtHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: usize,
    k: usize,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: usize,
    beta: *const cuComplex,
    C: *mut cuComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtZsyrk(
    handle: cublasXtHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: usize,
    k: usize,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: usize,
    beta: *const cuDoubleComplex,
    C: *mut cuDoubleComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtCherk(
    handle: cublasXtHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: usize,
    k: usize,
    alpha: *const f32,
    A: *const cuComplex,
    lda: usize,
    beta: *const f32,
    C: *mut cuComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtZherk(
    handle: cublasXtHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: usize,
    k: usize,
    alpha: *const f64,
    A: *const cuDoubleComplex,
    lda: usize,
    beta: *const f64,
    C: *mut cuDoubleComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtSsyr2k(
    handle: cublasXtHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: usize,
    k: usize,
    alpha: *const f32,
    A: *const f32,
    lda: usize,
    B: *const f32,
    ldb: usize,
    beta: *const f32,
    C: *mut f32,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtDsyr2k(
    handle: cublasXtHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: usize,
    k: usize,
    alpha: *const f64,
    A: *const f64,
    lda: usize,
    B: *const f64,
    ldb: usize,
    beta: *const f64,
    C: *mut f64,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtCsyr2k(
    handle: cublasXtHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: usize,
    k: usize,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: usize,
    B: *const cuComplex,
    ldb: usize,
    beta: *const cuComplex,
    C: *mut cuComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtZsyr2k(
    handle: cublasXtHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: usize,
    k: usize,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: usize,
    B: *const cuDoubleComplex,
    ldb: usize,
    beta: *const cuDoubleComplex,
    C: *mut cuDoubleComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtCherkx(
    handle: cublasXtHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: usize,
    k: usize,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: usize,
    B: *const cuComplex,
    ldb: usize,
    beta: *const f32,
    C: *mut cuComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtZherkx(
    handle: cublasXtHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: usize,
    k: usize,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: usize,
    B: *const cuDoubleComplex,
    ldb: usize,
    beta: *const f64,
    C: *mut cuDoubleComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtStrsm(
    handle: cublasXtHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: usize,
    n: usize,
    alpha: *const f32,
    A: *const f32,
    lda: usize,
    B: *mut f32,
    ldb: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtDtrsm(
    handle: cublasXtHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: usize,
    n: usize,
    alpha: *const f64,
    A: *const f64,
    lda: usize,
    B: *mut f64,
    ldb: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtCtrsm(
    handle: cublasXtHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: usize,
    n: usize,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: usize,
    B: *mut cuComplex,
    ldb: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtZtrsm(
    handle: cublasXtHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: usize,
    n: usize,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: usize,
    B: *mut cuDoubleComplex,
    ldb: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtSsymm(
    handle: cublasXtHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: usize,
    n: usize,
    alpha: *const f32,
    A: *const f32,
    lda: usize,
    B: *const f32,
    ldb: usize,
    beta: *const f32,
    C: *mut f32,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtDsymm(
    handle: cublasXtHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: usize,
    n: usize,
    alpha: *const f64,
    A: *const f64,
    lda: usize,
    B: *const f64,
    ldb: usize,
    beta: *const f64,
    C: *mut f64,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtCsymm(
    handle: cublasXtHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: usize,
    n: usize,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: usize,
    B: *const cuComplex,
    ldb: usize,
    beta: *const cuComplex,
    C: *mut cuComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtZsymm(
    handle: cublasXtHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: usize,
    n: usize,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: usize,
    B: *const cuDoubleComplex,
    ldb: usize,
    beta: *const cuDoubleComplex,
    C: *mut cuDoubleComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtChemm(
    handle: cublasXtHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: usize,
    n: usize,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: usize,
    B: *const cuComplex,
    ldb: usize,
    beta: *const cuComplex,
    C: *mut cuComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtZhemm(
    handle: cublasXtHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: usize,
    n: usize,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: usize,
    B: *const cuDoubleComplex,
    ldb: usize,
    beta: *const cuDoubleComplex,
    C: *mut cuDoubleComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtSsyrkx(
    handle: cublasXtHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: usize,
    k: usize,
    alpha: *const f32,
    A: *const f32,
    lda: usize,
    B: *const f32,
    ldb: usize,
    beta: *const f32,
    C: *mut f32,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtDsyrkx(
    handle: cublasXtHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: usize,
    k: usize,
    alpha: *const f64,
    A: *const f64,
    lda: usize,
    B: *const f64,
    ldb: usize,
    beta: *const f64,
    C: *mut f64,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtCsyrkx(
    handle: cublasXtHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: usize,
    k: usize,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: usize,
    B: *const cuComplex,
    ldb: usize,
    beta: *const cuComplex,
    C: *mut cuComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtZsyrkx(
    handle: cublasXtHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: usize,
    k: usize,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: usize,
    B: *const cuDoubleComplex,
    ldb: usize,
    beta: *const cuDoubleComplex,
    C: *mut cuDoubleComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtCher2k(
    handle: cublasXtHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: usize,
    k: usize,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: usize,
    B: *const cuComplex,
    ldb: usize,
    beta: *const f32,
    C: *mut cuComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtZher2k(
    handle: cublasXtHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: usize,
    k: usize,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: usize,
    B: *const cuDoubleComplex,
    ldb: usize,
    beta: *const f64,
    C: *mut cuDoubleComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtSspmm(
    handle: cublasXtHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: usize,
    n: usize,
    alpha: *const f32,
    AP: *const f32,
    B: *const f32,
    ldb: usize,
    beta: *const f32,
    C: *mut f32,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtDspmm(
    handle: cublasXtHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: usize,
    n: usize,
    alpha: *const f64,
    AP: *const f64,
    B: *const f64,
    ldb: usize,
    beta: *const f64,
    C: *mut f64,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtCspmm(
    handle: cublasXtHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: usize,
    n: usize,
    alpha: *const cuComplex,
    AP: *const cuComplex,
    B: *const cuComplex,
    ldb: usize,
    beta: *const cuComplex,
    C: *mut cuComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtZspmm(
    handle: cublasXtHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: usize,
    n: usize,
    alpha: *const cuDoubleComplex,
    AP: *const cuDoubleComplex,
    B: *const cuDoubleComplex,
    ldb: usize,
    beta: *const cuDoubleComplex,
    C: *mut cuDoubleComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtStrmm(
    handle: cublasXtHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: usize,
    n: usize,
    alpha: *const f32,
    A: *const f32,
    lda: usize,
    B: *const f32,
    ldb: usize,
    C: *mut f32,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtDtrmm(
    handle: cublasXtHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: usize,
    n: usize,
    alpha: *const f64,
    A: *const f64,
    lda: usize,
    B: *const f64,
    ldb: usize,
    C: *mut f64,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtCtrmm(
    handle: cublasXtHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: usize,
    n: usize,
    alpha: *const cuComplex,
    A: *const cuComplex,
    lda: usize,
    B: *const cuComplex,
    ldb: usize,
    C: *mut cuComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cublasXtZtrmm(
    handle: cublasXtHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: usize,
    n: usize,
    alpha: *const cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: usize,
    B: *const cuDoubleComplex,
    ldb: usize,
    C: *mut cuDoubleComplex,
    ldc: usize,
) -> cublasStatus_t {
    crate::unsupported()
}
