#[allow(warnings)]
mod hipblaslt;
pub use hipblaslt::*;

impl hipblasOperation_t {
    pub const HIPBLAS_OP_N: hipblasOperation_t = hipblasOperation_t(111);
}
impl hipblasOperation_t {
    pub const HIPBLAS_OP_T: hipblasOperation_t = hipblasOperation_t(112);
}
impl hipblasOperation_t {
    pub const HIPBLAS_OP_C: hipblasOperation_t = hipblasOperation_t(113);
}
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipblasOperation_t(pub ::std::os::raw::c_int);
