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

impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_ADD: cudnnPointwiseMode_t = cudnnPointwiseMode_t(0);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_ADD_SQUARE: cudnnPointwiseMode_t = cudnnPointwiseMode_t(5);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_DIV: cudnnPointwiseMode_t = cudnnPointwiseMode_t(6);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_MAX: cudnnPointwiseMode_t = cudnnPointwiseMode_t(3);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_MIN: cudnnPointwiseMode_t = cudnnPointwiseMode_t(2);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_MOD: cudnnPointwiseMode_t = cudnnPointwiseMode_t(7);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_MUL: cudnnPointwiseMode_t = cudnnPointwiseMode_t(1);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_POW: cudnnPointwiseMode_t = cudnnPointwiseMode_t(8);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_SUB: cudnnPointwiseMode_t = cudnnPointwiseMode_t(9);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_ABS: cudnnPointwiseMode_t = cudnnPointwiseMode_t(10);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_CEIL: cudnnPointwiseMode_t = cudnnPointwiseMode_t(11);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_COS: cudnnPointwiseMode_t = cudnnPointwiseMode_t(12);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_EXP: cudnnPointwiseMode_t = cudnnPointwiseMode_t(13);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_FLOOR: cudnnPointwiseMode_t = cudnnPointwiseMode_t(14);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_LOG: cudnnPointwiseMode_t = cudnnPointwiseMode_t(15);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_NEG: cudnnPointwiseMode_t = cudnnPointwiseMode_t(16);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_RSQRT: cudnnPointwiseMode_t = cudnnPointwiseMode_t(17);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_SIN: cudnnPointwiseMode_t = cudnnPointwiseMode_t(18);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_SQRT: cudnnPointwiseMode_t = cudnnPointwiseMode_t(4);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_TAN: cudnnPointwiseMode_t = cudnnPointwiseMode_t(19);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_ERF: cudnnPointwiseMode_t = cudnnPointwiseMode_t(20);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_IDENTITY: cudnnPointwiseMode_t = cudnnPointwiseMode_t(21);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_RECIPROCAL: cudnnPointwiseMode_t = cudnnPointwiseMode_t(22);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_ATAN2: cudnnPointwiseMode_t = cudnnPointwiseMode_t(23);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_RELU_FWD: cudnnPointwiseMode_t = cudnnPointwiseMode_t(100);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_TANH_FWD: cudnnPointwiseMode_t = cudnnPointwiseMode_t(101);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_SIGMOID_FWD: cudnnPointwiseMode_t = cudnnPointwiseMode_t(102);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_ELU_FWD: cudnnPointwiseMode_t = cudnnPointwiseMode_t(103);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_GELU_FWD: cudnnPointwiseMode_t = cudnnPointwiseMode_t(104);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_SOFTPLUS_FWD: cudnnPointwiseMode_t = cudnnPointwiseMode_t(105);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_SWISH_FWD: cudnnPointwiseMode_t = cudnnPointwiseMode_t(106);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_GELU_APPROX_TANH_FWD: cudnnPointwiseMode_t =
        cudnnPointwiseMode_t(107);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_RELU_BWD: cudnnPointwiseMode_t = cudnnPointwiseMode_t(200);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_TANH_BWD: cudnnPointwiseMode_t = cudnnPointwiseMode_t(201);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_SIGMOID_BWD: cudnnPointwiseMode_t = cudnnPointwiseMode_t(202);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_ELU_BWD: cudnnPointwiseMode_t = cudnnPointwiseMode_t(203);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_GELU_BWD: cudnnPointwiseMode_t = cudnnPointwiseMode_t(204);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_SOFTPLUS_BWD: cudnnPointwiseMode_t = cudnnPointwiseMode_t(205);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_SWISH_BWD: cudnnPointwiseMode_t = cudnnPointwiseMode_t(206);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_GELU_APPROX_TANH_BWD: cudnnPointwiseMode_t =
        cudnnPointwiseMode_t(207);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_CMP_EQ: cudnnPointwiseMode_t = cudnnPointwiseMode_t(300);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_CMP_NEQ: cudnnPointwiseMode_t = cudnnPointwiseMode_t(301);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_CMP_GT: cudnnPointwiseMode_t = cudnnPointwiseMode_t(302);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_CMP_GE: cudnnPointwiseMode_t = cudnnPointwiseMode_t(303);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_CMP_LT: cudnnPointwiseMode_t = cudnnPointwiseMode_t(304);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_CMP_LE: cudnnPointwiseMode_t = cudnnPointwiseMode_t(305);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_LOGICAL_AND: cudnnPointwiseMode_t = cudnnPointwiseMode_t(400);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_LOGICAL_OR: cudnnPointwiseMode_t = cudnnPointwiseMode_t(401);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_LOGICAL_NOT: cudnnPointwiseMode_t = cudnnPointwiseMode_t(402);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_GEN_INDEX: cudnnPointwiseMode_t = cudnnPointwiseMode_t(501);
}
impl cudnnPointwiseMode_t {
    pub const CUDNN_POINTWISE_BINARY_SELECT: cudnnPointwiseMode_t = cudnnPointwiseMode_t(601);
}
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct cudnnPointwiseMode_t(pub ::std::os::raw::c_uint);
