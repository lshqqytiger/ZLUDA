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

impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_ADD: miopenPointwiseMode_t = miopenPointwiseMode_t(0);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_ADD_SQUARE: miopenPointwiseMode_t = miopenPointwiseMode_t(1);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_DIV: miopenPointwiseMode_t = miopenPointwiseMode_t(2);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_MAX: miopenPointwiseMode_t = miopenPointwiseMode_t(3);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_MIN: miopenPointwiseMode_t = miopenPointwiseMode_t(4);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_MOD: miopenPointwiseMode_t = miopenPointwiseMode_t(5);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_MUL: miopenPointwiseMode_t = miopenPointwiseMode_t(6);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_POW: miopenPointwiseMode_t = miopenPointwiseMode_t(7);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_SUB: miopenPointwiseMode_t = miopenPointwiseMode_t(8);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_ABS: miopenPointwiseMode_t = miopenPointwiseMode_t(9);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_CEIL: miopenPointwiseMode_t = miopenPointwiseMode_t(10);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_COS: miopenPointwiseMode_t = miopenPointwiseMode_t(11);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_EXP: miopenPointwiseMode_t = miopenPointwiseMode_t(12);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_FLOOR: miopenPointwiseMode_t = miopenPointwiseMode_t(13);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_LOG: miopenPointwiseMode_t = miopenPointwiseMode_t(14);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_NEG: miopenPointwiseMode_t = miopenPointwiseMode_t(15);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_RSQRT: miopenPointwiseMode_t = miopenPointwiseMode_t(16);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_SIN: miopenPointwiseMode_t = miopenPointwiseMode_t(17);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_SQRT: miopenPointwiseMode_t = miopenPointwiseMode_t(18);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_TAN: miopenPointwiseMode_t = miopenPointwiseMode_t(19);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_ERF: miopenPointwiseMode_t = miopenPointwiseMode_t(20);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_IDENTITY: miopenPointwiseMode_t = miopenPointwiseMode_t(21);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_RELU_FWD: miopenPointwiseMode_t = miopenPointwiseMode_t(22);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_TANH_FWD: miopenPointwiseMode_t = miopenPointwiseMode_t(23);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_SIGMOID_FWD: miopenPointwiseMode_t = miopenPointwiseMode_t(24);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_ELU_FWD: miopenPointwiseMode_t = miopenPointwiseMode_t(25);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_GELU_FWD: miopenPointwiseMode_t = miopenPointwiseMode_t(26);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_SOFTPLUS_FWD: miopenPointwiseMode_t = miopenPointwiseMode_t(27);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_SWISH_FWD: miopenPointwiseMode_t = miopenPointwiseMode_t(28);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_GELU_APPROX_TANH_FWD: miopenPointwiseMode_t =
        miopenPointwiseMode_t(29);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_RELU_BWD: miopenPointwiseMode_t = miopenPointwiseMode_t(30);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_TANH_BWD: miopenPointwiseMode_t = miopenPointwiseMode_t(31);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_SIGMOID_BWD: miopenPointwiseMode_t = miopenPointwiseMode_t(32);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_ELU_BWD: miopenPointwiseMode_t = miopenPointwiseMode_t(33);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_GELU_BWD: miopenPointwiseMode_t = miopenPointwiseMode_t(34);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_SOFTPLUS_BWD: miopenPointwiseMode_t = miopenPointwiseMode_t(35);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_SWISH_BWD: miopenPointwiseMode_t = miopenPointwiseMode_t(36);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_GELU_APPROX_TANH_BWD: miopenPointwiseMode_t =
        miopenPointwiseMode_t(37);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_CMP_EQ: miopenPointwiseMode_t = miopenPointwiseMode_t(38);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_CMP_NEQ: miopenPointwiseMode_t = miopenPointwiseMode_t(39);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_CMP_GT: miopenPointwiseMode_t = miopenPointwiseMode_t(40);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_CMP_GE: miopenPointwiseMode_t = miopenPointwiseMode_t(41);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_CMP_LT: miopenPointwiseMode_t = miopenPointwiseMode_t(42);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_CMP_LE: miopenPointwiseMode_t = miopenPointwiseMode_t(43);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_LOGICAL_AND: miopenPointwiseMode_t = miopenPointwiseMode_t(44);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_LOGICAL_OR: miopenPointwiseMode_t = miopenPointwiseMode_t(45);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_LOGICAL_NOT: miopenPointwiseMode_t = miopenPointwiseMode_t(46);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_GEN_INDEX: miopenPointwiseMode_t = miopenPointwiseMode_t(47);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_BINARY_SELECT: miopenPointwiseMode_t = miopenPointwiseMode_t(48);
}
impl miopenPointwiseMode_t {
    pub const MIOPEN_POINTWISE_RECIPROCAL: miopenPointwiseMode_t = miopenPointwiseMode_t(49);
}
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct miopenPointwiseMode_t(pub ::std::os::raw::c_uint);
