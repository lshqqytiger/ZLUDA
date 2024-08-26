#![allow(warnings)]
#[cfg(feature = "rocm5")]
pub mod hip_runtime_api_v5;
#[cfg(not(feature = "rocm5"))]
pub mod hip_runtime_api_v6;
pub mod hip_runtime_api_ext {
    use crate::hipStream_t;

    pub const hipStreamNull: hipStream_t = 0 as _;
    pub const hipStreamPerThread: hipStream_t = 2 as _;
    pub const HIP_TRSA_OVERRIDE_FORMAT: u32 = 1;
}
#[cfg(feature = "rocm5")]
pub use hip_runtime_api_v5::*;
#[cfg(not(feature = "rocm5"))]
pub use hip_runtime_api_v6::*;
pub use hip_runtime_api_ext::*;

#[macro_export]
#[cfg(feature = "rocm5")]
macro_rules! hipGetDeviceProperties {
    ($prop:expr, $id:expr) => {
        hipGetDeviceProperties($prop, $id)
    };
}
#[macro_export]
#[cfg(not(feature = "rocm5"))]
macro_rules! hipGetDeviceProperties {
    ($prop:expr, $id:expr) => {
        hipGetDevicePropertiesR0600($prop, $id)
    };
}
