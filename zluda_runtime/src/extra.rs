impl cudaOutputMode {
    #[doc = "< Output mode Key-Value pair format."]
    pub const cudaKeyValuePair: cudaOutputMode = cudaOutputMode(0);
}
impl cudaOutputMode {
    #[doc = "< Output mode Comma separated values format."]
    pub const cudaCSV: cudaOutputMode = cudaOutputMode(1);
}
#[repr(transparent)]
#[doc = " CUDA Profiler Output modes"]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct cudaOutputMode(pub ::std::os::raw::c_int);
#[doc = " CUDA output file modes"]
pub use self::cudaOutputMode as cudaOutputMode_t;
