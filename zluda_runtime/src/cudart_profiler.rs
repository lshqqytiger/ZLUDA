#[no_mangle]
pub extern "system" fn cudaProfilerInitialize(
    configFile: *const ::std::os::raw::c_char,
    outputFile: *const ::std::os::raw::c_char,
    outputMode: cudaOutputMode_t,
) -> cudaError_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cudaProfilerStart() -> cudaError_t {
    crate::unsupported()
}

#[no_mangle]
pub extern "system" fn cudaProfilerStop() -> cudaError_t {
    crate::unsupported()
}
