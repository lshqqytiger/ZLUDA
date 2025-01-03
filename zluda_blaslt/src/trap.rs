#[macro_export]
macro_rules! install_trap {
    ($name:ident) => {
        #[no_mangle]
        pub extern "system" fn $name() -> cublasStatus_t {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS
        }
    };
}