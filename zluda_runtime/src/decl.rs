#[macro_export]
macro_rules! decl {
    ($name:ident) => {
        #[no_mangle]
        pub extern "system" fn $name() -> cudaError_t {
            unimplemented!()
        }
    };
}
