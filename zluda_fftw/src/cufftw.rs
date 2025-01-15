/* automatically generated by rust-bindgen 0.69.4 */

#[repr(C)]
#[derive(Copy, Clone)]
pub struct _iobuf {
    pub _Placeholder: *mut ::std::os::raw::c_void,
}
pub type FILE = _iobuf;
pub type fftw_complex = [f64; 2usize];
pub type fftwf_complex = [f32; 2usize];
pub type fftw_plan = *mut ::std::os::raw::c_void;
pub type fftwf_plan = *mut ::std::os::raw::c_void;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct fftw_iodim {
    pub n: ::std::os::raw::c_int,
    pub is: ::std::os::raw::c_int,
    pub os: ::std::os::raw::c_int,
}
pub type fftwf_iodim = fftw_iodim;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct fftw_iodim64 {
    pub n: isize,
    pub is: isize,
    pub os: isize,
}
pub type fftwf_iodim64 = fftw_iodim64;

#[no_mangle]
pub extern "system" fn fftw_plan_dft_1d(
    n: ::std::os::raw::c_int,
    in_: *mut fftw_complex,
    out: *mut fftw_complex,
    sign: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_dft_2d(
    n0: ::std::os::raw::c_int,
    n1: ::std::os::raw::c_int,
    in_: *mut fftw_complex,
    out: *mut fftw_complex,
    sign: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_dft_3d(
    n0: ::std::os::raw::c_int,
    n1: ::std::os::raw::c_int,
    n2: ::std::os::raw::c_int,
    in_: *mut fftw_complex,
    out: *mut fftw_complex,
    sign: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_dft(
    rank: ::std::os::raw::c_int,
    n: *const ::std::os::raw::c_int,
    in_: *mut fftw_complex,
    out: *mut fftw_complex,
    sign: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_dft_r2c_1d(
    n: ::std::os::raw::c_int,
    in_: *mut f64,
    out: *mut fftw_complex,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_dft_r2c_2d(
    n0: ::std::os::raw::c_int,
    n1: ::std::os::raw::c_int,
    in_: *mut f64,
    out: *mut fftw_complex,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_dft_r2c_3d(
    n0: ::std::os::raw::c_int,
    n1: ::std::os::raw::c_int,
    n2: ::std::os::raw::c_int,
    in_: *mut f64,
    out: *mut fftw_complex,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_dft_r2c(
    rank: ::std::os::raw::c_int,
    n: *const ::std::os::raw::c_int,
    in_: *mut f64,
    out: *mut fftw_complex,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_dft_c2r_1d(
    n: ::std::os::raw::c_int,
    in_: *mut fftw_complex,
    out: *mut f64,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_dft_c2r_2d(
    n0: ::std::os::raw::c_int,
    n1: ::std::os::raw::c_int,
    in_: *mut fftw_complex,
    out: *mut f64,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_dft_c2r_3d(
    n0: ::std::os::raw::c_int,
    n1: ::std::os::raw::c_int,
    n2: ::std::os::raw::c_int,
    in_: *mut fftw_complex,
    out: *mut f64,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_dft_c2r(
    rank: ::std::os::raw::c_int,
    n: *const ::std::os::raw::c_int,
    in_: *mut fftw_complex,
    out: *mut f64,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_many_dft(
    rank: ::std::os::raw::c_int,
    n: *const ::std::os::raw::c_int,
    batch: ::std::os::raw::c_int,
    in_: *mut fftw_complex,
    inembed: *const ::std::os::raw::c_int,
    istride: ::std::os::raw::c_int,
    idist: ::std::os::raw::c_int,
    out: *mut fftw_complex,
    onembed: *const ::std::os::raw::c_int,
    ostride: ::std::os::raw::c_int,
    odist: ::std::os::raw::c_int,
    sign: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_many_dft_r2c(
    rank: ::std::os::raw::c_int,
    n: *const ::std::os::raw::c_int,
    batch: ::std::os::raw::c_int,
    in_: *mut f64,
    inembed: *const ::std::os::raw::c_int,
    istride: ::std::os::raw::c_int,
    idist: ::std::os::raw::c_int,
    out: *mut fftw_complex,
    onembed: *const ::std::os::raw::c_int,
    ostride: ::std::os::raw::c_int,
    odist: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_many_dft_c2r(
    rank: ::std::os::raw::c_int,
    n: *const ::std::os::raw::c_int,
    batch: ::std::os::raw::c_int,
    in_: *mut fftw_complex,
    inembed: *const ::std::os::raw::c_int,
    istride: ::std::os::raw::c_int,
    idist: ::std::os::raw::c_int,
    out: *mut f64,
    onembed: *const ::std::os::raw::c_int,
    ostride: ::std::os::raw::c_int,
    odist: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_guru_dft(
    rank: ::std::os::raw::c_int,
    dims: *const fftw_iodim,
    batch_rank: ::std::os::raw::c_int,
    batch_dims: *const fftw_iodim,
    in_: *mut fftw_complex,
    out: *mut fftw_complex,
    sign: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_guru_dft_r2c(
    rank: ::std::os::raw::c_int,
    dims: *const fftw_iodim,
    batch_rank: ::std::os::raw::c_int,
    batch_dims: *const fftw_iodim,
    in_: *mut f64,
    out: *mut fftw_complex,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_guru_dft_c2r(
    rank: ::std::os::raw::c_int,
    dims: *const fftw_iodim,
    batch_rank: ::std::os::raw::c_int,
    batch_dims: *const fftw_iodim,
    in_: *mut fftw_complex,
    out: *mut f64,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_execute(plan: fftw_plan) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_execute_dft(
    plan: fftw_plan,
    idata: *mut fftw_complex,
    odata: *mut fftw_complex,
) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_execute_dft_r2c(
    plan: fftw_plan,
    idata: *mut f64,
    odata: *mut fftw_complex,
) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_execute_dft_c2r(
    plan: fftw_plan,
    idata: *mut fftw_complex,
    odata: *mut f64,
) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_dft_1d(
    n: ::std::os::raw::c_int,
    in_: *mut fftwf_complex,
    out: *mut fftwf_complex,
    sign: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_dft_2d(
    n0: ::std::os::raw::c_int,
    n1: ::std::os::raw::c_int,
    in_: *mut fftwf_complex,
    out: *mut fftwf_complex,
    sign: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_dft_3d(
    n0: ::std::os::raw::c_int,
    n1: ::std::os::raw::c_int,
    n2: ::std::os::raw::c_int,
    in_: *mut fftwf_complex,
    out: *mut fftwf_complex,
    sign: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_dft(
    rank: ::std::os::raw::c_int,
    n: *const ::std::os::raw::c_int,
    in_: *mut fftwf_complex,
    out: *mut fftwf_complex,
    sign: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_dft_r2c_1d(
    n: ::std::os::raw::c_int,
    in_: *mut f32,
    out: *mut fftwf_complex,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_dft_r2c_2d(
    n0: ::std::os::raw::c_int,
    n1: ::std::os::raw::c_int,
    in_: *mut f32,
    out: *mut fftwf_complex,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_dft_r2c_3d(
    n0: ::std::os::raw::c_int,
    n1: ::std::os::raw::c_int,
    n2: ::std::os::raw::c_int,
    in_: *mut f32,
    out: *mut fftwf_complex,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_dft_r2c(
    rank: ::std::os::raw::c_int,
    n: *const ::std::os::raw::c_int,
    in_: *mut f32,
    out: *mut fftwf_complex,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_dft_c2r_1d(
    n: ::std::os::raw::c_int,
    in_: *mut fftwf_complex,
    out: *mut f32,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_dft_c2r_2d(
    n0: ::std::os::raw::c_int,
    n1: ::std::os::raw::c_int,
    in_: *mut fftwf_complex,
    out: *mut f32,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_dft_c2r_3d(
    n0: ::std::os::raw::c_int,
    n1: ::std::os::raw::c_int,
    n2: ::std::os::raw::c_int,
    in_: *mut fftwf_complex,
    out: *mut f32,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_dft_c2r(
    rank: ::std::os::raw::c_int,
    n: *const ::std::os::raw::c_int,
    in_: *mut fftwf_complex,
    out: *mut f32,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_many_dft(
    rank: ::std::os::raw::c_int,
    n: *const ::std::os::raw::c_int,
    batch: ::std::os::raw::c_int,
    in_: *mut fftwf_complex,
    inembed: *const ::std::os::raw::c_int,
    istride: ::std::os::raw::c_int,
    idist: ::std::os::raw::c_int,
    out: *mut fftwf_complex,
    onembed: *const ::std::os::raw::c_int,
    ostride: ::std::os::raw::c_int,
    odist: ::std::os::raw::c_int,
    sign: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_many_dft_r2c(
    rank: ::std::os::raw::c_int,
    n: *const ::std::os::raw::c_int,
    batch: ::std::os::raw::c_int,
    in_: *mut f32,
    inembed: *const ::std::os::raw::c_int,
    istride: ::std::os::raw::c_int,
    idist: ::std::os::raw::c_int,
    out: *mut fftwf_complex,
    onembed: *const ::std::os::raw::c_int,
    ostride: ::std::os::raw::c_int,
    odist: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_many_dft_c2r(
    rank: ::std::os::raw::c_int,
    n: *const ::std::os::raw::c_int,
    batch: ::std::os::raw::c_int,
    in_: *mut fftwf_complex,
    inembed: *const ::std::os::raw::c_int,
    istride: ::std::os::raw::c_int,
    idist: ::std::os::raw::c_int,
    out: *mut f32,
    onembed: *const ::std::os::raw::c_int,
    ostride: ::std::os::raw::c_int,
    odist: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_guru_dft(
    rank: ::std::os::raw::c_int,
    dims: *const fftwf_iodim,
    batch_rank: ::std::os::raw::c_int,
    batch_dims: *const fftwf_iodim,
    in_: *mut fftwf_complex,
    out: *mut fftwf_complex,
    sign: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_guru_dft_r2c(
    rank: ::std::os::raw::c_int,
    dims: *const fftwf_iodim,
    batch_rank: ::std::os::raw::c_int,
    batch_dims: *const fftwf_iodim,
    in_: *mut f32,
    out: *mut fftwf_complex,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_guru_dft_c2r(
    rank: ::std::os::raw::c_int,
    dims: *const fftwf_iodim,
    batch_rank: ::std::os::raw::c_int,
    batch_dims: *const fftwf_iodim,
    in_: *mut fftwf_complex,
    out: *mut f32,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_execute(plan: fftw_plan) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_execute_dft(
    plan: fftwf_plan,
    idata: *mut fftwf_complex,
    odata: *mut fftwf_complex,
) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_execute_dft_r2c(
    plan: fftwf_plan,
    idata: *mut f32,
    odata: *mut fftwf_complex,
) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_execute_dft_c2r(
    plan: fftwf_plan,
    idata: *mut fftwf_complex,
    odata: *mut f32,
) {
    unimplemented!()
}

#[doc = " CUFFTW 64-bit Guru Interface\n dp"]
#[no_mangle]
pub extern "system" fn fftw_plan_guru64_dft(
    rank: ::std::os::raw::c_int,
    dims: *const fftw_iodim64,
    batch_rank: ::std::os::raw::c_int,
    batch_dims: *const fftw_iodim64,
    in_: *mut fftw_complex,
    out: *mut fftw_complex,
    sign: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_guru64_dft_r2c(
    rank: ::std::os::raw::c_int,
    dims: *const fftw_iodim64,
    batch_rank: ::std::os::raw::c_int,
    batch_dims: *const fftw_iodim64,
    in_: *mut f64,
    out: *mut fftw_complex,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_plan_guru64_dft_c2r(
    rank: ::std::os::raw::c_int,
    dims: *const fftw_iodim64,
    batch_rank: ::std::os::raw::c_int,
    batch_dims: *const fftw_iodim64,
    in_: *mut fftw_complex,
    out: *mut f64,
    flags: ::std::os::raw::c_uint,
) -> fftw_plan {
    unimplemented!()
}

#[doc = " sp"]
#[no_mangle]
pub extern "system" fn fftwf_plan_guru64_dft(
    rank: ::std::os::raw::c_int,
    dims: *const fftwf_iodim64,
    batch_rank: ::std::os::raw::c_int,
    batch_dims: *const fftwf_iodim64,
    in_: *mut fftwf_complex,
    out: *mut fftwf_complex,
    sign: ::std::os::raw::c_int,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_guru64_dft_r2c(
    rank: ::std::os::raw::c_int,
    dims: *const fftwf_iodim64,
    batch_rank: ::std::os::raw::c_int,
    batch_dims: *const fftwf_iodim64,
    in_: *mut f32,
    out: *mut fftwf_complex,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_plan_guru64_dft_c2r(
    rank: ::std::os::raw::c_int,
    dims: *const fftwf_iodim64,
    batch_rank: ::std::os::raw::c_int,
    batch_dims: *const fftwf_iodim64,
    in_: *mut fftwf_complex,
    out: *mut f32,
    flags: ::std::os::raw::c_uint,
) -> fftwf_plan {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_malloc(n: usize) -> *mut ::std::os::raw::c_void {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_malloc(n: usize) -> *mut ::std::os::raw::c_void {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_free(pointer: *mut ::std::os::raw::c_void) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_free(pointer: *mut ::std::os::raw::c_void) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_export_wisdom_to_file(output_file: *mut FILE) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_export_wisdom_to_file(output_file: *mut FILE) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_import_wisdom_from_file(input_file: *mut FILE) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_import_wisdom_from_file(input_file: *mut FILE) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_print_plan(plan: fftw_plan) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_print_plan(plan: fftwf_plan) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_set_timelimit(seconds: f64) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_set_timelimit(seconds: f64) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_cost(plan: fftw_plan) -> f64 {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_cost(plan: fftw_plan) -> f64 {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_flops(plan: fftw_plan, add: *mut f64, mul: *mut f64, fma: *mut f64) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_flops(plan: fftw_plan, add: *mut f64, mul: *mut f64, fma: *mut f64) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_destroy_plan(plan: fftw_plan) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_destroy_plan(plan: fftwf_plan) {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftw_cleanup() {
    unimplemented!()
}

#[no_mangle]
pub extern "system" fn fftwf_cleanup() {
    unimplemented!()
}
