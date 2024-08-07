/* automatically generated by rust-bindgen 0.69.4 */

#[repr(C)]
#[derive(Copy, Clone)]
pub struct float2 {
    pub x: f32,
    pub y: f32,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct double2 {
    pub x: f64,
    pub y: f64,
}
pub type hipFloatComplex = float2;
pub type hipDoubleComplex = double2;
pub type hipComplex = hipFloatComplex;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ihipStream_t {
    _unused: [u8; 0],
}
pub type hipStream_t = *mut ihipStream_t;
impl hipfftResult_t {
    #[doc = " hipFFT operation was successful"]
    pub const HIPFFT_SUCCESS: hipfftResult_t = hipfftResult_t(0);
}
impl hipfftResult_t {
    #[doc = " hipFFT was passed an invalid plan handle"]
    pub const HIPFFT_INVALID_PLAN: hipfftResult_t = hipfftResult_t(1);
}
impl hipfftResult_t {
    #[doc = " hipFFT failed to allocate GPU or CPU memory"]
    pub const HIPFFT_ALLOC_FAILED: hipfftResult_t = hipfftResult_t(2);
}
impl hipfftResult_t {
    #[doc = " No longer used"]
    pub const HIPFFT_INVALID_TYPE: hipfftResult_t = hipfftResult_t(3);
}
impl hipfftResult_t {
    #[doc = " User specified an invalid pointer or parameter"]
    pub const HIPFFT_INVALID_VALUE: hipfftResult_t = hipfftResult_t(4);
}
impl hipfftResult_t {
    #[doc = " Driver or internal hipFFT library error"]
    pub const HIPFFT_INTERNAL_ERROR: hipfftResult_t = hipfftResult_t(5);
}
impl hipfftResult_t {
    #[doc = " Failed to execute an FFT on the GPU"]
    pub const HIPFFT_EXEC_FAILED: hipfftResult_t = hipfftResult_t(6);
}
impl hipfftResult_t {
    #[doc = " hipFFT failed to initialize"]
    pub const HIPFFT_SETUP_FAILED: hipfftResult_t = hipfftResult_t(7);
}
impl hipfftResult_t {
    #[doc = " User specified an invalid transform size"]
    pub const HIPFFT_INVALID_SIZE: hipfftResult_t = hipfftResult_t(8);
}
impl hipfftResult_t {
    #[doc = " No longer used"]
    pub const HIPFFT_UNALIGNED_DATA: hipfftResult_t = hipfftResult_t(9);
}
impl hipfftResult_t {
    #[doc = " Missing parameters in call"]
    pub const HIPFFT_INCOMPLETE_PARAMETER_LIST: hipfftResult_t = hipfftResult_t(10);
}
impl hipfftResult_t {
    #[doc = " Execution of a plan was on different GPU than plan creation"]
    pub const HIPFFT_INVALID_DEVICE: hipfftResult_t = hipfftResult_t(11);
}
impl hipfftResult_t {
    #[doc = " Internal plan database error"]
    pub const HIPFFT_PARSE_ERROR: hipfftResult_t = hipfftResult_t(12);
}
impl hipfftResult_t {
    #[doc = " No workspace has been provided prior to plan execution"]
    pub const HIPFFT_NO_WORKSPACE: hipfftResult_t = hipfftResult_t(13);
}
impl hipfftResult_t {
    #[doc = " Function does not implement functionality for parameters given."]
    pub const HIPFFT_NOT_IMPLEMENTED: hipfftResult_t = hipfftResult_t(14);
}
impl hipfftResult_t {
    #[doc = " Operation is not supported for parameters given."]
    pub const HIPFFT_NOT_SUPPORTED: hipfftResult_t = hipfftResult_t(16);
}
#[repr(transparent)]
#[doc = " @brief Result/status/error codes"]
#[must_use]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipfftResult_t(pub ::std::os::raw::c_int);
#[doc = " @brief Result/status/error codes"]
pub use self::hipfftResult_t as hipfftResult;
impl hipfftType_t {
    #[doc = " Real to complex (interleaved)"]
    pub const HIPFFT_R2C: hipfftType_t = hipfftType_t(42);
}
impl hipfftType_t {
    #[doc = " Complex (interleaved) to real"]
    pub const HIPFFT_C2R: hipfftType_t = hipfftType_t(44);
}
impl hipfftType_t {
    #[doc = " Complex to complex (interleaved)"]
    pub const HIPFFT_C2C: hipfftType_t = hipfftType_t(41);
}
impl hipfftType_t {
    #[doc = " Double to double-complex (interleaved)"]
    pub const HIPFFT_D2Z: hipfftType_t = hipfftType_t(106);
}
impl hipfftType_t {
    #[doc = " Double-complex (interleaved) to double"]
    pub const HIPFFT_Z2D: hipfftType_t = hipfftType_t(108);
}
impl hipfftType_t {
    #[doc = " Double-complex to double-complex (interleaved)"]
    pub const HIPFFT_Z2Z: hipfftType_t = hipfftType_t(105);
}
#[repr(transparent)]
#[doc = " @brief Transform type\n  @details This type is used to declare the Fourier transform type that will be executed."]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipfftType_t(pub ::std::os::raw::c_int);
#[doc = " @brief Transform type\n  @details This type is used to declare the Fourier transform type that will be executed."]
pub use self::hipfftType_t as hipfftType;
impl hipfftLibraryPropertyType_t {
    pub const HIPFFT_MAJOR_VERSION: hipfftLibraryPropertyType_t = hipfftLibraryPropertyType_t(0);
}
impl hipfftLibraryPropertyType_t {
    pub const HIPFFT_MINOR_VERSION: hipfftLibraryPropertyType_t = hipfftLibraryPropertyType_t(1);
}
impl hipfftLibraryPropertyType_t {
    pub const HIPFFT_PATCH_LEVEL: hipfftLibraryPropertyType_t = hipfftLibraryPropertyType_t(2);
}
#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct hipfftLibraryPropertyType_t(pub ::std::os::raw::c_int);
pub use self::hipfftLibraryPropertyType_t as hipfftLibraryPropertyType;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct hipfftHandle_t {
    _unused: [u8; 0],
}
pub type hipfftHandle = *mut hipfftHandle_t;
pub type hipfftComplex = hipComplex;
pub type hipfftDoubleComplex = hipDoubleComplex;
pub type hipfftReal = f32;
pub type hipfftDoubleReal = f64;
extern "C" {
    #[doc = " @brief Create a new one-dimensional FFT plan.\n\n  @details Allocate and initialize a new one-dimensional FFT plan.\n\n  @param[out] plan Pointer to the FFT plan handle.\n  @param[in] nx FFT length.\n  @param[in] type FFT type.\n  @param[in] batch Number of batched transforms to compute."]
    pub fn hipfftPlan1d(
        plan: *mut hipfftHandle,
        nx: ::std::os::raw::c_int,
        type_: hipfftType,
        batch: ::std::os::raw::c_int,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Create a new two-dimensional FFT plan.\n\n  @details Allocate and initialize a new two-dimensional FFT plan.\n  Two-dimensional data should be stored in C ordering (row-major\n  format), so that indexes in y-direction (j index) vary the\n  fastest.\n\n  @param[out] plan Pointer to the FFT plan handle.\n  @param[in] nx Number of elements in the x-direction (slow index).\n  @param[in] ny Number of elements in the y-direction (fast index).\n  @param[in] type FFT type."]
    pub fn hipfftPlan2d(
        plan: *mut hipfftHandle,
        nx: ::std::os::raw::c_int,
        ny: ::std::os::raw::c_int,
        type_: hipfftType,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Create a new three-dimensional FFT plan.\n\n  @details Allocate and initialize a new three-dimensional FFT plan.\n  Three-dimensional data should be stored in C ordering (row-major\n  format), so that indexes in z-direction (k index) vary the\n  fastest.\n\n  @param[out] plan Pointer to the FFT plan handle.\n  @param[in] nx Number of elements in the x-direction (slowest index).\n  @param[in] ny Number of elements in the y-direction.\n  @param[in] nz Number of elements in the z-direction (fastest index).\n  @param[in] type FFT type."]
    pub fn hipfftPlan3d(
        plan: *mut hipfftHandle,
        nx: ::std::os::raw::c_int,
        ny: ::std::os::raw::c_int,
        nz: ::std::os::raw::c_int,
        type_: hipfftType,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Create a new batched rank-dimensional FFT plan with advanced data layout.\n\n @details Allocate and initialize a new batched rank-dimensional\n  FFT plan. The number of elements to transform in each direction of\n  the input data is specified in n.\n\n  The batch parameter tells hipFFT how many transforms to perform.\n  The distance between the first elements of two consecutive batches\n  of the input and output data are specified with the idist and odist\n  parameters.\n\n  The inembed and onembed parameters define the input and output data\n  layouts. The number of elements in the data is assumed to be larger\n  than the number of elements in the transform. Strided data layouts\n  are also supported. Strides along the fastest direction in the input\n  and output data are specified via the istride and ostride parameters.\n\n  If both inembed and onembed parameters are set to NULL, all the\n  advanced data layout parameters are ignored and reverted to default\n  values, i.e., the batched transform is performed with non-strided data\n  access and the number of data/transform elements are assumed to be\n  equivalent.\n\n  @param[out] plan Pointer to the FFT plan handle.\n  @param[in] rank Dimension of transform (1, 2, or 3).\n  @param[in] n Number of elements to transform in the x/y/z directions.\n  @param[in] inembed Number of elements in the input data in the x/y/z directions.\n  @param[in] istride Distance between two successive elements in the input data.\n  @param[in] idist Distance between input batches.\n  @param[in] onembed Number of elements in the output data in the x/y/z directions.\n  @param[in] ostride Distance between two successive elements in the output data.\n  @param[in] odist Distance between output batches.\n  @param[in] type FFT type.\n  @param[in] batch Number of batched transforms to perform."]
    pub fn hipfftPlanMany(
        plan: *mut hipfftHandle,
        rank: ::std::os::raw::c_int,
        n: *mut ::std::os::raw::c_int,
        inembed: *mut ::std::os::raw::c_int,
        istride: ::std::os::raw::c_int,
        idist: ::std::os::raw::c_int,
        onembed: *mut ::std::os::raw::c_int,
        ostride: ::std::os::raw::c_int,
        odist: ::std::os::raw::c_int,
        type_: hipfftType,
        batch: ::std::os::raw::c_int,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Allocate a new plan."]
    pub fn hipfftCreate(plan: *mut hipfftHandle) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Set scaling factor.\n\n  @details hipFFT multiplies each element of the result by the given factor at the end of the transform.\n\n  The supplied factor must be a finite number.  That is, it must neither be infinity nor NaN.\n\n  This function must be called after the plan is allocated using\n  ::hipfftCreate, but before the plan is initialized by any of the\n  \"MakePlan\" functions.  Therefore, API functions that combine\n  creation and initialization (::hipfftPlan1d, ::hipfftPlan2d,\n  ::hipfftPlan3d, and ::hipfftPlanMany) cannot set a scale factor.\n\n  Note that the scale factor applies to both forward and\n  backward transforms executed with the specified plan handle."]
    pub fn hipfftExtPlanScaleFactor(plan: hipfftHandle, scalefactor: f64) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Initialize a new one-dimensional FFT plan.\n\n  @details Assumes that the plan has been created already, and\n  modifies the plan associated with the plan handle.\n\n  @param[in] plan Handle of the FFT plan.\n  @param[in] nx FFT length.\n  @param[in] type FFT type.\n  @param[in] batch Number of batched transforms to compute.\n  @param[out] workSize Pointer to work area size (returned value)."]
    pub fn hipfftMakePlan1d(
        plan: hipfftHandle,
        nx: ::std::os::raw::c_int,
        type_: hipfftType,
        batch: ::std::os::raw::c_int,
        workSize: *mut usize,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Initialize a new two-dimensional FFT plan.\n\n  @details Assumes that the plan has been created already, and\n  modifies the plan associated with the plan handle.\n  Two-dimensional data should be stored in C ordering (row-major\n  format), so that indexes in y-direction (j index) vary the\n  fastest.\n\n  @param[in] plan Handle of the FFT plan.\n  @param[in] nx Number of elements in the x-direction (slow index).\n  @param[in] ny Number of elements in the y-direction (fast index).\n  @param[in] type FFT type.\n  @param[out] workSize Pointer to work area size (returned value)."]
    pub fn hipfftMakePlan2d(
        plan: hipfftHandle,
        nx: ::std::os::raw::c_int,
        ny: ::std::os::raw::c_int,
        type_: hipfftType,
        workSize: *mut usize,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Initialize a new two-dimensional FFT plan.\n\n  @details Assumes that the plan has been created already, and\n  modifies the plan associated with the plan handle.\n  Three-dimensional data should be stored in C ordering (row-major\n  format), so that indexes in z-direction (k index) vary the\n  fastest.\n\n  @param[in] plan Handle of the FFT plan.\n  @param[in] nx Number of elements in the x-direction (slowest index).\n  @param[in] ny Number of elements in the y-direction.\n  @param[in] nz Number of elements in the z-direction (fastest index).\n  @param[in] type FFT type.\n  @param[out] workSize Pointer to work area size (returned value)."]
    pub fn hipfftMakePlan3d(
        plan: hipfftHandle,
        nx: ::std::os::raw::c_int,
        ny: ::std::os::raw::c_int,
        nz: ::std::os::raw::c_int,
        type_: hipfftType,
        workSize: *mut usize,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Initialize a new batched rank-dimensional FFT plan with advanced data layout.\n\n  @details Assumes that the plan has been created already, and\n  modifies the plan associated with the plan handle. The number\n  of elements to transform in each direction of the input data\n  in the FFT plan is specified in n.\n\n  The batch parameter tells hipFFT how many transforms to perform.\n  The distance between the first elements of two consecutive batches\n  of the input and output data are specified with the idist and odist\n  parameters.\n\n  The inembed and onembed parameters define the input and output data\n  layouts. The number of elements in the data is assumed to be larger\n  than the number of elements in the transform. Strided data layouts\n  are also supported. Strides along the fastest direction in the input\n  and output data are specified via the istride and ostride parameters.\n\n  If both inembed and onembed parameters are set to NULL, all the\n  advanced data layout parameters are ignored and reverted to default\n  values, i.e., the batched transform is performed with non-strided data\n  access and the number of data/transform elements are assumed to be\n  equivalent.\n\n  @param[out] plan Pointer to the FFT plan handle.\n  @param[in] rank Dimension of transform (1, 2, or 3).\n  @param[in] n Number of elements to transform in the x/y/z directions.\n  @param[in] inembed Number of elements in the input data in the x/y/z directions.\n  @param[in] istride Distance between two successive elements in the input data.\n  @param[in] idist Distance between input batches.\n  @param[in] onembed Number of elements in the output data in the x/y/z directions.\n  @param[in] ostride Distance between two successive elements in the output data.\n  @param[in] odist Distance between output batches.\n  @param[in] type FFT type.\n  @param[in] batch Number of batched transforms to perform.\n  @param[out] workSize Pointer to work area size (returned value)."]
    pub fn hipfftMakePlanMany(
        plan: hipfftHandle,
        rank: ::std::os::raw::c_int,
        n: *mut ::std::os::raw::c_int,
        inembed: *mut ::std::os::raw::c_int,
        istride: ::std::os::raw::c_int,
        idist: ::std::os::raw::c_int,
        onembed: *mut ::std::os::raw::c_int,
        ostride: ::std::os::raw::c_int,
        odist: ::std::os::raw::c_int,
        type_: hipfftType,
        batch: ::std::os::raw::c_int,
        workSize: *mut usize,
    ) -> hipfftResult;
}
extern "C" {
    pub fn hipfftMakePlanMany64(
        plan: hipfftHandle,
        rank: ::std::os::raw::c_int,
        n: *mut ::std::os::raw::c_longlong,
        inembed: *mut ::std::os::raw::c_longlong,
        istride: ::std::os::raw::c_longlong,
        idist: ::std::os::raw::c_longlong,
        onembed: *mut ::std::os::raw::c_longlong,
        ostride: ::std::os::raw::c_longlong,
        odist: ::std::os::raw::c_longlong,
        type_: hipfftType,
        batch: ::std::os::raw::c_longlong,
        workSize: *mut usize,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Return an estimate of the work area size required for a 1D plan.\n\n  @param[in] nx Number of elements in the x-direction.\n  @param[in] type FFT type.\n  @param[in] batch Number of batched transforms to perform.\n  @param[out] workSize Pointer to work area size (returned value)."]
    pub fn hipfftEstimate1d(
        nx: ::std::os::raw::c_int,
        type_: hipfftType,
        batch: ::std::os::raw::c_int,
        workSize: *mut usize,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Return an estimate of the work area size required for a 2D plan.\n\n  @param[in] nx Number of elements in the x-direction.\n  @param[in] ny Number of elements in the y-direction.\n  @param[in] type FFT type.\n  @param[out] workSize Pointer to work area size (returned value)."]
    pub fn hipfftEstimate2d(
        nx: ::std::os::raw::c_int,
        ny: ::std::os::raw::c_int,
        type_: hipfftType,
        workSize: *mut usize,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Return an estimate of the work area size required for a 3D plan.\n\n  @param[in] nx Number of elements in the x-direction.\n  @param[in] ny Number of elements in the y-direction.\n  @param[in] nz Number of elements in the z-direction.\n  @param[in] type FFT type.\n  @param[out] workSize Pointer to work area size (returned value)."]
    pub fn hipfftEstimate3d(
        nx: ::std::os::raw::c_int,
        ny: ::std::os::raw::c_int,
        nz: ::std::os::raw::c_int,
        type_: hipfftType,
        workSize: *mut usize,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Return an estimate of the work area size required for a rank-dimensional plan.\n\n  @param[in] rank Dimension of FFT transform (1, 2, or 3).\n  @param[in] n Number of elements in the x/y/z directions.\n  @param[in] inembed\n  @param[in] istride\n  @param[in] idist Distance between input batches.\n  @param[in] onembed\n  @param[in] ostride\n  @param[in] odist Distance between output batches.\n  @param[in] type FFT type.\n  @param[in] batch Number of batched transforms to perform.\n  @param[out] workSize Pointer to work area size (returned value)."]
    pub fn hipfftEstimateMany(
        rank: ::std::os::raw::c_int,
        n: *mut ::std::os::raw::c_int,
        inembed: *mut ::std::os::raw::c_int,
        istride: ::std::os::raw::c_int,
        idist: ::std::os::raw::c_int,
        onembed: *mut ::std::os::raw::c_int,
        ostride: ::std::os::raw::c_int,
        odist: ::std::os::raw::c_int,
        type_: hipfftType,
        batch: ::std::os::raw::c_int,
        workSize: *mut usize,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Return size of the work area size required for a 1D plan.\n\n  @param[in] plan Pointer to the FFT plan.\n  @param[in] nx Number of elements in the x-direction.\n  @param[in] type FFT type.\n  @param[in] batch Number of batched transforms to perform.\n  @param[out] workSize Pointer to work area size (returned value)."]
    pub fn hipfftGetSize1d(
        plan: hipfftHandle,
        nx: ::std::os::raw::c_int,
        type_: hipfftType,
        batch: ::std::os::raw::c_int,
        workSize: *mut usize,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Return size of the work area size required for a 2D plan.\n\n  @param[in] plan Pointer to the FFT plan.\n  @param[in] nx Number of elements in the x-direction.\n  @param[in] ny Number of elements in the y-direction.\n  @param[in] type FFT type.\n  @param[out] workSize Pointer to work area size (returned value)."]
    pub fn hipfftGetSize2d(
        plan: hipfftHandle,
        nx: ::std::os::raw::c_int,
        ny: ::std::os::raw::c_int,
        type_: hipfftType,
        workSize: *mut usize,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Return size of the work area size required for a 3D plan.\n\n  @param[in] plan Pointer to the FFT plan.\n  @param[in] nx Number of elements in the x-direction.\n  @param[in] ny Number of elements in the y-direction.\n  @param[in] nz Number of elements in the z-direction.\n  @param[in] type FFT type.\n  @param[out] workSize Pointer to work area size (returned value)."]
    pub fn hipfftGetSize3d(
        plan: hipfftHandle,
        nx: ::std::os::raw::c_int,
        ny: ::std::os::raw::c_int,
        nz: ::std::os::raw::c_int,
        type_: hipfftType,
        workSize: *mut usize,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Return size of the work area size required for a rank-dimensional plan.\n\n  @param[in] plan Pointer to the FFT plan.\n  @param[in] rank Dimension of FFT transform (1, 2, or 3).\n  @param[in] n Number of elements in the x/y/z directions.\n  @param[in] inembed\n  @param[in] istride\n  @param[in] idist Distance between input batches.\n  @param[in] onembed\n  @param[in] ostride\n  @param[in] odist Distance between output batches.\n  @param[in] type FFT type.\n  @param[in] batch Number of batched transforms to perform.\n  @param[out] workSize Pointer to work area size (returned value)."]
    pub fn hipfftGetSizeMany(
        plan: hipfftHandle,
        rank: ::std::os::raw::c_int,
        n: *mut ::std::os::raw::c_int,
        inembed: *mut ::std::os::raw::c_int,
        istride: ::std::os::raw::c_int,
        idist: ::std::os::raw::c_int,
        onembed: *mut ::std::os::raw::c_int,
        ostride: ::std::os::raw::c_int,
        odist: ::std::os::raw::c_int,
        type_: hipfftType,
        batch: ::std::os::raw::c_int,
        workSize: *mut usize,
    ) -> hipfftResult;
}
extern "C" {
    pub fn hipfftGetSizeMany64(
        plan: hipfftHandle,
        rank: ::std::os::raw::c_int,
        n: *mut ::std::os::raw::c_longlong,
        inembed: *mut ::std::os::raw::c_longlong,
        istride: ::std::os::raw::c_longlong,
        idist: ::std::os::raw::c_longlong,
        onembed: *mut ::std::os::raw::c_longlong,
        ostride: ::std::os::raw::c_longlong,
        odist: ::std::os::raw::c_longlong,
        type_: hipfftType,
        batch: ::std::os::raw::c_longlong,
        workSize: *mut usize,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Return size of the work area size required for a rank-dimensional plan.\n\n  @param[in] plan Pointer to the FFT plan.\n  @param[out] workSize Pointer to work area size (returned value)."]
    pub fn hipfftGetSize(plan: hipfftHandle, workSize: *mut usize) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Set the plan's auto-allocation flag.  The plan will allocate its own workarea.\n\n  @param[in] plan Pointer to the FFT plan.\n  @param[in] autoAllocate 0 to disable auto-allocation, non-zero to enable."]
    pub fn hipfftSetAutoAllocation(
        plan: hipfftHandle,
        autoAllocate: ::std::os::raw::c_int,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Set the plan's work area.\n\n  @param[in] plan Pointer to the FFT plan.\n  @param[in] workArea Pointer to the work area (on device)."]
    pub fn hipfftSetWorkArea(
        plan: hipfftHandle,
        workArea: *mut ::std::os::raw::c_void,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Execute a (float) complex-to-complex FFT.\n\n  @details If the input and output buffers are equal, an in-place\n  transform is performed.\n\n  @param plan The FFT plan.\n  @param idata Input data (on device).\n  @param odata Output data (on device).\n  @param direction Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`."]
    pub fn hipfftExecC2C(
        plan: hipfftHandle,
        idata: *mut hipfftComplex,
        odata: *mut hipfftComplex,
        direction: ::std::os::raw::c_int,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Execute a (float) real-to-complex FFT.\n\n  @details If the input and output buffers are equal, an in-place\n  transform is performed.\n\n  @param plan The FFT plan.\n  @param idata Input data (on device).\n  @param odata Output data (on device)."]
    pub fn hipfftExecR2C(
        plan: hipfftHandle,
        idata: *mut hipfftReal,
        odata: *mut hipfftComplex,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Execute a (float) complex-to-real FFT.\n\n  @details If the input and output buffers are equal, an in-place\n  transform is performed.\n\n  @param plan The FFT plan.\n  @param idata Input data (on device).\n  @param odata Output data (on device)."]
    pub fn hipfftExecC2R(
        plan: hipfftHandle,
        idata: *mut hipfftComplex,
        odata: *mut hipfftReal,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Execute a (double) complex-to-complex FFT.\n\n  @details If the input and output buffers are equal, an in-place\n  transform is performed.\n\n  @param plan The FFT plan.\n  @param idata Input data (on device).\n  @param odata Output data (on device).\n  @param direction Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`."]
    pub fn hipfftExecZ2Z(
        plan: hipfftHandle,
        idata: *mut hipfftDoubleComplex,
        odata: *mut hipfftDoubleComplex,
        direction: ::std::os::raw::c_int,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Execute a (double) real-to-complex FFT.\n\n  @details If the input and output buffers are equal, an in-place\n  transform is performed.\n\n  @param plan The FFT plan.\n  @param idata Input data (on device).\n  @param odata Output data (on device)."]
    pub fn hipfftExecD2Z(
        plan: hipfftHandle,
        idata: *mut hipfftDoubleReal,
        odata: *mut hipfftDoubleComplex,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Execute a (double) complex-to-real FFT.\n\n  @details If the input and output buffers are equal, an in-place\n  transform is performed.\n\n  @param plan The FFT plan.\n  @param idata Input data (on device).\n  @param odata Output data (on device)."]
    pub fn hipfftExecZ2D(
        plan: hipfftHandle,
        idata: *mut hipfftDoubleComplex,
        odata: *mut hipfftDoubleReal,
    ) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Set HIP stream to execute plan on.\n\n @details Associates a HIP stream with a hipFFT plan.  All kernels\n launched by this plan are associated with the provided stream.\n\n @param plan The FFT plan.\n @param stream The HIP stream."]
    pub fn hipfftSetStream(plan: hipfftHandle, stream: hipStream_t) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Destroy and deallocate an existing plan."]
    pub fn hipfftDestroy(plan: hipfftHandle) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Get rocFFT/cuFFT version.\n\n  @param[out] version cuFFT/rocFFT version (returned value)."]
    pub fn hipfftGetVersion(version: *mut ::std::os::raw::c_int) -> hipfftResult;
}
extern "C" {
    #[doc = " @brief Get library property.\n\n  @param[in] type Property type.\n  @param[out] value Returned value."]
    pub fn hipfftGetProperty(
        type_: hipfftLibraryPropertyType,
        value: *mut ::std::os::raw::c_int,
    ) -> hipfftResult;
}
