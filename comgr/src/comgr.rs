use std::ffi::CStr;
use std::{mem, ptr};

use crate::amd_comgr;
use crate::amd_comgr_3;

#[derive(Debug)]
pub enum Error {
    Generic,
    InvalidArgument,
    OutOfResources,
}

trait IntoResult<T> {
    fn into_result(self) -> Result<T>;
}

impl IntoResult<()> for amd_comgr::amd_comgr_status_t {
    fn into_result(self) -> Result<()> {
        match self {
            amd_comgr::amd_comgr_status_t::AMD_COMGR_STATUS_SUCCESS => Ok(()),
            amd_comgr::amd_comgr_status_t::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT => {
                Err(Error::InvalidArgument)
            }
            amd_comgr::amd_comgr_status_t::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES => {
                Err(Error::OutOfResources)
            }
            _ => Err(Error::Generic),
        }
    }
}

impl IntoResult<()> for amd_comgr_3::amd_comgr_status_t {
    fn into_result(self) -> Result<()> {
        match self {
            amd_comgr_3::amd_comgr_status_t::AMD_COMGR_STATUS_SUCCESS => Ok(()),
            amd_comgr_3::amd_comgr_status_t::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT => {
                Err(Error::InvalidArgument)
            }
            amd_comgr_3::amd_comgr_status_t::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES => {
                Err(Error::OutOfResources)
            }
            _ => Err(Error::Generic),
        }
    }
}

trait And {
    fn and<V>(self, value: V) -> Result<V>;
}

impl<T: IntoResult<()>> And for T {
    fn and<V>(self, value: V) -> Result<V> {
        let result: Result<()> = self.into_result();
        result.and(Ok(value))
    }
}

impl Into<amd_comgr_3::amd_comgr_data_t> for amd_comgr::amd_comgr_data_t {
    fn into(self) -> amd_comgr_3::amd_comgr_data_t {
        unsafe { mem::transmute(self) }
    }
}

impl Into<amd_comgr::amd_comgr_data_t> for amd_comgr_3::amd_comgr_data_t {
    fn into(self) -> amd_comgr::amd_comgr_data_t {
        unsafe { mem::transmute(self) }
    }
}

impl Into<amd_comgr::amd_comgr_data_kind_t> for amd_comgr_3::amd_comgr_data_kind_t {
    fn into(self) -> amd_comgr::amd_comgr_data_kind_t {
        if self.0 <= amd_comgr::amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_LAST.0 {
            return amd_comgr::amd_comgr_data_kind_t(self.0);
        }
        unreachable!()
    }
}

impl Into<amd_comgr_3::amd_comgr_data_set_t> for amd_comgr::amd_comgr_data_set_t {
    fn into(self) -> amd_comgr_3::amd_comgr_data_set_t {
        unsafe { mem::transmute(self) }
    }
}

impl Into<amd_comgr::amd_comgr_data_set_t> for amd_comgr_3::amd_comgr_data_set_t {
    fn into(self) -> amd_comgr::amd_comgr_data_set_t {
        unsafe { mem::transmute(self) }
    }
}

impl Into<amd_comgr_3::amd_comgr_action_info_t> for amd_comgr::amd_comgr_action_info_t {
    fn into(self) -> amd_comgr_3::amd_comgr_action_info_t {
        unsafe { mem::transmute(self) }
    }
}

impl Into<amd_comgr::amd_comgr_action_info_t> for amd_comgr_3::amd_comgr_action_info_t {
    fn into(self) -> amd_comgr::amd_comgr_action_info_t {
        unsafe { mem::transmute(self) }
    }
}

impl Into<amd_comgr::amd_comgr_language_t> for amd_comgr_3::amd_comgr_language_t {
    fn into(self) -> amd_comgr::amd_comgr_language_t {
        match self {
            amd_comgr_3::amd_comgr_language_t::AMD_COMGR_LANGUAGE_NONE => {
                amd_comgr::amd_comgr_language_t::AMD_COMGR_LANGUAGE_NONE
            }
            amd_comgr_3::amd_comgr_language_t::AMD_COMGR_LANGUAGE_OPENCL_1_2 => {
                amd_comgr::amd_comgr_language_t::AMD_COMGR_LANGUAGE_OPENCL_1_2
            }
            amd_comgr_3::amd_comgr_language_t::AMD_COMGR_LANGUAGE_OPENCL_2_0 => {
                amd_comgr::amd_comgr_language_t::AMD_COMGR_LANGUAGE_OPENCL_2_0
            }
            amd_comgr_3::amd_comgr_language_t::AMD_COMGR_LANGUAGE_HIP => {
                amd_comgr::amd_comgr_language_t::AMD_COMGR_LANGUAGE_HIP
            }
            amd_comgr_3::amd_comgr_language_t::AMD_COMGR_LANGUAGE_LLVM_IR => {
                amd_comgr::amd_comgr_language_t::AMD_COMGR_LANGUAGE_LLVM_IR
            }
            _ => unreachable!(),
        }
    }
}

pub(crate) enum ActionKind {
    SourceToPreprocessor,
    LinkBcToBc,
    CodegenBcToRelocatable,
    LinkRelocatableToExecutable,
    AssembleSourceToRelocatable,

    AddDeviceLibrariesV2,

    CompileSourceWithDeviceLibsToBcV3,
}

impl ActionKind {
    fn into_v2(self) -> amd_comgr::amd_comgr_action_kind_t {
        match self {
            ActionKind::SourceToPreprocessor => {
                amd_comgr::amd_comgr_action_kind_t::AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR
            }
            ActionKind::LinkBcToBc => {
                amd_comgr::amd_comgr_action_kind_t::AMD_COMGR_ACTION_LINK_BC_TO_BC
            }
            ActionKind::CodegenBcToRelocatable => {
                amd_comgr::amd_comgr_action_kind_t::AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE
            }
            ActionKind::LinkRelocatableToExecutable => {
                amd_comgr::amd_comgr_action_kind_t::AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE
            }
            ActionKind::AssembleSourceToRelocatable => {
                amd_comgr::amd_comgr_action_kind_t::AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE
            }
            ActionKind::AddDeviceLibrariesV2 => {
                amd_comgr::amd_comgr_action_kind_t::AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES
            }
            _ => unreachable!(),
        }
    }
}

impl ActionKind {
    fn into_v3(self) -> amd_comgr_3::amd_comgr_action_kind_t {
        match self {
            ActionKind::SourceToPreprocessor => {
                amd_comgr_3::amd_comgr_action_kind_t::AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR
            }
            ActionKind::LinkBcToBc => {
                amd_comgr_3::amd_comgr_action_kind_t::AMD_COMGR_ACTION_LINK_BC_TO_BC
            }
            ActionKind::CodegenBcToRelocatable => {
                amd_comgr_3::amd_comgr_action_kind_t::AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE
            }
            ActionKind::LinkRelocatableToExecutable => {
                amd_comgr_3::amd_comgr_action_kind_t::AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE
            }
            ActionKind::AssembleSourceToRelocatable => {
                amd_comgr_3::amd_comgr_action_kind_t::AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE
            }
            ActionKind::CompileSourceWithDeviceLibsToBcV3 => {
                amd_comgr_3::amd_comgr_action_kind_t::AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC
            }
            _ => unreachable!(),
        }
    }
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

macro_rules! with {
    ($wrapper:expr, $comgr:ident => $expr:expr) => {
        match $wrapper {
            LibComgr::V2($comgr) => unsafe { $expr },
            LibComgr::V3($comgr) => unsafe { $expr },
        }
    };
}

macro_rules! with_v2 {
    ($wrapper:expr, $comgr:ident => $expr:expr) => {
        if let LibComgr::V2($comgr) = $wrapper {
            unsafe { $expr }
        } else {
            unreachable!()
        }
    };
}

macro_rules! with_v3 {
    ($wrapper:expr, $comgr:ident => $expr:expr) => {
        if let LibComgr::V3($comgr) = $wrapper {
            unsafe { $expr }
        } else {
            unreachable!()
        }
    };
}

#[allow(unused_must_use)]
impl<'a> Drop for ActionInfo<'a> {
    fn drop(&mut self) {
        with!(self.1, comgr => {
            comgr.amd_comgr_destroy_action_info(self.get().into());
        });
    }
}

pub struct Bitcode<'a>(pub(crate) DataSet<'a>);

impl<'a> Bitcode<'a> {
    pub fn get_data(&self) -> Result<Vec<u8>> {
        self.0
            .get_data(
                amd_comgr_3::amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_BC,
                0,
            )?
            .get_data()
    }
}

pub struct Relocatable<'a>(pub(crate) Data<'a>);

impl<'a> Relocatable<'a> {
    pub(crate) fn from_data(data: Data<'a>, suffix: u64) -> Result<Self> {
        let name = format!("reloc_{}.o\0", suffix);
        with!(data.1, comgr => {
            comgr
                .amd_comgr_set_data_name(data.get().into(), name.as_ptr().cast())
                .into_result()
        })?;
        Ok(Self(data))
    }

    pub fn get_data(&self) -> Result<Vec<u8>> {
        self.0.get_data()
    }
}

pub(crate) struct ActionInfo<'a>(amd_comgr_3::amd_comgr_action_info_t, &'a LibComgr);

impl<'a> ActionInfo<'a> {
    pub(crate) fn new(comgr: &'a LibComgr, isa: &'a CStr) -> Result<Self> {
        let action_info = unsafe { comgr.create_action_info() }?;
        with!(comgr, comgr => {
            comgr
                .amd_comgr_action_info_set_isa_name(action_info.into(), isa.as_ptr())
                .into_result()
        })?;
        Ok(ActionInfo(action_info, comgr))
    }

    fn get(&self) -> amd_comgr_3::amd_comgr_action_info_t {
        self.0
    }

    pub(crate) fn set_options<'cstr>(
        &self,
        options: impl Iterator<Item = &'cstr CStr>,
    ) -> Result<()> {
        let mut options_c = options.map(CStr::as_ptr).collect::<Vec<_>>();
        with!(self.1, comgr => {
            comgr
                .amd_comgr_action_info_set_option_list(
                    self.0.into(),
                    options_c.as_mut_ptr(),
                    options_c.len(),
                )
                .into_result()
        })
    }

    pub(crate) fn set_language(&self, language: amd_comgr_3::amd_comgr_language_t) -> Result<()> {
        with!(self.1, comgr => {
            comgr
                .amd_comgr_action_info_set_language(self.0.into(), language.into())
                .into_result()
        })
    }

    pub(crate) fn execute(
        &self,
        kind: ActionKind,
        input: &DataSet<'a>,
        output: &DataSet<'a>,
    ) -> Result<()> {
        with_v2!(self.1, comgr => {
            comgr
                .amd_comgr_do_action(kind.into_v2(), self.get().into(), input.get().into(), output.get().into())
                .into_result()
        })
    }

    pub(crate) fn execute3(
        &self,
        kind: ActionKind,
        input: &DataSet<'a>,
        output: &DataSet<'a>,
    ) -> Result<()> {
        with_v3!(self.1, comgr => {
            comgr
                .amd_comgr_do_action(kind.into_v3(), self.get(), input.get(), output.get())
                .into_result()
        })
    }
}

pub(crate) struct DataSet<'a>(amd_comgr_3::amd_comgr_data_set_t, &'a LibComgr);

impl<'a> DataSet<'a> {
    pub(crate) fn new(comgr: &'a LibComgr) -> Result<DataSet<'a>> {
        unsafe {
            let data_set = comgr.create_data_set()?;
            Ok(DataSet(data_set, comgr))
        }
    }

    pub(crate) fn add(&mut self, data: &Data<'a>) -> Result<()> {
        with!(self.1, comgr => {
            comgr
                .amd_comgr_data_set_add(self.0.into(), data.get().into())
                .into_result()
        })
    }

    pub(crate) fn get_data(
        &self,
        kind: amd_comgr_3::amd_comgr_data_kind_t,
        index: usize,
    ) -> Result<Data<'a>> {
        let output = unsafe { self.1.action_data_get_data(self.0, kind, index) }?;
        Ok(Data(output, self.1))
    }

    fn get(&self) -> amd_comgr_3::amd_comgr_data_set_t {
        self.0
    }
}

#[allow(unused_must_use)]
impl<'a> Drop for DataSet<'a> {
    fn drop(&mut self) {
        with!(self.1, comgr => {
            comgr.amd_comgr_destroy_data_set(self.get().into());
        });
    }
}

pub(crate) struct Data<'a>(amd_comgr_3::amd_comgr_data_t, &'a LibComgr);

impl<'a> Data<'a> {
    pub(crate) fn new(
        comgr: &'a LibComgr,
        kind: amd_comgr_3::amd_comgr_data_kind_t,
        data: &[u8],
        name: &CStr,
    ) -> Result<Data<'a>> {
        let comgr_data = unsafe { comgr.create_data(kind) }?;
        with!(comgr, comgr => {
            comgr
                .amd_comgr_set_data_name(comgr_data.into(), name.as_ptr())
                .into_result()?;
            comgr
                .amd_comgr_set_data(comgr_data.into(), data.len(), data.as_ptr().cast())
                .into_result()?;
        });
        Ok(Self(comgr_data, comgr))
    }

    fn get(&self) -> amd_comgr_3::amd_comgr_data_t {
        self.0
    }

    pub(crate) fn get_data(&self) -> Result<Vec<u8>> {
        unsafe {
            let size = self.1.get_data_size(self.get())?;
            self.1.get_data(self.get(), size)
        }
    }
}

#[allow(unused_must_use)]
impl<'a> Drop for Data<'a> {
    fn drop(&mut self) {
        with!(self.1, comgr => {
            comgr.amd_comgr_release_data(self.get().into());
        });
    }
}

pub(crate) enum LibComgr {
    // Comgr version 1, 2
    // for backward compatability
    V2(amd_comgr::LibComgr),
    // Comgr version 3
    V3(amd_comgr_3::LibComgr),
}

impl LibComgr {
    #[cfg(windows)]
    pub(crate) unsafe fn new() -> std::result::Result<Self, libloading::Error> {
        let library = libloading::Library::new("amd_comgr_3.dll")
            .or_else(|_| libloading::Library::new("amd_comgr_2.dll"))
            .or_else(|_| libloading::Library::new("amd_comgr.dll"))?;
        let get_version = library
            .get::<fn(major: *mut usize, minor: *mut usize) -> std::ffi::c_void>(
                b"amd_comgr_get_version",
            )?;
        let (mut major, mut minor) = mem::zeroed();
        get_version(&mut major, &mut minor);
        if major == 2 && minor < 9 || major < 2 {
            let library = amd_comgr::LibComgr::from_library(library)?;
            return Ok(LibComgr::V2(library));
        }
        let library = amd_comgr_3::LibComgr::from_library(library)?;
        Ok(LibComgr::V3(library))
    }

    #[cfg(not(windows))]
    pub(crate) unsafe fn new() -> std::result::Result<Self, libloading::Error> {
        let library = amd_comgr_3::LibComgr::new("libamd_comgr.so.3")
            .or_else(|_| amd_comgr_3::LibComgr::new("/opt/rocm/lib/libamd_comgr.so.3"));
        if let Ok(library) = library {
            return Ok(LibComgr::V3(library));
        }
        Ok(LibComgr::V2(
            amd_comgr::LibComgr::new("libamd_comgr.so.2")
                .or_else(|_| amd_comgr::LibComgr::new("/opt/rocm/lib/libamd_comgr.so.2"))?,
        ))
    }

    unsafe fn create_action_info(&self) -> Result<amd_comgr_3::amd_comgr_action_info_t> {
        with!(self, comgr => {
            let mut info = mem::zeroed();
            comgr
                .amd_comgr_create_action_info(&mut info)
                .and(info.into())
        })
    }

    unsafe fn action_data_get_data(
        &self,
        data_set: amd_comgr_3::amd_comgr_data_set_t,
        kind: amd_comgr_3::amd_comgr_data_kind_t,
        index: usize,
    ) -> Result<amd_comgr_3::amd_comgr_data_t> {
        with!(self, comgr => {
            let mut data = mem::zeroed();
            comgr.amd_comgr_action_data_get_data(data_set.into(), kind.into(), index, &mut data).and(data.into())
        })
    }

    unsafe fn create_data(
        &self,
        kind: amd_comgr_3::amd_comgr_data_kind_t,
    ) -> Result<amd_comgr_3::amd_comgr_data_t> {
        with!(self, comgr => {
            let mut data = mem::zeroed();
            comgr.amd_comgr_create_data(kind.into(), &mut data).and(data.into())
        })
    }

    unsafe fn get_data_size(&self, data: amd_comgr_3::amd_comgr_data_t) -> Result<usize> {
        with!(self, comgr => {
            let mut size = 0;
            comgr.amd_comgr_get_data(data.into(), &mut size, ptr::null_mut()).and(size)
        })
    }

    unsafe fn get_data(
        &self,
        data: amd_comgr_3::amd_comgr_data_t,
        mut size: usize,
    ) -> Result<Vec<u8>> {
        with!(self, comgr => {
            let mut bytes = vec![0u8; size];
            comgr.amd_comgr_get_data(data.into(), &mut size, bytes.as_mut_ptr().cast()).and(bytes)
        })
    }

    unsafe fn create_data_set(&self) -> Result<amd_comgr_3::amd_comgr_data_set_t> {
        with!(self, comgr => {
            let mut data_set = mem::zeroed();
            comgr
                .amd_comgr_create_data_set(&mut data_set)
                .and(data_set.into())
        })
    }
}
