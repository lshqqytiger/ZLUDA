#[allow(non_camel_case_types)]
#[allow(dead_code)]
pub mod amd_comgr;
#[allow(non_camel_case_types)]
#[allow(dead_code)]
pub mod amd_comgr_3;

pub mod comgr;
use comgr::{ActionInfo, ActionKind, Data, DataSet, LibComgr, Relocatable, Result};
pub use comgr::{Bitcode, Error};

use hip_common::CompilationMode;
use std::{
    env,
    ffi::CStr,
    iter,
    sync::atomic::{AtomicU64, Ordering},
};

#[cfg(windows)]
static CODE_OBJECT_VERSION_FLAG: &'static [u8] = b"code_object_v4\0";
#[cfg(not(windows))]
static CODE_OBJECT_VERSION_FLAG: &'static [u8] = b"code_object_v5\0";

pub struct Comgr(LibComgr, AtomicU64, u32);

static WAVE32_MODULE: &'static [u8] = include_bytes!("wave32.ll");
static WAVE32_ON_WAVE64_MODULE: &'static [u8] = include_bytes!("wave32_on_wave64.ll");
static DOUBLE_WAVE32_ON_WAVE64_MODULE: &'static [u8] = include_bytes!("double_wave32_on_wave64.ll");

#[cfg(windows)]
static OS_MODULE: &'static [u8] = include_bytes!("windows.ll");
#[cfg(not(windows))]
static OS_MODULE: &'static [u8] = include_bytes!("linux.ll");

impl Comgr {
    pub fn find_and_load() -> Result<Self> {
        match unsafe { LibComgr::new() } {
            Ok(libcomgr) => Ok(Self(
                libcomgr,
                AtomicU64::new(1),
                env::var("ZLUDA_COMGR_LOG_LEVEL")
                    .unwrap_or("0".into())
                    .parse()
                    .expect("Unexpected value for ZLUDA_COMGR_LOG_LEVEL."),
            )),
            Err(_) => Err(Error::Generic),
        }
    }

    fn get(&self) -> &LibComgr {
        &self.0
    }

    pub fn compile<'a>(
        &self,
        compilation_mode: CompilationMode,
        isa: &'a CStr,
        input_bitcode: impl Iterator<Item = (impl AsRef<[u8]>, impl AsRef<CStr>)>,
        linker_module: &[u8],
    ) -> Result<Vec<u8>> {
        let bitcode = self.link_bitcode_impl(compilation_mode, isa, input_bitcode)?;
        let relocatable = self.build_relocatable_impl(compilation_mode, isa, &bitcode)?;
        if !linker_module.is_empty() {
            let source = self.assemble_source(isa, linker_module)?;
            self.link_relocatable_impl(
                isa,
                IntoIterator::into_iter([
                    &relocatable.get_data(
                        amd_comgr_3::amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_RELOCATABLE,
                        0,
                    )?,
                    &source,
                ]),
            )
        } else {
            self.link_relocatable_impl(
                isa,
                iter::once(&relocatable.get_data(
                    amd_comgr_3::amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_RELOCATABLE,
                    0,
                )?),
            )
        }
    }

    #[cfg(windows)]
    #[inline]
    unsafe fn get_code_object_version_flag<'cstr>(&self) -> &'cstr CStr {
        let flag = if let LibComgr::V3(_) = self.0 {
            b"-mcode-object-version=5\0"
        } else {
            b"-mcode-object-version=4\0"
        };
        CStr::from_bytes_with_nul_unchecked(flag)
    }

    #[cfg(not(windows))]
    #[inline]
    unsafe fn get_code_object_version_flag<'cstr>(&self) -> &'cstr CStr {
        CStr::from_bytes_with_nul_unchecked(b"-mcode-object-version=5\0")
    }

    #[inline]
    unsafe fn get_base_compiler_options<'cstr>(
        &self,
        compilation_mode: CompilationMode,
    ) -> [&'cstr CStr; 2] {
        let compilation_mode = if compilation_mode == CompilationMode::Wave32 {
            CStr::from_bytes_with_nul_unchecked(b"-mno-wavefrontsize64\0")
        } else {
            CStr::from_bytes_with_nul_unchecked(b"-mwavefrontsize64\0")
        };
        let code_object_version_flag = self.get_code_object_version_flag();
        [compilation_mode, code_object_version_flag]
    }

    pub fn link_bitcode<'this, 'a>(
        &'this self,
        compilation_mode: CompilationMode,
        isa: &'a CStr,
        input_bitcode: impl Iterator<Item = (impl AsRef<[u8]>, &'a CStr)>,
    ) -> Result<Bitcode<'this>> {
        let data_set_bitcode = self.link_bitcode_impl(compilation_mode, isa, input_bitcode)?;
        Ok(Bitcode(data_set_bitcode))
    }

    pub fn bitcode_to_relocatable<'this, 'a>(
        &'this self,
        compilation_mode: CompilationMode,
        isa: &'a CStr,
        bc: &'this Bitcode,
    ) -> Result<Relocatable<'this>> {
        let data_set_relocatable = self.build_relocatable_impl(compilation_mode, isa, &bc.0)?;
        let suffix = self.1.fetch_add(1, Ordering::Relaxed);
        Ok(Relocatable::from_data(
            data_set_relocatable.get_data(
                amd_comgr_3::amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_RELOCATABLE,
                0,
            )?,
            suffix,
        )?)
    }

    pub fn build_relocatable<'this, 'a>(
        &'this self,
        compilation_mode: CompilationMode,
        isa: &'a CStr,
        input_bitcode: impl Iterator<Item = (impl AsRef<[u8]>, &'a CStr)>,
    ) -> Result<Relocatable<'this>> {
        let bitcode = self.link_bitcode_impl(compilation_mode, isa, input_bitcode)?;
        let data_set_relocatable = self.build_relocatable_impl(compilation_mode, isa, &bitcode)?;
        let suffix = self.1.fetch_add(1, Ordering::Relaxed);
        Ok(Relocatable::from_data(
            data_set_relocatable.get_data(
                amd_comgr_3::amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_RELOCATABLE,
                0,
            )?,
            suffix,
        )?)
    }

    pub fn link_relocatable<'this, 'a>(
        &'this self,
        isa: &'a CStr,
        modules: impl Iterator<Item = &'this Relocatable<'this>>,
    ) -> Result<Vec<u8>> {
        self.link_relocatable_impl(isa, modules.map(|reloc| &reloc.0))
    }

    pub fn version(&self) -> Result<String> {
        let mut data_set = DataSet::new(self.get())?;
        let data = Data::new(
            self.get(),
            amd_comgr_3::amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_SOURCE,
            b"__VERSION__",
            unsafe { CStr::from_bytes_with_nul_unchecked(b"version.h\0") },
        )?;
        data_set.add(&data)?;
        let result = self.do_action(
            ActionKind::SourceToPreprocessor,
            &data_set,
            unsafe { CStr::from_bytes_with_nul_unchecked(b"\0") },
            iter::once(unsafe { CStr::from_bytes_with_nul_unchecked(b"-nogpuinc\0") }),
            Some(amd_comgr_3::amd_comgr_language_t::AMD_COMGR_LANGUAGE_HIP),
        )?;
        let result = result.get_data(
            amd_comgr_3::amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_SOURCE,
            0,
        )?;
        let result = result.get_data()?;
        let end_quote = result
            .iter()
            .copied()
            .rposition(|c| c as char == '"')
            .ok_or(Error::Generic)?;
        let start_quote = result[..end_quote]
            .iter()
            .copied()
            .rposition(|c| c as char == '"')
            .ok_or(Error::Generic)?;
        String::from_utf8(result[start_quote + 1..end_quote].to_vec()).map_err(|_| Error::Generic)
    }

    fn link_bitcode_impl<'this, 'a>(
        &'this self,
        compilation_mode: CompilationMode,
        isa: &'a CStr,
        input_bitcode: impl Iterator<Item = (impl AsRef<[u8]>, impl AsRef<CStr>)>,
    ) -> Result<DataSet<'this>> {
        let mut bitcode_modules = DataSet::new(self.get())?;
        for (bc, name) in input_bitcode {
            bitcode_modules.add(&Data::new(
                self.get(),
                amd_comgr_3::amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_BC,
                bc.as_ref(),
                name.as_ref(),
            )?)?;
        }
        let wave_module_text = match compilation_mode {
            CompilationMode::Wave32 => WAVE32_MODULE,
            CompilationMode::Wave32OnWave64 => WAVE32_ON_WAVE64_MODULE,
            CompilationMode::DoubleWave32OnWave64 => DOUBLE_WAVE32_ON_WAVE64_MODULE,
        };
        let wave_module = Data::new(
            self.get(),
            amd_comgr_3::amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_BC,
            wave_module_text,
            unsafe { CStr::from_bytes_with_nul_unchecked(b"wave.ll\0") },
        )?;
        bitcode_modules.add(&wave_module)?;
        let os_module = Data::new(
            self.get(),
            amd_comgr_3::amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_BC,
            OS_MODULE,
            unsafe { CStr::from_bytes_with_nul_unchecked(b"os.ll\0") },
        )?;
        bitcode_modules.add(&os_module)?;
        if let LibComgr::V3(_) = self.0 {
            let compiler_options = unsafe { self.get_base_compiler_options(compilation_mode) };
            let linked_module = self.do_action(
                ActionKind::LinkBcToBc,
                &bitcode_modules,
                isa,
                compiler_options.iter().copied(),
                None,
            )?;
            self.do_action(
                ActionKind::CompileSourceWithDeviceLibsToBcV3,
                &linked_module,
                isa,
                compiler_options.into_iter().chain(
                    unsafe {
                        [
                            CStr::from_bytes_with_nul_unchecked(b"-Xclang\0"),
                            CStr::from_bytes_with_nul_unchecked(
                                b"-mno-link-builtin-bitcode-postopt\0",
                            ),
                        ]
                    }
                    .into_iter(),
                ),
                Some(amd_comgr_3::amd_comgr_language_t::AMD_COMGR_LANGUAGE_LLVM_IR),
            )
        } else {
            let mut comgr_options =
                vec![unsafe { CStr::from_bytes_with_nul_unchecked(CODE_OBJECT_VERSION_FLAG) }];
            if compilation_mode == CompilationMode::Wave32OnWave64
                || compilation_mode == CompilationMode::DoubleWave32OnWave64
            {
                comgr_options
                    .push(unsafe { CStr::from_bytes_with_nul_unchecked(b"wavefrontsize64\0") });
            }
            let device_linking_output: DataSet<'_> = self.do_action(
                ActionKind::AddDeviceLibrariesV2,
                &bitcode_modules,
                isa,
                comgr_options.iter().copied(), //.chain([wavefront_option].into_iter()),
                Some(amd_comgr_3::amd_comgr_language_t::AMD_COMGR_LANGUAGE_OPENCL_2_0),
            )?;
            comgr_options.truncate(1);
            self.do_action(
                ActionKind::LinkBcToBc,
                &device_linking_output,
                isa,
                comgr_options.into_iter(),
                None,
            )
        }
    }

    fn build_relocatable_impl<'this, 'a>(
        &'this self,
        compilation_mode: CompilationMode,
        isa: &'a CStr,
        bc_linking_output: &DataSet<'this>,
    ) -> Result<DataSet<'this>> {
        let debug_level = if cfg!(debug_assertions) {
            unsafe {
                [
                    CStr::from_bytes_with_nul_unchecked(b"-g\0"),
                    CStr::from_bytes_with_nul_unchecked(b"\0"),
                    CStr::from_bytes_with_nul_unchecked(b"\0"),
                    CStr::from_bytes_with_nul_unchecked(b"\0"),
                    CStr::from_bytes_with_nul_unchecked(b"\0"),
                ]
            }
        } else {
            unsafe {
                [
                    CStr::from_bytes_with_nul_unchecked(b"-g0\0"),
                    CStr::from_bytes_with_nul_unchecked(b"-mllvm\0"),
                    CStr::from_bytes_with_nul_unchecked(b"-inline-threshold=2250\0"),
                    CStr::from_bytes_with_nul_unchecked(b"-mllvm\0"),
                    CStr::from_bytes_with_nul_unchecked(b"-inlinehint-threshold=3250\0"),
                ]
            }
        };
        if self.2 == 1 {
            eprintln!("Compilation is in progress. Please wait...");
        }
        let relocatable = self.do_action(
            ActionKind::CodegenBcToRelocatable,
            bc_linking_output,
            isa,
            unsafe { self.get_base_compiler_options(compilation_mode) }
                .into_iter()
                .chain(
                    unsafe {
                        [
                            CStr::from_bytes_with_nul_unchecked(b"-O3\0"),
                            // TODO: measure more
                            // Slightly more efficient in Blender
                            CStr::from_bytes_with_nul_unchecked(b"-mcumode\0"),
                            CStr::from_bytes_with_nul_unchecked(b"-ffp-contract=off\0"),
                            CStr::from_bytes_with_nul_unchecked(b"-mllvm\0"),
                            CStr::from_bytes_with_nul_unchecked(b"-amdgpu-internalize-symbols\0"),
                            // TODO: This emits scratch_ instead of buffer_ instructions
                            // for stack spills&fills, measure impact
                            // CStr::from_bytes_with_nul_unchecked(b"-Xclang\0"),
                            // CStr::from_bytes_with_nul_unchecked(b"-target-feature\0"),
                            // CStr::from_bytes_with_nul_unchecked(b"-Xclang\0"),
                            // CStr::from_bytes_with_nul_unchecked(b"+enable-flat-scratch\0"),
                            // Useful for debugging miscompilations
                            // CStr::from_bytes_with_nul_unchecked(b"-mllvm\0"),
                            // CStr::from_bytes_with_nul_unchecked(b"-opt-bisect-limit=-1\0"),
                        ]
                    }
                    .into_iter(),
                )
                .chain(debug_level.into_iter()),
            None,
        )?;
        Ok(relocatable)
    }

    fn link_relocatable_impl<'this, 'a>(
        &'this self,
        isa: &'a CStr,
        modules: impl Iterator<Item = &'this Data<'this>>,
    ) -> Result<Vec<u8>> {
        let mut input = DataSet::new(self.get())?;
        for module in modules {
            input.add(module)?;
        }
        let executable_set: DataSet = self.do_action(
            ActionKind::LinkRelocatableToExecutable,
            &input,
            isa,
            unsafe {
                [
                    CStr::from_bytes_with_nul_unchecked(b"-Xlinker\0"),
                    CStr::from_bytes_with_nul_unchecked(b"--no-undefined\0"),
                ]
            }
            .into_iter(),
            None,
        )?;
        let executable_data = executable_set.get_data(
            amd_comgr_3::amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_EXECUTABLE,
            0,
        )?;
        executable_data.get_data()
    }

    fn assemble_source(&self, isa: &CStr, src: &[u8]) -> Result<Data> {
        let data = Data::new(
            self.get(),
            amd_comgr_3::amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_SOURCE,
            src,
            unsafe { CStr::from_bytes_with_nul_unchecked(b"input.s\0") },
        )?;
        let mut data_set = DataSet::new(self.get())?;
        data_set.add(&data)?;
        let assembled = self.do_action(
            ActionKind::AssembleSourceToRelocatable,
            &data_set,
            isa,
            iter::once(unsafe { self.get_code_object_version_flag() }),
            None,
        )?;
        assembled.get_data(
            amd_comgr_3::amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_RELOCATABLE,
            0,
        )
    }

    fn do_action<'a, 'cstr>(
        &'a self,
        kind: ActionKind,
        input: &DataSet,
        isa: &CStr,
        options: impl Iterator<Item = &'cstr CStr>,
        language: Option<amd_comgr_3::amd_comgr_language_t>,
    ) -> Result<DataSet<'a>> {
        let output = DataSet::new(self.get())?;
        let action = ActionInfo::new(self.get(), isa)?;
        if options.size_hint().1.unwrap() > 0 {
            action.set_options(options)?;
        }
        if let Some(lang) = language {
            action.set_language(lang)?;
        }
        if let LibComgr::V3(_) = self.0 {
            action.execute3(kind, &input, &output)?;
        } else {
            action.execute(kind, &input, &output)?;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use crate::Comgr;

    #[test]
    fn version() {
        let comgr = Comgr::find_and_load().unwrap();
        let version = comgr.version().unwrap();
        assert!(version.contains("Clang"));
        assert!(!version.contains("\""));
        assert!(!version.contains("\n"));
    }
}
