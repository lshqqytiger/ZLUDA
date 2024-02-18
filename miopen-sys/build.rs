use std::env::VarError;
use std::{env, path::PathBuf};

fn main() -> Result<(), VarError> {
    println!("cargo:rustc-link-lib=dylib=MIOpen");
    if cfg!(windows) {
        let mut path = PathBuf::from(env::var("MIOPEN_PATH")?);
        path.push("lib");
        println!("cargo:rustc-link-search=native={}", path.display());
    } else {
        println!("cargo:rustc-link-search=native=/opt/rocm/lib/");
    }
    Ok(())
}
