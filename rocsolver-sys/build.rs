use std::env::VarError;
use std::{env, path::PathBuf};

fn main() -> Result<(), VarError> {
    println!("cargo:rustc-link-lib=dylib=rocsolver");
    let mut path = PathBuf::from(env::var("HIP_PATH")?);
    path.push("lib");
    println!("cargo:rustc-link-search=native={}", path.display());
    Ok(())
}
