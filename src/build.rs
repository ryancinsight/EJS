// build.rs

extern crate winres;

fn main() {
    let mut res = winres::WindowsResource::new();
    res.set_icon("src/sona.ico")
        .set("InternalName", "SONA.EXE")
        .set("CompanyName", "SonALAsense");
    res.compile().unwrap();
}
