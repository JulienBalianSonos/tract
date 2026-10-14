#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]
#![allow(deref_nullptr)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::redundant_static_lifetimes)]
#![allow(clippy::useless_transmute)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::ptr_offset_with_cast)]
#![allow(clippy::manual_div_ceil)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
