use std::convert::TryFrom;

pub trait FromUsize: Sized {
    fn from_usize(x: usize) -> Self;
}

impl FromUsize for i32 {
    fn from_usize(x: usize) -> Self {
        i32::try_from(x).unwrap_or_else(|_| panic!("Failed to cast usize to i32"))
    }
}

impl FromUsize for u32 {
    fn from_usize(x: usize) -> Self {
        u32::try_from(x).unwrap_or_else(|_| panic!("Failed to cast usize to u32"))
    }
}

impl FromUsize for f32 {
    fn from_usize(x: usize) -> Self {
        x as f32
    }
}

impl FromUsize for f64 {
    fn from_usize(x: usize) -> Self {
        x as f64
    }
}
