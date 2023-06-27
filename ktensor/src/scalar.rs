use num_traits::NumOps;

pub struct Scalar<T>
where
    T: NumOps
{
    data: T,
}

impl<T> Scalar<T>
where
    T: NumOps
{
    pub fn new(data: T) -> Self {
        Self { data }
    }

    pub fn data(&self) -> &T {
        &self.data
    }

    pub fn data_type(&self) -> &str {
        std::any::type_name::<T>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_type_sample() {
        let usize_scalar = Scalar::new(1usize);
        let isize_scalar = Scalar::new(1isize);
        let u8_scalar = Scalar::new(1u8);
        let u16_scalar = Scalar::new(1u16);
        let u32_scalar = Scalar::new(1u32);
        let u64_scalar = Scalar::new(1u64);
        let u128_scalar = Scalar::new(1u128);
        let i8_scalar = Scalar::new(1i8);
        let i16_scalar = Scalar::new(1i16);
        let i32_scalar = Scalar::new(1i32);
        let i64_scalar = Scalar::new(1i64);
        let i128_scalar = Scalar::new(1i128);
        let f32_scalar = Scalar::new(1f32);
        let f64_scalar = Scalar::new(1f64);

        assert_eq!(usize_scalar.data_type(), "usize");
        assert_eq!(isize_scalar.data_type(), "isize");
        assert_eq!(u8_scalar.data_type(), "u8");
        assert_eq!(u16_scalar.data_type(), "u16");
        assert_eq!(u32_scalar.data_type(), "u32");
        assert_eq!(u64_scalar.data_type(), "u64");
        assert_eq!(u128_scalar.data_type(), "u128");
        assert_eq!(i8_scalar.data_type(), "i8");
        assert_eq!(i16_scalar.data_type(), "i16");
        assert_eq!(i32_scalar.data_type(), "i32");
        assert_eq!(i64_scalar.data_type(), "i64");
        assert_eq!(i128_scalar.data_type(), "i128");
        assert_eq!(f32_scalar.data_type(), "f32");
        assert_eq!(f64_scalar.data_type(), "f64");
    }
}
