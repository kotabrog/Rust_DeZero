/// Scalar is a wrapper of single value
/// 
/// # Fields
/// 
/// * `value` - The value of the scalar
#[derive(Debug, Clone, PartialEq)]
pub struct Scalar<T>
{
    value: T,
}

impl<T> Scalar<T>
{
    /// Create a new Scalar
    pub fn new(value: T) -> Self {
        Self { value }
    }

    /// Get the value
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Get the value type
    pub fn value_type(&self) -> &str {
        std::any::type_name::<T>()
    }
}

impl<T: Copy> Copy for Scalar<T> {}

impl<T> From<T> for Scalar<T>
{
    fn from(value: T) -> Self {
        Self { value }
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

        assert_eq!(usize_scalar.value_type(), "usize");
        assert_eq!(isize_scalar.value_type(), "isize");
        assert_eq!(u8_scalar.value_type(), "u8");
        assert_eq!(u16_scalar.value_type(), "u16");
        assert_eq!(u32_scalar.value_type(), "u32");
        assert_eq!(u64_scalar.value_type(), "u64");
        assert_eq!(u128_scalar.value_type(), "u128");
        assert_eq!(i8_scalar.value_type(), "i8");
        assert_eq!(i16_scalar.value_type(), "i16");
        assert_eq!(i32_scalar.value_type(), "i32");
        assert_eq!(i64_scalar.value_type(), "i64");
        assert_eq!(i128_scalar.value_type(), "i128");
        assert_eq!(f32_scalar.value_type(), "f32");
        assert_eq!(f64_scalar.value_type(), "f64");

        assert_eq!(usize_scalar.value(), &1usize);
        assert_eq!(isize_scalar.value(), &1isize);
        assert_eq!(u8_scalar.value(), &1u8);
        assert_eq!(u16_scalar.value(), &1u16);
        assert_eq!(u32_scalar.value(), &1u32);
        assert_eq!(u64_scalar.value(), &1u64);
        assert_eq!(u128_scalar.value(), &1u128);
        assert_eq!(i8_scalar.value(), &1i8);
        assert_eq!(i16_scalar.value(), &1i16);
        assert_eq!(i32_scalar.value(), &1i32);
        assert_eq!(i64_scalar.value(), &1i64);
        assert_eq!(i128_scalar.value(), &1i128);
        assert_eq!(f32_scalar.value(), &1f32);
        assert_eq!(f64_scalar.value(), &1f64);
    }

    #[test]
    fn scalar_copy() {
        fn copy_scalar<T>(scalar: Scalar<T>) -> Scalar<T> {
            scalar
        }
        let usize_scalar = Scalar::new(1usize);
        let usize_scalar_copy = copy_scalar(usize_scalar);
        assert_eq!(usize_scalar.value(), &1usize);
        assert_eq!(usize_scalar_copy.value(), &1usize);

        let box_scalar = Scalar::new(Box::new(1usize));
        let box_scalar_copy = copy_scalar(box_scalar);
        // The following is an error
        // assert_eq!(**box_scalar.value(), 1usize);
        assert_eq!(**box_scalar_copy.value(), 1usize);
    }

    #[test]
    fn from_and_into() {
        let usize_scalar = Scalar::from(1usize);
        assert_eq!(usize_scalar, 1.into());
    }
}
