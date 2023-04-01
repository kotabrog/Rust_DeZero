/// Scalar is a wrapper of a single value
/// 
/// # Fields
/// 
/// * `data` - The value of the scalar
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Scaler<T>
{
    data: T,
}

impl<T> Scaler<T>
{
    /// Create a new Scaler
    pub fn new(data: T) -> Self {
        Self { data }
    }

    /// Get the data
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Get the data type
    pub fn data_type(&self) -> &str {
        std::any::type_name::<T>()
    }
}

impl<T> From<T> for Scaler<T>
{
    fn from(data: T) -> Self {
        Self { data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_normal() {
        let x = Scaler::<f32>::new(1.0);
        assert_eq!(*x.data(), 1.0);
        assert_eq!(x.data_type(), "f32");
    }

    #[test]
    fn from_normal() {
        let x = Scaler::<f32>::new(1.0);
        assert_eq!(x, Scaler::<f32>::from(1.0));
        assert_eq!(x, 1.0.into());
        assert_eq!(x.data_type(), "f32");
    }
}
