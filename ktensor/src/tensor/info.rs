use super::Tensor;

impl<T> Tensor<T>
{
    /// Get the number of dimensions of the tensor
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the size of the tensor
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get the type of the tensor data
    pub fn data_type(&self) -> &str {
        std::any::type_name::<T>()
    }

    /// Check if the tensor is a scalar
    pub fn is_scalar(&self) -> bool {
        self.ndim() == 0
    }

    /// Check if the tensor is a vector
    pub fn is_vector(&self) -> bool {
        self.ndim() == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndim_normal() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        assert_eq!(x.ndim(), 1);
        let x = Tensor::new([0.0, 1.0, 2.0], [3, 1]).unwrap();
        assert_eq!(x.ndim(), 2);
        let x = Tensor::new([0.0, 1.0, 2.0], [3, 1, 1]).unwrap();
        assert_eq!(x.ndim(), 3);
    }

    #[test]
    fn ndin_zero_dim() {
        let x = Tensor::new([0.0], []).unwrap();
        assert_eq!(x.ndim(), 0);
    }

    #[test]
    fn size_normal() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        assert_eq!(x.size(), 3);
        let x = Tensor::new([0.0, 1.0, 2.0, 3.0], [2, 2]).unwrap();
        assert_eq!(x.size(), 4);
    }

    #[test]
    fn size_zero_dim() {
        let x = Tensor::new([0.0], []).unwrap();
        assert_eq!(x.size(), 1);
    }

    #[test]
    fn data_type_normal() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        assert_eq!(x.data_type(), "f64");
    }

    #[test]
    fn is_scalar_normal() {
        let x = Tensor::new([0.0], []).unwrap();
        assert!(x.is_scalar());
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        assert!(!x.is_scalar());
    }

    #[test]
    fn is_vector_normal() {
        let x = Tensor::new([0.0], []).unwrap();
        assert!(!x.is_vector());
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        assert!(x.is_vector());
        let x = Tensor::new([0.0, 1.0, 2.0], [3, 1]).unwrap();
        assert!(!x.is_vector());
    }
}
