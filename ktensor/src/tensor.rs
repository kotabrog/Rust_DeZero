mod utility;
mod info;
mod data_access;
mod shape;
mod create;
mod sum;
mod ops;
mod matmul;
mod math;
mod random;
mod convert;
pub mod iter;

use anyhow::Result;
pub use random::TensorRng;

/// Tensor
/// 
/// # Fields
/// 
/// * `data` - The data of the tensor
/// * `shape` - The shape of the tensor
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T>
{
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T> Tensor<T>
{
    /// Create a new Tensor
    /// 
    /// # Arguments
    /// 
    /// * `data` - The data of the tensor
    /// * `shape` - The shape of the tensor
    /// 
    /// # Returns
    /// 
    /// * `Result<Self>` - Result of the creation
    /// 
    /// # Note
    /// 
    /// If the shape is not correct, `TensorError::ShapeError` is returned
    pub fn new<U: Into<Vec<T>>, V: Into<Vec<usize>>>(data: U, shape: V) -> Result<Self> {
        let data = data.into();
        let shape = shape.into();
        Self::check_shape(&data, &shape)?;
        Ok(Self { data, shape })
    }

    /// Get the data of the tensor
    pub fn get_data(&self) -> &Vec<T> {
        &self.data
    }

    /// Get the shape of the tensor
    pub fn get_shape(&self) -> &Vec<usize> {
        &self.shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_normal() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        assert_eq!(x.get_data(), &vec![0.0, 1.0, 2.0]);
        assert_eq!(x.get_shape(), &vec![3]);
    }

    #[test]
    fn new_zero_dim() {
        let x = Tensor::new([1.0], []).unwrap();
        assert_eq!(x.get_data(), &vec![1.0]);
        assert_eq!(x.get_shape(), &vec![]);
    }

    #[test]
    #[should_panic]
    fn new_error_mismatch_shape() {
        let _ = Tensor::new([0.0, 1.0, 2.0], [2,]).unwrap();
    }

    #[test]
    fn new_from_vec() {
        let x = Tensor::new(
            vec![0.0, 1.0, 2.0],
            vec![3,]
        ).unwrap();
        assert_eq!(x.get_data(), &vec![0.0, 1.0, 2.0]);
        assert_eq!(x.get_shape(), &vec![3]);
    }
}
