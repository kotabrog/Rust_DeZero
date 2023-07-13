use anyhow::Result;
use crate::error::TensorError;

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
    /// Check the shape of the tensor
    /// 
    /// # Arguments
    /// 
    /// * `data` - The data of the tensor
    /// * `shape` - The shape of the tensor
    /// 
    /// # Returns
    /// 
    /// * `Result<()>` - Result of the check
    /// 
    /// # Note
    /// 
    /// The shape of the tensor is checked by the following rules:
    /// 
    /// * If the shape is empty, the data length must be 1
    /// * If the shape is not empty, the product of the shape must be equal to the data length
    /// 
    /// If the shape is not correct, `TensorError::ShapeError` is returned
    fn check_shape(data: &Vec<T>, shape: &Vec<usize>) -> Result<()> {
        if shape.len() == 0 {
            if data.len() != 1 {
                return Err(TensorError::ShapeError(data.len(), 0).into())
            }
        } else {
            let size = shape.iter().product();
            if size != data.len() {
                return Err(TensorError::ShapeError(data.len(), size).into())
            }
        }
        Ok(())
    }

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
    fn check_shape_normal() {
        Tensor::check_shape(
            &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &vec![2, 3]
        ).unwrap()
    }

    #[test]
    #[should_panic]
    fn check_shape_error_mismatch() {
        Tensor::check_shape(
            &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &vec![2, 2]
        ).unwrap()
    }

    #[test]
    fn check_shape_zero_dim() {
        Tensor::check_shape(
            &vec![1.0],
            &vec![]
        ).unwrap()
    }

    #[test]
    fn check_shape_empty() {
        Tensor::<f32>::check_shape(
            &vec![],
            &vec![1, 0, 2]
        ).unwrap()
    }

    #[test]
    #[should_panic]
    fn check_shape_error_empty_mismatch() {
        Tensor::<f32>::check_shape(
            &vec![],
            &vec![]
        ).unwrap()
    }

    #[test]
    fn new_normal() -> Result<()> {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,])?;
        assert_eq!(x.get_data(), &vec![0.0, 1.0, 2.0]);
        assert_eq!(x.get_shape(), &vec![3]);
        Ok(())
    }

    #[test]
    fn new_zero_dim() -> Result<()> {
        let x = Tensor::new([1.0], [])?;
        assert_eq!(x.get_data(), &vec![1.0]);
        assert_eq!(x.get_shape(), &vec![]);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn new_error_mismatch_shape() {
        let _ = Tensor::new([0.0, 1.0, 2.0], [2,]).unwrap();
    }

    #[test]
    fn new_from_vec() -> Result<()> {
        let x = Tensor::new(
            vec![0.0, 1.0, 2.0],
            vec![3,]
        )?;
        assert_eq!(x.get_data(), &vec![0.0, 1.0, 2.0]);
        assert_eq!(x.get_shape(), &vec![3]);
        Ok(())
    }
}