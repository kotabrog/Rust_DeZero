use anyhow::Result;
use num_traits::{NumCast, Zero, One};
use super::Tensor;
use crate::error::TensorError;

impl<T> Tensor<T>
where
    T: NumCast,
{
    /// Create a tensor with the specified shape with evenly spaced values
    /// 
    /// # Arguments
    /// 
    /// * `shape` - The shape of the tensor
    /// 
    /// # Returns
    /// 
    /// * `Result<Self>` - Result of the creation
    /// 
    /// # Note
    /// 
    /// If value cannot be cast to `T`, `TensorError::CastError` is returned
    pub fn arrange<U: Into<Vec<usize>>>(shape: U) -> Result<Self> {
        let shape = shape.into();
        let size: usize = shape.iter().product();
        if size == 0 {
            return Ok(Self {
                data: vec![],
                shape,
            })
        }
        let data: Result<Vec<_>, _> = (0..size)
            .map(|i| NumCast::from(i)
                .ok_or_else(|| TensorError::CastError(
                    std::any::type_name::<T>().to_string())))
            .collect();
        let data = data?;
        Ok(Self { data, shape })
    }
}

impl<T> Tensor<T>
where
    T: Zero + Clone
{
    /// Create a tensor with the specified shape with all values set to 0
    /// 
    /// # Arguments
    /// 
    /// * `shape` - The shape of the tensor
    pub fn zeros<U: Into<Vec<usize>>>(shape: U) -> Self {
        let shape = shape.into();
        let size: usize = shape.iter().product();
        Self {
            data: vec![T::zero(); size],
            shape,
        }
    }

    /// Create a tensor with the same shape as the argument with all values set to 0
    /// 
    /// # Arguments
    /// 
    /// * `tensor` - The tensor to be used as a reference for the shape
    pub fn zeros_like(tensor: &Self) -> Self {
        Self::zeros(tensor.shape.clone())
    }
}

impl<T> Tensor<T>
where
    T: One + Clone
{
    /// Create a tensor with the specified shape with all values set to 1
    /// 
    /// # Arguments
    /// 
    /// * `shape` - The shape of the tensor
    pub fn ones<U: Into<Vec<usize>>>(shape: U) -> Self {
        let shape = shape.into();
        let size: usize = shape.iter().product();
        Self {
            data: vec![T::one(); size],
            shape,
        }
    }

    /// Create a tensor with the same shape as the argument with all values set to 1
    /// 
    /// # Arguments
    /// 
    /// * `tensor` - The tensor to be used as a reference for the shape
    pub fn ones_like(tensor: &Self) -> Self {
        Self::ones(tensor.shape.clone())
    }
}

impl<T> Tensor<T>
where
    T: Clone
{
    /// Create a tensor with the specified shape with all values set to the specified value
    /// 
    /// # Arguments
    /// 
    /// * `value` - The value to be used for the tensor
    /// * `shape` - The shape of the tensor
    pub fn full<U: Into<Vec<usize>>>(value: T, shape: U) -> Self {
        let shape = shape.into();
        let size: usize = shape.iter().product();
        Self {
            data: vec![value; size],
            shape,
        }
    }

    /// Create a tensor with the same shape as the argument with all values set to the specified value
    /// 
    /// # Arguments
    /// 
    /// * `value` - The value to be used for the tensor
    /// * `tensor` - The tensor to be used as a reference for the shape
    pub fn full_like(value: T, tensor: &Self) -> Self {
        Self::full(value, tensor.shape.clone())
    }
}

impl<T> Tensor<T> {
    /// Create a scalar tensor with the specified value
    /// 
    /// # Arguments
    /// 
    /// * `value` - The value to be used for the tensor
    pub fn scalar(value: T) -> Self {
        Self {
            data: vec![value],
            shape: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrange_normal() {
        let x = Tensor::<i32>::arrange([3, 2]).unwrap();
        assert_eq!(x.get_data(), &vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(x.get_shape(), &vec![3, 2]);
    }

    #[test]
    fn arrange_zero_dim() {
        let x = Tensor::<i32>::arrange([]).unwrap();
        assert_eq!(x.get_data(), &vec![0]);
        assert_eq!(x.get_shape(), &vec![]);
    }

    #[test]
    fn arrange_zero_shape() {
        let x = Tensor::<i32>::arrange([0, 1]).unwrap();
        assert_eq!(x.get_data(), &vec![]);
        assert_eq!(x.get_shape(), &vec![0, 1]);
    }

    #[test]
    fn arrange_float() {
        let x = Tensor::<f32>::arrange([3, 2]).unwrap();
        assert_eq!(x.get_data(), &vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(x.get_shape(), &vec![3, 2]);
    }

    #[test]
    fn arrange_error_cast() {
        let x = Tensor::<i8>::arrange([129]);
        match x {
            Ok(_) => panic!("Should be error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::CastError("i8".to_string()))
            }
        }
    }

    #[test]
    fn zeros_normal() {
        let x = Tensor::<i32>::zeros([3, 2]);
        assert_eq!(x.get_data(), &vec![0, 0, 0, 0, 0, 0]);
        assert_eq!(x.get_shape(), &vec![3, 2]);
    }

    #[test]
    fn zeros_empty() {
        let x = Tensor::<i32>::zeros([]);
        assert_eq!(x.get_data(), &vec![0]);
        assert_eq!(x.get_shape(), &vec![]);
    }

    #[test]
    fn zeros_zero_shape() {
        let x = Tensor::<i32>::zeros([0, 1]);
        assert_eq!(x.get_data(), &vec![]);
        assert_eq!(x.get_shape(), &vec![0, 1]);
    }

    #[test]
    fn zeros_like_normal() {
        let x = Tensor::new([1, 2, 3, 4, 5, 6], [3, 2]).unwrap();
        let y = Tensor::<i32>::zeros_like(&x);
        assert_eq!(y.get_data(), &vec![0, 0, 0, 0, 0, 0]);
        assert_eq!(y.get_shape(), &vec![3, 2]);
    }

    #[test]
    fn ones_normal() {
        let x = Tensor::<i32>::ones([3, 2]);
        assert_eq!(x.get_data(), &vec![1, 1, 1, 1, 1, 1]);
        assert_eq!(x.get_shape(), &vec![3, 2]);
    }

    #[test]
    fn ones_empty() {
        let x = Tensor::<i32>::ones([]);
        assert_eq!(x.get_data(), &vec![1]);
        assert_eq!(x.get_shape(), &vec![]);
    }

    #[test]
    fn ones_zero_shape() {
        let x = Tensor::<i32>::ones([0, 1]);
        assert_eq!(x.get_data(), &vec![]);
        assert_eq!(x.get_shape(), &vec![0, 1]);
    }

    #[test]
    fn ones_like_normal() {
        let x = Tensor::new([1, 2, 3, 4, 5, 6], [3, 2]).unwrap();
        let y = Tensor::<i32>::ones_like(&x);
        assert_eq!(y.get_data(), &vec![1, 1, 1, 1, 1, 1]);
        assert_eq!(y.get_shape(), &vec![3, 2]);
    }

    #[test]
    fn full_normal() {
        let x = Tensor::<i32>::full(10, [3, 2]);
        assert_eq!(x.get_data(), &vec![10, 10, 10, 10, 10, 10]);
        assert_eq!(x.get_shape(), &vec![3, 2]);
    }

    #[test]
    fn full_empty() {
        let x = Tensor::<i32>::full(10, []);
        assert_eq!(x.get_data(), &vec![10]);
        assert_eq!(x.get_shape(), &vec![]);
    }

    #[test]
    fn full_zero_shape() {
        let x = Tensor::<i32>::full(10, [0, 1]);
        assert_eq!(x.get_data(), &vec![]);
        assert_eq!(x.get_shape(), &vec![0, 1]);
    }

    #[test]
    fn full_like_normal() {
        let x = Tensor::new([1, 2, 3, 4, 5, 6], [3, 2]).unwrap();
        let y = Tensor::<i32>::full_like(10, &x);
        assert_eq!(y.get_data(), &vec![10, 10, 10, 10, 10, 10]);
        assert_eq!(y.get_shape(), &vec![3, 2]);
    }

    #[test]
    fn scalar_normal() {
        let x = Tensor::<i32>::scalar(10);
        assert_eq!(x.get_data(), &vec![10]);
        assert_eq!(x.get_shape(), &vec![]);
    }
}
