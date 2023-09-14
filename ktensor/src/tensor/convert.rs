use anyhow::Result;
use num_traits::NumCast;
use super::Tensor;
use crate::error::TensorError;

impl<T> Tensor<T>
where
    T: Clone,
{
    /// Convert to scalar when scalar
    /// 
    /// # Returns
    /// 
    /// * `Result<T>` - Result of the conversion
    /// 
    /// # Note
    /// 
    /// If the tensor is not scalar, `TensorError::NotScalarError` is returned
    pub fn to_scalar(&self) -> Result<T> {
        if self.is_scalar() {
            Ok(self.data[0].clone())
        } else {
            Err(
                TensorError::NotScalarError(
                    self.shape.clone()
                ).into()
            )
        }
    }

    /// Convert to vector when vector
    /// 
    /// # Returns
    /// 
    /// * `Result<Vec<T>>` - Result of the conversion
    /// 
    /// # Note
    /// 
    /// If the tensor is not vector, `TensorError::NotVectorError` is returned
    pub fn to_vector(&self) -> Result<Vec<T>> {
        if self.is_vector() {
            Ok(self.data.clone())
        } else {
            Err(
                TensorError::NotVectorError(
                    self.shape.clone()
                ).into()
            )
        }
    }
}

impl<T> Tensor<T>
where
    T: NumCast,
{
    /// Convert to tensor of the specified type
    pub fn as_type<U: NumCast>(self) -> Result<Tensor<U>> {
        let data: Result<Vec<_>, _> = self.data.into_iter()
            .map(|x| NumCast::from(x)
                .ok_or_else(|| TensorError::CastError(
                    std::any::type_name::<U>().to_string())))
            .collect();
        let data = data?;
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_scalar_normal() {
        let tensor = Tensor::new(vec![1], vec![]).unwrap();
        assert_eq!(tensor.to_scalar().unwrap(), 1);
    }

    #[test]
    fn to_scalar_error() {
        let tensor = Tensor::new(vec![1, 2], vec![2]).unwrap();
        match tensor.to_scalar() {
            Ok(_) => panic!("Should be error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::NotScalarError(vec![2]))
            }
        }
    }

    #[test]
    fn to_vector_normal() {
        let tensor = Tensor::new(vec![1, 2], vec![2]).unwrap();
        assert_eq!(tensor.to_vector().unwrap(), vec![1, 2]);
    }

    #[test]
    fn to_vector_error() {
        let tensor = Tensor::new(vec![1], vec![]).unwrap();
        match tensor.to_vector() {
            Ok(_) => panic!("Should be error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::NotVectorError(vec![]))
            }
        }
    }

    #[test]
    fn as_type_normal() {
        let tensor = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let tensor = tensor.as_type::<f32>().unwrap();
        assert_eq!(tensor.get_data(), &vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tensor.get_shape(), &vec![2, 2]);
        assert_eq!(tensor.data_type(), "f32");
    }

    #[test]
    fn as_type_error() {
        let tensor = Tensor::new(vec![255, 256, 257, 258], vec![2, 2]).unwrap();
        match tensor.as_type::<i8>() {
            Ok(_) => panic!("Should be error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::CastError("i8".to_string()))
            }
        }
    }
}
