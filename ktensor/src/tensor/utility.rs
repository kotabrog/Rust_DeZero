use anyhow::Result;
use super::Tensor;
use crate::error::TensorError;

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
    pub(crate) fn check_shape(data: &Vec<T>, shape: &Vec<usize>) -> Result<()> {
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
    fn check_shape_error_mismatch() {
        let x = Tensor::check_shape(
            &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &vec![2, 2]
        );
        match x {
            Ok(_) => panic!("Should be error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::ShapeError(6, 4))
            }
        }
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
}
