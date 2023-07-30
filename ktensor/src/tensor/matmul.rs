use num_traits::NumAssign;
use anyhow::Result;
use super::Tensor;
use crate::error::TensorError;

impl<T> Tensor<T>
where
    T: NumAssign + Clone
{
    /// Matrix multiplication
    /// 
    /// # Arguments
    /// 
    /// * `rhs` - The right hand side of the matrix multiplication
    /// 
    /// # Returns
    /// 
    /// * `Result<Self>` - Result of the matrix multiplication
    /// 
    /// # Note
    /// 
    /// If the shape is not correct, following errors are returned:
    /// 
    /// * `TensorError::ShapeSizeError` - If the shape size is not correct
    /// * `TensorError::ShapeError` - If the shape is not correct
    /// * `TensorError::Error` - If the shape is zero
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        if self.ndim() != 2 {
            return Err(TensorError::ShapeSizeError(self.ndim(), 2).into())
        }
        if rhs.ndim() != 2 {
            return Err(TensorError::ShapeSizeError(rhs.ndim(), 2).into())
        }
        if self.shape[1] != rhs.shape[0] {
            return Err(TensorError::ShapeError(self.shape.clone(), rhs.shape.clone()).into())
        }
        if self.shape[0] == 0 || self.shape[1] == 0 {
            return Err(TensorError::Error(format!("The shape is zero: {:?}", self.shape).to_string()).into())
        }
        if rhs.shape[1] == 0 {
            return Err(TensorError::Error(format!("The shape is zero: {:?}", rhs.shape).to_string()).into())
        }
        let mut data = vec![T::zero(); self.shape[0] * rhs.shape[1]];
        for i in 0..self.shape[0] {
            for j in 0..rhs.shape[1] {
                for k in 0..self.shape[1] {
                    data[i * rhs.shape[1] + j] += self.data[i * self.shape[1] + k].clone() * rhs.data[k * rhs.shape[1] + j].clone();
                }
            }
        }
        Ok(Self{ data, shape: vec![self.shape[0], rhs.shape[1]] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_normal() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3, 1]).unwrap();
        let y = Tensor::new([3.0, 4.0, 5.0], [1, 3]).unwrap();
        let z = x.matmul(&y).unwrap();
        assert_eq!(z.get_data(), &vec![0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
        assert_eq!(z.get_shape(), &vec![3, 3]);
    }

    #[test]
    fn matmul_non_one() {
        let x = Tensor::<i32>::arrange([3, 2]).unwrap();
        let y = Tensor::arrange([2, 2]).unwrap();
        let z = x.matmul(&y).unwrap();
        assert_eq!(z.get_data(), &vec![2, 3, 6, 11, 10, 19]);
        assert_eq!(z.get_shape(), &vec![3, 2]);
    }

    #[test]
    fn matmul_error_shape_zero() {
        let x = Tensor::<i32>::arrange([3, 0]).unwrap();
        let y = Tensor::arrange([0, 2]).unwrap();
        let z = x.matmul(&y);
        match z {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::Error("The shape is zero: [3, 0]".to_string()));
            }
        }
    }

    #[test]
    fn matmul_error_mismatch_shape() {
        let x = Tensor::<i32>::arrange([3, 2]).unwrap();
        let y = Tensor::arrange([3, 2]).unwrap();
        let z = x.matmul(&y);
        match z {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::ShapeError(vec![3, 2], vec![3, 2]));
            }
        }
    }

    #[test]
    fn mutmul_error_mismatch_ndim_left() {
        let x = Tensor::<i32>::arrange([3, 2, 1]).unwrap();
        let y = Tensor::arrange([2, 2]).unwrap();
        let z = x.matmul(&y);
        match z {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::ShapeSizeError(3, 2));
            }
        }
    }

    #[test]
    fn mutmul_error_mismatch_ndim_right() {
        let x = Tensor::<i32>::arrange([3, 2]).unwrap();
        let y = Tensor::arrange([2, 2, 1]).unwrap();
        let z = x.matmul(&y);
        match z {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::ShapeSizeError(3, 2));
            }
        }
    }
}
