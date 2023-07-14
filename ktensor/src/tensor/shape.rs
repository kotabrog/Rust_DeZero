use anyhow::Result;
use super::Tensor;

impl<T> Tensor<T>
where
    T: Clone,
{
    /// Reshape the tensor
    /// 
    /// # Arguments
    /// 
    /// * `shape` - The new shape of the tensor
    /// 
    /// # Returns
    /// 
    /// * `Result<Self>` - Result of the reshape
    /// 
    /// # Note
    /// 
    /// If the shape is not correct, `TensorError::ShapeError` is returned
    pub fn reshape<U: Into<Vec<usize>>>(&self, shape: U) -> Result<Self> {
        let shape = shape.into();
        Self::check_shape(&self.data, &shape)?;
        Ok(Self {
            data: self.data.clone(),
            shape: shape.to_vec(),
        })
    }

    /// Transpose the tensor
    pub fn transpose(&self) -> Self {
        let mut shape = self.shape.clone();
        shape.reverse();
        let mut new_tensor = Self::new(
            self.data.clone(), shape
        ).unwrap();

        let mut index = vec![0; self.ndim()];
        for _ in 0..self.data.len() {
            let value = self.at(&index).unwrap();
            let mut new_index = index.clone();
            new_index.reverse();
            *new_tensor.at_mut(&new_index).unwrap() = value.clone();

            for j in 0..self.ndim() {
                index[j] += 1;
                if index[j] < self.shape[j] {
                    break;
                }
                index[j] = 0;
            }
        }
        new_tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::TensorError;

    #[test]
    fn reshape_normal() {
        let x = Tensor::new([0.0, 1.0, 2.0, 3.0], [2, 2]).unwrap();
        let x = x.reshape([4]).unwrap();
        assert_eq!(x.get_shape(), &[4]);
        assert_eq!(x.get_data(), &[0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn reshape_zero_dim() {
        let x = Tensor::new([0.0], []).unwrap();
        let x = x.reshape([1]).unwrap();
        assert_eq!(x.get_shape(), &[1]);
        assert_eq!(x.get_data(), &[0.0]);
    }

    #[test]
    fn reshape_error_mismatch() {
        let x = Tensor::new([0.0, 1.0, 2.0, 3.0], [2, 2]).unwrap();
        let x = x.reshape([3]);
        match x {
            Ok(_) => panic!("Should be error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::ShapeError(4, 3))
            }
        }
    }

    #[test]
    fn transpose_normal() {
        let x = Tensor::new([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [2, 3]).unwrap();
        let x = x.transpose();
        assert_eq!(x.get_shape(), &[3, 2]);
        assert_eq!(x.get_data(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn transpose_zero_dim() {
        let x = Tensor::new([0.0], []).unwrap();
        let x = x.transpose();
        assert_eq!(x.get_shape(), &[]);
        assert_eq!(x.get_data(), &[0.0]);
    }

    #[test]
    fn transpose_1d() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        let x = x.transpose();
        assert_eq!(x.get_shape(), &[3]);
        assert_eq!(x.get_data(), &[0.0, 1.0, 2.0]);
    }
}
