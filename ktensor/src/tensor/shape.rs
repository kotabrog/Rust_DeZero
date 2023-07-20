use anyhow::Result;
use super::Tensor;
use crate::error::TensorError;

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
    /// 
    /// # Returns
    /// 
    /// * `Self` - Transposed tensor
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

    /// Broadcast the tensor
    /// 
    /// # Arguments
    /// 
    /// * `shape` - The new shape of the tensor
    /// 
    /// # Returns
    /// 
    /// * `Result<Self>` - Result of the broadcast
    /// 
    /// # Note
    /// 
    /// If the shape is not correct, following errors are returned:
    /// 
    /// * `TensorError::ShapeError` - If the shape is not correct
    /// * `TensorError::ShapeSizeError` - If the shape size is not correct
    pub fn broadcast_to<U: Into<Vec<usize>>>(&self, shape: U) -> Result<Self> {
        fn in_zero(shape: &Vec<usize>) -> bool {
            for i in shape {
                if *i == 0 {
                    return true
                }
            }
            false
        }

        fn check_shape(shape: &Vec<usize>, target_shape: &Vec<usize>) -> Result<()> {
            if in_zero(shape) || in_zero(target_shape) {
                return Err(TensorError::ShapeError(
                    shape.clone(), target_shape.clone()
                ).into())
            }
            if target_shape.len() < shape.len() {
                return Err(TensorError::ShapeSizeError(
                    shape.len(), target_shape.len()
                ).into())
            }
            Ok(())
        }

        fn update_data<V: Clone>(data: Vec<V>, shape: &Vec<usize>,
                                 target_index: usize, size: usize) -> (Vec<V>, usize) {
            let mut new_data = Vec::new();
            let mut index = 0;
            while index < data.len() {
                for _ in 0..shape[target_index] {
                    new_data.extend(data[index..index + size].iter().map(|x| x.clone()))
                }
                index += size;
            }
            (new_data, size * shape[target_index])
        }

        let shape = shape.into();
        check_shape(&self.shape, &shape)?;
        let mut self_shape= vec![1; shape.len() - self.ndim()];
        self_shape.extend(self.get_shape().iter());

        let mut data = self.data.clone();
        let mut size = 1;
        for i in (0..self_shape.len()).rev() {
            if shape[i] == self_shape[i] {
                size *= shape[i];
                continue;
            }
            if self_shape[i] != 1 {
                return Err(TensorError::ShapeError(
                    self.shape.clone(), shape
                ).into())
            }
            (data, size) = update_data(data, &shape, i, size);
        }

        Ok(Self { data, shape: shape.to_vec() })
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
                assert_eq!(e, TensorError::ShapeSizeError(4, 3))
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

    #[test]
    fn broadcast_to_normal() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        let x = x.broadcast_to([2, 3]).unwrap();
        assert_eq!(x.get_shape(), &[2, 3]);
        assert_eq!(x.get_data(), &[0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn broadcast_to_3d() {
        let x = Tensor::new([0.0, 1.0, 2.0, 3.0], [2, 1, 2]).unwrap();
        let x = x.broadcast_to([2, 2, 2]).unwrap();
        assert_eq!(x.get_shape(), &[2, 2, 2]);
        assert_eq!(x.get_data(), &[0.0, 1.0, 0.0, 1.0,
                                    2.0, 3.0, 2.0, 3.0]);
    }

    #[test]
    fn broadcast_to_add_left() {
        let x = Tensor::new([0.0, 1.0], [2,]).unwrap();
        let x = x.broadcast_to([3, 2]).unwrap();
        assert_eq!(x.get_shape(), &[3, 2]);
        assert_eq!(x.get_data(), &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn broadcast_to_scaler() {
        let x = Tensor::new([0.0], []).unwrap();
        let x = x.broadcast_to([3, 2]).unwrap();
        assert_eq!(x.get_shape(), &[3, 2]);
        assert_eq!(x.get_data(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn broadcast_to_0d() {
        let x = Tensor::new([0.0], []).unwrap();
        let x = x.broadcast_to([]).unwrap();
        assert_eq!(x.get_shape(), &[]);
        assert_eq!(x.get_data(), &[0.0]);
    }

    #[test]
    fn broadcast_to_error_mismatch_ndim() {
        let x = Tensor::new([0.0, 1.0], [2, 1]).unwrap();
        let x = x.broadcast_to([2]);
        match x {
            Ok(_) => panic!("Should be error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::ShapeSizeError(2, 1))
            }
        }
    }

    #[test]
    fn broadcast_to_error_mismatch_shape() {
        let x = Tensor::new([0.0, 1.0], [2, 1]).unwrap();
        let x = x.broadcast_to([4, 2]);
        match x {
            Ok(_) => panic!("Should be error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::ShapeError(vec![2, 1], vec![4, 2]))
            }
        }
    }

    #[test]
    fn broadcast_to_error_mismatch_shape_left() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        let x = x.broadcast_to([3, 2]);
        match x {
            Ok(_) => panic!("Should be error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::ShapeError(vec![3,], vec![3, 2]))
            }
        }
    }

    #[test]
    fn broadcast_to_error_to_shape_0() {
        let x = Tensor::new([0.0], [1]).unwrap();
        let x = x.broadcast_to([0, 1]);
        match x {
            Ok(_) => panic!("Should be error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::ShapeError(vec![1], vec![0, 1]))
            }
        }
    }

    #[test]
    fn broadcast_to_error_shape_0() {
        let x = Tensor::<f32>::new([], [0]).unwrap();
        let x = x.broadcast_to([1]);
        match x {
            Ok(_) => panic!("Should be error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::ShapeError(vec![0], vec![1]))
            }
        }
    }
}
