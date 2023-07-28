use std::collections::HashSet;
use num_traits::NumAssign;
use anyhow::Result;
use super::Tensor;
use crate::error::TensorError;

impl<T> Tensor<T>
where
    T: NumAssign + Clone,
{
    /// Sum all the elements of the tensor
    pub fn sum_all(&self) -> T {
        self.data.iter()
            .fold(T::zero(), |acc, x| acc + x.clone())
    }

    /// Sum the elements of the tensor along the specified axis
    /// 
    /// # Arguments
    /// 
    /// * `axis` - The axis along which the sum is performed
    /// * `keepdims` - Whether to keep the dimension of the tensor
    /// 
    /// # Returns
    /// 
    /// * `Self` - The result of the sum
    pub fn sum<U: AsRef<[usize]>>(&self, axis: Option<U>, keepdims: bool) -> Self {
        fn make_axis(axis: Option<&[usize]>, ndim: usize) -> HashSet<usize> {
            match axis {
                None => (0..ndim).collect(),
                Some(axis) => {
                    match axis.len() {
                        0 => HashSet::new(),
                        _ => axis.iter().cloned().collect(),
                    }
                }
            }
        }

        fn make_shape(shape: &[usize], axis: &HashSet<usize>, keepdims: bool) -> Vec<usize> {
            if axis.is_empty() {
                return shape.to_vec()
            }
            let mut new_shape = Vec::new();
            for i in 0..shape.len() {
                if axis.contains(&i) {
                    if keepdims {
                        new_shape.push(1);
                    }
                } else {
                    new_shape.push(shape[i]);
                }
            }
            new_shape
        }

        let axis = axis.as_ref().map(|x| x.as_ref());
        let axis = make_axis(axis, self.ndim());
        let new_shape = make_shape(&self.shape, &axis, true);
        let mut data = vec![T::zero(); new_shape.iter().product()];
        for (i, value) in self.data.iter().enumerate() {
            let indexes = Tensor::<T>::data_index_to_indexes(i, &self.shape);
            let mut index = 0;
            let mut size = 1;
            for j in (0..self.ndim()).rev() {
                if axis.contains(&j) {
                    continue;
                }
                index += indexes[j] * size;
                size *= new_shape[j];
            }
            data[index] += value.clone();
        }
        let new_shape = make_shape(&self.shape, &axis, keepdims);
        Self { data, shape: new_shape }
    }

    /// Sum the values for the given shape
    /// 
    /// # Arguments
    /// 
    /// * `shape` - The shape to sum
    /// 
    /// # Returns
    /// 
    /// * `Result<Self>` - Result of the sum
    /// 
    /// # Note
    /// 
    /// If the shape is not correct, following errors are returned:
    /// 
    /// * `TensorError::ShapeError` - If the shape is not correct
    /// * `TensorError::ShapeSizeError` - If the shape size is not correct
    pub fn sum_to<U: AsRef<[usize]>>(&self, shape: U) -> Result<Self> {
        let shape = shape.as_ref();
        if shape.len() > self.ndim() {
            return Err(TensorError::ShapeSizeError(self.ndim(), shape.len()).into());
        }
        let diff = self.ndim() - shape.len();
        let mut axis = (0..diff).collect::<Vec<_>>();
        for i in 0..shape.len() {
            if shape[i] == 1 {
                axis.push(i + diff);
            } else if shape[i] != self.shape[i + diff] {
                return Err(TensorError::ShapeError(self.shape.clone(), shape.to_vec()).into());
            }
        }
        let mut tensor = self.sum(Some(axis), true);
        tensor.shape = shape.to_vec();
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_all_normal() {
        let x = Tensor::new([0, 1, 2], [3,]).unwrap();
        assert_eq!(x.sum_all(), 3);
        let x = Tensor::new([0, 1, 2, 3], [2, 2]).unwrap();
        assert_eq!(x.sum_all(), 6);
    }

    #[test]
    fn sum_all_zero_dim() {
        let x = Tensor::new([0], []).unwrap();
        assert_eq!(x.sum_all(), 0);
    }

    #[test]
    fn sum_all_zero_shape() {
        let x = Tensor::<i32>::new([], [0, 1]).unwrap();
        assert_eq!(x.sum_all(), 0);
    }

    #[test]
    fn sum_all_float() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        assert_eq!(x.sum_all(), 3.0);
        let x = Tensor::new([0.0, 1.0, 2.0, 3.0], [2, 2]).unwrap();
        assert_eq!(x.sum_all(), 6.0);
    }

    #[test]
    fn sum_normal() {
        let x = Tensor::<f32>::arrange([2, 3]).unwrap();
        let y = x.sum(Some([0]), false);
        assert_eq!(y.get_data(), &vec![3.0, 5.0, 7.0]);
        assert_eq!(y.get_shape(), &vec![3]);
    }

    #[test]
    fn sum_axis_1() {
        let x = Tensor::<f32>::arrange([2, 3]).unwrap();
        let y = x.sum(Some([1]), false);
        assert_eq!(y.get_data(), &vec![3.0, 12.0]);
        assert_eq!(y.get_shape(), &vec![2]);
    }

    #[test]
    fn sum_3_dim_2_axis() {
        let x = Tensor::<f32>::arrange([2, 3, 4]).unwrap();
        let y = x.sum(Some([0, 2]), false);
        assert_eq!(y.get_data(), &vec![60.0, 92.0, 124.0]);
        assert_eq!(y.get_shape(), &vec![3]);
    }

    #[test]
    fn sum_keepdims() {
        let x = Tensor::<f32>::arrange([2, 3, 4]).unwrap();
        let y = x.sum(Some([0, 2]), true);
        assert_eq!(y.get_data(), &vec![60.0, 92.0, 124.0]);
        assert_eq!(y.get_shape(), &vec![1, 3, 1]);
    }

    #[test]
    fn sum_empty_axis() {
        let x = Tensor::<f32>::arrange([2, 3, 4]).unwrap();
        let y = x.sum(Some([]), false);
        assert_eq!(y, Tensor::<f32>::arrange([2, 3, 4]).unwrap());
    }

    #[test]
    fn sum_full_axis() {
        let x = Tensor::<f32>::arrange([2, 3, 4]).unwrap();
        let y = x.sum(Some([0, 1, 2]), false);
        assert_eq!(y.get_data(), &vec![276.0]);
        assert_eq!(y.get_shape(), &vec![]);
    }

    #[test]
    fn sum_unrelated_axis() {
        let x = Tensor::<f32>::arrange([2, 3, 4]).unwrap();
        let y = x.sum(Some([0, 2, 3]), false);
        assert_eq!(y.get_data(), &vec![60.0, 92.0, 124.0]);
        assert_eq!(y.get_shape(), &vec![3]);
    }

    #[test]
    fn sum_none_axis() {
        let x = Tensor::<f32>::arrange([2, 3, 4]).unwrap();
        let y = x.sum::<&[usize]>(None, false);
        assert_eq!(y.get_data(), &vec![276.0]);
        assert_eq!(y.get_shape(), &vec![]);
    }

    #[test]
    fn sum_to_normal() {
        let x = Tensor::<f32>::arrange([2, 3]).unwrap();
        let y = x.sum_to([2, 1]).unwrap();
        assert_eq!(y.get_data(), &vec![3.0, 12.0]);
        assert_eq!(y.get_shape(), &vec![2, 1]);
    }

    #[test]
    fn sum_to_3d() {
        let x = Tensor::<f32>::arrange([2, 3, 4]).unwrap();
        let y = x.sum_to([2, 1, 4]).unwrap();
        assert_eq!(y.get_data(), &vec![12.0, 15.0, 18.0, 21.0, 48.0, 51.0, 54.0, 57.0]);
        assert_eq!(y.get_shape(), &vec![2, 1, 4]);
    }

    #[test]
    fn sum_to_add_left() {
        let x = Tensor::<f32>::arrange([3, 2]).unwrap();
        let y = x.sum_to([2]).unwrap();
        assert_eq!(y.get_data(), &vec![6.0, 9.0]);
        assert_eq!(y.get_shape(), &vec![2]);
    }

    #[test]
    fn sum_to_scaler() {
        let x = Tensor::<f32>::arrange([3, 2]).unwrap();
        let y = x.sum_to([]).unwrap();
        assert_eq!(y.get_data(), &vec![15.0]);
        assert_eq!(y.get_shape(), &vec![]);
    }

    #[test]
    fn sum_to_0d() {
        let x = Tensor::<f32>::arrange([]).unwrap();
        let y = x.sum_to([]).unwrap();
        assert_eq!(y.get_data(), &vec![0.0]);
        assert_eq!(y.get_shape(), &vec![]);
    }

    #[test]
    fn sum_to_error_mismatch_ndim() {
        let x = Tensor::<f32>::arrange([3, 2]).unwrap();
        let y = x.sum_to([2, 1, 4]);
        match y {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::ShapeSizeError(2, 3));
            }
        }
    }

    #[test]
    fn sum_to_error_mismatch_shape() {
        let x = Tensor::<f32>::arrange([3, 2]).unwrap();
        let y = x.sum_to([2, 1]);
        match y {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::ShapeError(vec![3, 2], vec![2, 1]));
            }
        }
    }

    #[test]
    fn suu_to_error_mismatch_shape_left() {
        let x = Tensor::<f32>::arrange([3, 2]).unwrap();
        let y = x.sum_to([3]);
        match y {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::ShapeError(vec![3, 2], vec![3]));
            }
        }
    }
}
