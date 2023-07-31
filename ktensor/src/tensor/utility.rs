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
    /// If the shape is not correct, `TensorError::ShapeSizeError` is returned
    pub(crate) fn check_shape(data: &Vec<T>, shape: &Vec<usize>) -> Result<()> {
        if shape.len() == 0 {
            if data.len() != 1 {
                return Err(TensorError::ShapeSizeError(data.len(), 0).into())
            }
        } else {
            let size = shape.iter().product();
            if size != data.len() {
                return Err(TensorError::ShapeSizeError(data.len(), size).into())
            }
        }
        Ok(())
    }

    /// Calculate the data index from the shape index
    /// 
    /// # Arguments
    /// 
    /// * `index` - The shape index
    /// 
    /// # Returns
    /// 
    /// * `Result<usize>` - Result of the calculation
    /// 
    /// # Note
    /// 
    /// If the shape index is not correct, `TensorError::IndexError` is returned
    pub(crate) fn calc_data_index<U: AsRef<[usize]>>(&self, index: U) -> Result<usize> {
        let index = index.as_ref();
        if index.len() != self.ndim() {
            return Err(TensorError::IndexError(
                self.get_shape().clone(),
                index.to_vec()
            ).into())
        }
        for i in 0..self.ndim() {
            if index[i] >= self.shape[i] {
                return Err(TensorError::IndexError(
                    self.get_shape().clone(),
                    index.to_vec()
                ).into())
            }
        }
        let mut data_index = 0;
        for i in 0..self.ndim() {
            data_index += index[i];
            if i < self.ndim() - 1 {
                data_index *= self.shape[i + 1];
            }
        }
        Ok(data_index)
    }

    /// Convert the data index to the shape index
    /// 
    /// # Arguments
    /// 
    /// * `index` - The data index
    /// * `shape` - The shape of the tensor
    /// 
    /// # Returns
    /// 
    /// * `Vec<usize>` - The shape index
    /// 
    /// # Note
    /// 
    /// The behavior when the index is out of range is not defined.
    pub(crate) fn data_index_to_indexes(index: usize, shape: &Vec<usize>) -> Vec<usize> {
        let mut indexes = Vec::new();
        let mut index = index;
        for i in (0..shape.len()).rev() {
            indexes.push(index % shape[i]);
            index /= shape[i];
        }
        indexes.iter().rev().cloned().collect()
    }
}

impl<T> Tensor<T>
where
    T: Clone
{
    /// Adapt the function to each element of the tensor
    /// 
    /// # Arguments
    /// 
    /// * `f` - The function to be adapted
    pub(crate) fn iter_func(&self, f: impl Fn(T) -> T) -> Self {
        Self {
            data: self.data
                .iter()
                .map(|x| f(x.clone()))
                .collect(),
            shape: self.shape.clone(),
        }
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
                assert_eq!(e, TensorError::ShapeSizeError(6, 4))
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

    #[test]
    fn calc_data_index_normal() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11., 12.],
            vec![2, 2, 3]
        ).unwrap();
        assert_eq!(x.calc_data_index(&[0, 0, 0]).unwrap(), 0);
        assert_eq!(x.calc_data_index(&[0, 0, 1]).unwrap(), 1);
        assert_eq!(x.calc_data_index(&[0, 0, 2]).unwrap(), 2);
        assert_eq!(x.calc_data_index(&[0, 1, 0]).unwrap(), 3);
        assert_eq!(x.calc_data_index(&[0, 1, 1]).unwrap(), 4);
        assert_eq!(x.calc_data_index(&[0, 1, 2]).unwrap(), 5);
        assert_eq!(x.calc_data_index(&[1, 0, 0]).unwrap(), 6);
        assert_eq!(x.calc_data_index(&[1, 0, 1]).unwrap(), 7);
        assert_eq!(x.calc_data_index(&[1, 0, 2]).unwrap(), 8);
        assert_eq!(x.calc_data_index(&[1, 1, 0]).unwrap(), 9);
        assert_eq!(x.calc_data_index(&[1, 1, 1]).unwrap(), 10);
        assert_eq!(x.calc_data_index(&[1, 1, 2]).unwrap(), 11);
    }

    #[test]
    fn calc_data_index_error_ndim() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2]
        ).unwrap();
        let x = x.calc_data_index(&[0, 0, 0]);
        match x {
            Ok(_) => panic!("Should be error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::IndexError(vec![2, 2], vec![0, 0, 0]))
            }
        }
    }

    #[test]
    fn calc_data_index_zero_dim() {
        let x = Tensor::new(
            vec![1.0],
            vec![]
        ).unwrap();
        assert_eq!(x.calc_data_index(&[]).unwrap(), 0);
    }

    #[test]
    fn calc_data_index_error_out_index() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2]
        ).unwrap();
        let x = x.calc_data_index(&[0, 3]);
        match x {
            Ok(_) => panic!("Should be error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::IndexError(vec![2, 2], vec![0, 3]))
            }
        }
    }

    #[test]
    fn data_index_to_indexes_normal() {
        assert_eq!(Tensor::<f64>::data_index_to_indexes(0, &vec![2, 3]), vec![0, 0]);
        assert_eq!(Tensor::<f64>::data_index_to_indexes(1, &vec![2, 3]), vec![0, 1]);
        assert_eq!(Tensor::<f64>::data_index_to_indexes(2, &vec![2, 3]), vec![0, 2]);
        assert_eq!(Tensor::<f64>::data_index_to_indexes(3, &vec![2, 3]), vec![1, 0]);
        assert_eq!(Tensor::<f64>::data_index_to_indexes(4, &vec![2, 3]), vec![1, 1]);
        assert_eq!(Tensor::<f64>::data_index_to_indexes(5, &vec![2, 3]), vec![1, 2]);
    }
}
