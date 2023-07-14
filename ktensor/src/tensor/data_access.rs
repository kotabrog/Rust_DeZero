use anyhow::Result;
use super::Tensor;

impl<T> Tensor<T>
{
    /// Get the value of the tensor at the specified index
    /// 
    /// # Arguments
    /// 
    /// * `indexes` - The indexes of the tensor
    /// 
    /// # Returns
    /// 
    /// * `Result<&T>` - Result of the access
    /// 
    /// # Note
    /// 
    /// If the indexes are not correct, `TensorError::IndexError` is returned
    pub fn at<U: AsRef<[usize]>>(&self, indexes: U) -> Result<&T> {
        let index = self.calc_data_index(indexes)?;
        Ok(&self.data[index])
    }

    /// Get the mutable value of the tensor at the specified index
    /// 
    /// # Arguments
    /// 
    /// * `indexes` - The indexes of the tensor
    /// 
    /// # Returns
    /// 
    /// * `Result<&mut T>` - Result of the access
    /// 
    /// # Note
    /// 
    /// If the indexes are not correct, `TensorError::IndexError` is returned
    pub fn at_mut<U: AsRef<[usize]>>(&mut self, indexes: U) -> Result<&mut T> {
        let index = self.calc_data_index(indexes)?;
        Ok(&mut self.data[index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::TensorError;

    #[test]
    fn at_normal() {
        let x = Tensor::new(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [3, 2]
        ).unwrap();
        assert_eq!(x.at([0, 1]).unwrap(), &1.0);
        assert_eq!(x.at([1, 1]).unwrap(), &3.0);
        assert_eq!(x.at([2, 0]).unwrap(), &4.0);
    }

    #[test]
    fn at_scaler() {
        let x = Tensor::new([0.0], []).unwrap();
        assert_eq!(x.at([]).unwrap(), &0.0);
    }

    #[test]
    fn at_error() {
        let x = Tensor::new(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [3, 2]
        ).unwrap();
        let x = x.at([3, 2]);
        match x {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::IndexError(vec![3, 2], vec![3, 2]));
            }
        }
    }

    #[test]
    fn at_mut_normal() {
        let mut x = Tensor::new(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [3, 2]
        ).unwrap();
        *x.at_mut([0, 1]).unwrap() = 10.0;
        assert_eq!(x.at([0, 1]).unwrap(), &10.0);
    }

    #[test]
    fn at_mut_scaler() {
        let mut x = Tensor::new([0.0], []).unwrap();
        *x.at_mut([]).unwrap() = 10.0;
        assert_eq!(x.at([]).unwrap(), &10.0);
    }

    #[test]
    fn at_mut_error() {
        let mut x = Tensor::new(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [3, 2]
        ).unwrap();
        let x = x.at_mut([3, 2]);
        match x {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::IndexError(vec![3, 2], vec![3, 2]));
            }
        }
    }
}
