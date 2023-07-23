use anyhow::Result;
use num_traits::NumCast;
use super::Tensor;
use crate::error::TensorError;

impl<T> Tensor<T>
where
    T: NumCast,
{
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
}
