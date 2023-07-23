// use anyhow::{Context, Result};
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum TensorError {
    #[error("SampleError: {0}")]
    #[allow(dead_code)]
    Sample(String),
    #[error("ShapeError: data: {0:?}, shape: {1:?}")]
    ShapeError(Vec<usize>, Vec<usize>),
    #[error("ShapeSizeError: data: {0}, shape: {1}")]
    ShapeSizeError(usize, usize),
    #[error("IndexError: shape: {0:?}, index: {1:?}")]
    IndexError(Vec<usize>, Vec<usize>),
    #[error("CastError: type: {0}")]
    CastError(String),
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use super::*;

    fn error_sample() -> Result<()> {
        Err(TensorError::Sample("SampleError".to_string()).into())
    }

    #[test]
    fn tensor_error_sample() -> Result<()> {
        match error_sample() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(e.to_string(), "SampleError: SampleError");
                Ok(())
            }
        }
    }

    fn error_shape() -> Result<()> {
        Err(TensorError::ShapeError(vec![1, 2], vec![3, 4]).into())
    }

    #[test]
    fn tensor_error_shape() -> Result<()> {
        match error_shape() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(e.to_string(), "ShapeError: data: [1, 2], shape: [3, 4]");
                Ok(())
            }
        }
    }

    fn error_shape_size() -> Result<()> {
        Err(TensorError::ShapeSizeError(1, 2).into())
    }

    #[test]
    fn tensor_error_shape_size() -> Result<()> {
        match error_shape_size() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(e.to_string(), "ShapeSizeError: data: 1, shape: 2");
                Ok(())
            }
        }
    }

    fn error_index() -> Result<()> {
        Err(TensorError::IndexError(vec![1, 2], vec![3, 4]).into())
    }

    #[test]
    fn tensor_error_index() -> Result<()> {
        match error_index() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(e.to_string(), "IndexError: shape: [1, 2], index: [3, 4]");
                Ok(())
            }
        }
    }

    fn error_cast() -> Result<()> {
        Err(TensorError::CastError("Vec<i32>".to_string()).into())
    }

    #[test]
    fn tensor_error_cast() -> Result<()> {
        match error_cast() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(e.to_string(), "CastError: type: Vec<i32>");
                Ok(())
            }
        }
    }
}
