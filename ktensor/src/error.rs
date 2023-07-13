// use anyhow::{Context, Result};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("SampleError: {0}")]
    #[allow(dead_code)]
    Sample(String),
    #[error("ShapeError: data: {0}, shape: {1}")]
    ShapeError(usize, usize),
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
        Err(TensorError::ShapeError(1, 2).into())
    }

    #[test]
    fn tensor_error_shape() -> Result<()> {
        match error_shape() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(e.to_string(), "ShapeError: data: 1, shape: 2");
                Ok(())
            }
        }
    }
}
