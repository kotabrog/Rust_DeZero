// use anyhow::{Context, Result};
use thiserror::Error;

#[derive(Debug, Error)]
enum TensorError {
    #[error("SampleError: {0}")]
    #[allow(dead_code)]
    Sample(String),
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use super::*;

    fn error() -> Result<()> {
        Err(TensorError::Sample("SampleError".to_string()).into())
    }

    #[test]
    fn tensor_error_sample() -> Result<()> {
        match error() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(e.to_string(), "SampleError: SampleError");
                Ok(())
            }
        }
    }
}
