use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum KdezeroError {
    #[error("Error: {0}")]
    #[allow(dead_code)]
    Error(String),
    #[error("NotFoundError: {0} is not found in {1}")]
    NotFoundError(String, String),
    #[error("SizeError: {0} is not equal to {1}. {2}")]
    SizeError(String, usize, usize),
    #[error("NotImplementedTypeError: {0} is not implemented for {1}")]
    NotImplementedTypeError(String, String),
    #[error("NotCollectTypeError: {0} is not collect type. Expected {1}")]
    NotCollectTypeError(String, String),
    #[error("ExistError: {0} is already exist in {1}")]
    ExistError(String, String),
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use super::*;

    fn error() -> Result<()> {
        Err(KdezeroError::Error("Error".to_string()).into())
    }

    #[test]
    fn kdezero_error() -> Result<()> {
        match error() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KdezeroError>().context("downcast error")?;
                assert_eq!(e.to_string(), "Error: Error");
                Ok(())
            }
        }
    }
}
