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
    #[error("SizeSmallError: {0} is smaller than {1}. {2}")]
    SizeSmallError(String, usize, usize),
    #[error("NotScalarError: {0} is not scalar. {1:?}")]
    NotScalarError(String, Vec<usize>),
    #[error("NotImplementedTypeError: {0} is not implemented for {1}")]
    NotImplementedTypeError(String, String),
    #[error("NotCollectTypeError: {0} is not collect type. Expected {1}")]
    NotCollectTypeError(String, String),
    #[error("NotSetError: {0} is not set")]
    NotSetError(String),
    #[error("ExistError: {0} is already exist in {1}")]
    ExistError(String, String),
    #[error("NotCollectGraphError: Not collect graph. {0}")]
    NotCollectGraphError(String),
    #[error("OperatorError: {0}")]
    OperatorError(String),
    #[error("DuplicateError: {0} is duplicated in {1}")]
    DuplicateError(String, String),
    #[error("OverflowError: {0} is overflow")]
    OverflowError(String),
    #[error("PrameterError: {0} is not a suitable parameter for {1}. Expect {2}")]
    ParameterError(String, String, String),
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
