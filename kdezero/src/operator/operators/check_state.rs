use anyhow::Result;
use super::Operators;
use crate::error::KdezeroError;

impl Operators {
    pub(crate) fn check_id_in_operators(&self, id: usize) -> Result<()> {
        if self.operators.contains_key(&id) {
            return Err(
                KdezeroError::ExistError(
                    id.to_string(),
                    "Operators".to_string()
                ).into()
            );
        }
        Ok(())
    }

    pub(crate) fn check_id_not_in_operators(&self, id: usize) -> Result<()> {
        if !self.operators.contains_key(&id) {
            return Err(
                KdezeroError::NotFoundError(
                    id.to_string(),
                    "Operators".to_string()
                ).into()
            );
        }
        Ok(())
    }

}
