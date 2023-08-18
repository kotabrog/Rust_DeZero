use anyhow::Result;
use super::Variables;
use crate::error::KdezeroError;

impl Variables {
    pub(crate) fn check_id_in_variables(&self, id: usize) -> Result<()> {
        if self.variables.contains_key(&id) {
            return Err(
                KdezeroError::ExistError(
                    id.to_string(),
                    "Variables".to_string()
                ).into()
            );
        }
        Ok(())
    }

    pub(crate) fn check_id_not_in_variables(&self, id: usize) -> Result<()> {
        if !self.variables.contains_key(&id) {
            return Err(
                KdezeroError::NotFoundError(
                    id.to_string(),
                    "Variables".to_string()
                ).into()
            );
        }
        Ok(())
    }
}
