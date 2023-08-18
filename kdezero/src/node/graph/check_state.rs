use anyhow::Result;
use super::Graph;
use crate::error::KdezeroError;

impl Graph {
    pub(crate) fn check_id_in_nodes(&self, id: usize) -> Result<()> {
        if self.nodes.contains_key(&id) {
            return Err(
                KdezeroError::ExistError(
                    id.to_string(),
                    "Graph".to_string()
                ).into()
            );
        }
        Ok(())
    }

    pub(crate) fn check_id_not_in_nodes(&self, id: usize) -> Result<()> {
        if !self.nodes.contains_key(&id) {
            return Err(
                KdezeroError::NotFoundError(
                    id.to_string(),
                    "Graph".to_string()
                ).into()
            );
        }
        Ok(())
    }
}
