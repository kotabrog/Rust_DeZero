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

    pub(crate) fn check_inputs_len(&self, node_id: usize, len: usize) -> Result<()> {
        self.get_node(node_id)?
            .check_inputs_len(len)
    }

    pub(crate) fn check_outputs_len(&self, node_id: usize, len: usize) -> Result<()> {
        self.get_node(node_id)?
            .check_outputs_len(len)
    }

    pub(crate) fn check_inputs_len_at_least(&self, node_id: usize, len: usize) -> Result<()> {
        self.get_node(node_id)?
            .check_inputs_len_at_least(len)
    }

    pub(crate) fn check_outputs_len_at_least(&self, node_id: usize, len: usize) -> Result<()> {
        self.get_node(node_id)?
            .check_outputs_len_at_least(len)
    }
}
