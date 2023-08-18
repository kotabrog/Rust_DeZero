use std::collections::HashMap;
use anyhow::Result;
use super::{Node, Graph};
use crate::error::KdezeroError;

impl Graph {
    pub fn get_nodes(&self) -> &HashMap<usize, Node> {
        &self.nodes
    }

    pub fn get_next_id(&self) -> usize {
        self.next_id
    }

    pub fn get_node(&self, id: usize) -> Result<&Node> {
        self.check_id_not_in_nodes(id)?;
        Ok(self.nodes.get(&id).unwrap())
    }

    pub fn get_node_from_name(&self, name: &str) -> Result<&Node> {
        for node in self.nodes.values() {
            if node.get_name() == name {
                return Ok(node);
            }
        }
        Err(KdezeroError::NotFoundError(
            name.to_string(),
            "Graph".to_string()
        ).into())
    }

    pub(crate) fn get_node_mut(&mut self, id: usize) -> Result<&mut Node> {
        self.check_id_not_in_nodes(id)?;
        Ok(self.nodes.get_mut(&id).unwrap())
    }

    pub(crate) fn get_nodes_mut(&mut self) -> &mut HashMap<usize, Node> {
        &mut self.nodes
    }

    // pub(crate) fn get_node_data_mut(&mut self, id: usize) -> Result<&mut NodeData> {
    //     self.check_id_not_in_nodes(id)?;
    //     let node = self.nodes.get_mut(&id).unwrap();
    //     Ok(node.get_data_mut())
    // }

    pub(crate) fn move_all_node(self) -> HashMap<usize, Node> {
        self.nodes
    }
}
