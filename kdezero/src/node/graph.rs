use anyhow::Result;
use std::collections::HashMap;
use super::{Node, NodeData};
use crate::error::KdezeroError;

pub struct Graph {
    nodes: HashMap<usize, Node>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    pub fn get_nodes(&self) -> &HashMap<usize, Node> {
        &self.nodes
    }

    pub fn add_node(&mut self, node: Node) -> Option<Node>{
        self.nodes.insert(node.get_id(), node)
    }

    pub fn get_node(&self, id: usize) -> Result<&Node> {
        match self.nodes.get(&id) {
            Some(node) => Ok(node),
            None => Err(
                KdezeroError::NotFoundError(
                    id.to_string(),
                    "Graph".to_string()
                ).into()
            ),
        }
    }

    pub fn add_new_node(
        &mut self, id: usize, name: String,
        data: NodeData, inputs: Vec<usize>, outputs: Vec<usize>)
    -> Result<()> {
        if self.nodes.contains_key(&id) {
            return Err(
                KdezeroError::ExistError(
                    id.to_string(),
                    "Graph".to_string()
                ).into()
            );
        }
        let node = Node::new(id, name, data, inputs, outputs);
        self.nodes.insert(id, node);
        Ok(())
    }
}
