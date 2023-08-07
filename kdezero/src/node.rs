pub mod graph;
pub mod node_data;

pub use node_data::NodeData;
pub use graph::Graph;

use anyhow::Result;
use crate::error::KdezeroError;

#[derive(Debug, Clone)]
pub struct Node {
    id: usize,
    name: String,
    data: NodeData,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

impl Node {
    /// Create a new Node instance.
    /// 
    /// # Arguments
    /// 
    /// * `id` - Node ID
    /// * `name` - Node name
    pub fn new(id: usize, name: String, data: NodeData, inputs: Vec<usize>, outputs: Vec<usize>) -> Self {
        Self {
            id,
            name,
            data,
            inputs,
            outputs,
        }
    }

    /// Get the ID of the Node.
    pub fn get_id(&self) -> usize {
        self.id
    }

    /// Get the name of the Node.
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Get the data of the Node.
    pub fn get_data(&self) -> &NodeData {
        &self.data
    }

    /// Get inputs of the Node.
    pub fn get_inputs(&self) -> &Vec<usize> {
        &self.inputs
    }

    /// Get outputs of the Node.
    pub fn get_outputs(&self) -> &Vec<usize> {
        &self.outputs
    }

    pub fn check_inputs_len(&self, len: usize) -> Result<()> {
        if self.inputs.len() != len {
            return Err(
                KdezeroError::SizeError(
                    "inputs".to_string(),
                    len,
                    self.inputs.len()
                ).into()
            );
        }
        Ok(())
    }

    pub fn check_outputs_len(&self, len: usize) -> Result<()> {
        if self.outputs.len() != len {
            return Err(
                KdezeroError::SizeError(
                    "outputs".to_string(),
                    len,
                    self.outputs.len()
                ).into()
            );
        }
        Ok(())
    }
}
