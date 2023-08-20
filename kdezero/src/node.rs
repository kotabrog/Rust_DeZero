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

    pub fn get_variable_id(&self) -> Result<usize> {
        self.data.get_variable_id()
    }

    pub fn get_operator_id(&self) -> Result<usize> {
        self.data.get_operator_id()
    }

    /// Get outputs of the Node.
    pub fn get_outputs(&self) -> &Vec<usize> {
        &self.outputs
    }

    // pub(crate) fn get_data_mut(&mut self) -> &mut NodeData {
    //     &mut self.data
    // }

    pub(crate) fn set_inputs(&mut self, inputs: Vec<usize>) {
        self.inputs = inputs;
    }

    pub(crate) fn set_outputs(&mut self, outputs: Vec<usize>) {
        self.outputs = outputs;
    }

    pub(crate) fn set_id(&mut self, id: usize) {
        self.id = id;
    }

    pub(crate) fn set_data(&mut self, data: NodeData) {
        self.data = data;
    }

    pub(crate) fn add_input(&mut self, input: usize) {
        self.inputs.push(input);
    }

    pub(crate) fn add_output(&mut self, output: usize) {
        self.outputs.push(output);
    }

    pub(crate) fn change_input_and_output_node_id(&mut self, old_id: usize, new_id: usize) {
        for input in self.inputs.iter_mut() {
            if *input == old_id {
                *input = new_id;
            }
        }
        for output in self.outputs.iter_mut() {
            if *output == old_id {
                *output = new_id;
            }
        }
    }

    pub(crate) fn check_inputs_len(&self, len: usize) -> Result<()> {
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

    pub(crate) fn check_outputs_len(&self, len: usize) -> Result<()> {
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

    pub(crate) fn check_inputs_len_at_least(&self, len: usize) -> Result<()> {
        if self.inputs.len() < len {
            return Err(
                KdezeroError::SizeSmallError(
                    "inputs".to_string(),
                    len,
                    self.inputs.len()
                ).into()
            );
        }
        Ok(())
    }

    pub(crate) fn check_outputs_len_at_least(&self, len: usize) -> Result<()> {
        if self.outputs.len() < len {
            return Err(
                KdezeroError::SizeSmallError(
                    "outputs".to_string(),
                    len,
                    self.outputs.len()
                ).into()
            );
        }
        Ok(())
    }

    pub(crate) fn move_output(self) -> Vec<usize> {
        self.outputs
    }
}
