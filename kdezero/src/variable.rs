pub mod variables;
pub mod variable_data;

pub use variables::Variables;
pub use variable_data::VariableData;

use anyhow::Result;
use crate::error::KdezeroError;

#[derive(Debug, Clone, PartialEq)]
pub struct Variable {
    id: usize,
    node: Option<usize>,
    data: VariableData,
    grad: Option<usize>,
}

impl Variable {
    pub fn new(id: usize, node: Option<usize>, data: VariableData) -> Self {
        Self {
            id,
            node,
            data,
            grad: None,
        }
    }

    pub fn get_id(&self) -> usize {
        self.id
    }

    pub fn get_node(&self) -> Option<usize> {
        self.node
    }

    pub fn get_data(&self) -> &VariableData {
        &self.data
    }

    pub fn get_grad(&self) -> Option<usize> {
        self.grad
    }

    pub(crate) fn get_grad_id(&self) -> Result<usize> {
        self.grad.ok_or_else(
            || KdezeroError::NotFoundError(
                "grad".to_string(),
                "Variable".to_string()
            ).into()
        )
    }

    pub fn get_shape(&self) -> Result<&Vec<usize>> {
        self.data.get_shape()
    }

    pub fn ndim(&self) -> Result<usize> {
        self.data.ndim()
    }

    pub fn size(&self) -> Result<usize> {
        self.data.size()
    }

    pub fn get_type(&self) -> String {
        self.data.to_string()
    }

    pub fn set_data(&mut self, data: VariableData) {
        self.data = data;
    }

    pub(crate) fn set_grad(&mut self, grad: Option<usize>) {
        self.grad = grad;
    }

    pub(crate) fn set_node(&mut self, node: Option<usize>) {
        self.node = node;
    }

    pub(crate) fn set_id(&mut self, id: usize) {
        self.id = id;
    }

    pub fn clear_grad(&mut self) {
        self.grad = None;
    }
}
