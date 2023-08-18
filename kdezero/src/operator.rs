pub mod operators;
pub mod operator_contents;

pub use operator_contents::{OperatorContents, OperatorContentsWrapper};
pub use operators::Operators;

use anyhow::Result;
use crate::error::KdezeroError;

pub struct Operator {
    id: usize,
    node: Option<usize>,
    params: Vec<usize>,
    operator: Box<dyn OperatorContents>,
}

impl Operator {
    pub fn new(id: usize, node: Option<usize>, params: Vec<usize>, operator: Box<dyn OperatorContents>) -> Self {
        Self {
            id,
            node,
            params,
            operator,
        }
    }

    pub fn get_id(&self) -> usize {
        self.id
    }

    pub fn get_node(&self) -> Option<usize> {
        self.node
    }

    pub fn get_params(&self) -> &Vec<usize> {
        &self.params
    }

    pub fn get_operator(&self) -> &Box<dyn OperatorContents> {
        &self.operator
    }

    pub fn get_node_id(&self) -> Result<usize> {
        match self.node {
            Some(node_id) => Ok(node_id),
            None => Err(
                KdezeroError::OperatorError(
                    "node id is not set".to_string()
                ).into()
            )
        }
    }

    pub(crate) fn set_node(&mut self, node: Option<usize>) {
        self.node = node;
    }

    pub fn get_forward_set(&self) -> Result<(usize, OperatorContentsWrapper)> {
        Ok((
            self.get_node_id()?,
            OperatorContentsWrapper::new(self.operator.clone_operator())
        ))
    }

    pub fn get_backward_set(&self) -> Result<(usize, OperatorContentsWrapper)> {
        Ok((
            self.get_node_id()?,
            OperatorContentsWrapper::new(self.operator.clone_operator())
        ))
    }
}
