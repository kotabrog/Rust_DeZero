pub mod operators;
pub mod operator_contents;

pub use operator_contents::{OperatorContents, OperatorContentsWrapper};
pub use operators::Operators;

use anyhow::Result;
use crate::error::KdezeroError;

pub struct Operator {
    id: usize,
    node: Option<usize>,
    operator: Option<OperatorContentsWrapper>,
}

impl Operator {
    pub fn new(id: usize, node: Option<usize>, operator: Box<dyn OperatorContents>) -> Self {
        Self {
            id,
            node,
            operator: Some(operator.into()),
        }
    }

    pub fn get_id(&self) -> usize {
        self.id
    }

    pub fn get_node(&self) -> Option<usize> {
        self.node
    }

    pub fn get_operator(&self) -> &Option<OperatorContentsWrapper> {
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

    pub(crate) fn set_operator(&mut self, operator: OperatorContentsWrapper) {
        self.operator = Some(operator);
    }

    // pub(crate) fn take_operator(&mut self) -> Option<OperatorContentsWrapper> {
    //     self.operator.take()
    // }

    pub(crate) fn take_operator_result(&mut self) -> Result<OperatorContentsWrapper> {
        match self.operator.take() {
            Some(operator) => Ok(operator),
            None => Err(
                KdezeroError::OperatorError(
                    "operator contents is not set".to_string()
                ).into()
            )
        }
    }
}

impl Clone for Operator {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            node: self.node,
            operator: self.operator.clone(),
        }
    }
}
