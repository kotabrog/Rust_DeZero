pub mod operators;
pub mod operator_contents;
pub mod layer;
pub mod contents;

pub use operator_contents::{OperatorContents, OperatorContentsWrapper};
pub use operators::Operators;
pub use layer::Layer;
pub use contents::Contents;

use anyhow::Result;
use crate::error::KdezeroError;

#[derive(Clone)]
pub struct Operator {
    id: usize,
    node: Option<usize>,
    operator: Contents,
}

impl Operator {
    pub fn new(id: usize, node: Option<usize>, operator: Contents) -> Self {
        Self {
            id,
            node,
            operator,
        }
    }

    pub fn get_id(&self) -> usize {
        self.id
    }

    pub fn get_node(&self) -> Option<usize> {
        self.node
    }

    pub fn get_operator(&self) -> &Contents {
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

    pub(crate) fn set_operator(&mut self, operator: Contents) {
        self.operator = operator;
    }

    // pub(crate) fn take_operator(&mut self) -> Option<OperatorContentsWrapper> {
    //     self.operator.take()
    // }

    pub(crate) fn take_operator(&mut self) -> Contents {
        let contents = self.operator.take();
        contents
    }
}
