pub mod operators;
pub mod operator_contents;

pub use operator_contents::OperatorContents;
pub use operators::Operators;

use anyhow::Result;
use crate::variable::Variables;
use crate::node::Graph;
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

    pub fn forward(
        &self, graph: &Graph, variables: &mut Variables,
    ) -> Result<Vec<usize>> {
        let node_id = match self.node {
            Some(node_id) => node_id,
            None => return Err(
                KdezeroError::OperatorError(
                    "node id is not set".to_string()
                ).into()
            )
        };
        self.operator.forward(node_id, graph, variables)
    }
}
