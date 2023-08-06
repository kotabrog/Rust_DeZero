pub mod operators;

use std::collections::HashMap;
use anyhow::Result;
use crate::variable::Variable;
use crate::node::Node;

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
}

pub trait OperatorContents {
    fn forward(
        &self, node_id: usize,
        nodes: &HashMap<usize, Node>,
        variables: &mut HashMap<usize, Variable>,
    ) -> Result<Vec<usize>>;
}
