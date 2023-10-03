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
    operator: Option<OperatorContentsWrapper>,
}

impl Operator {
    pub fn new(id: usize, node: Option<usize>, params: Vec<usize>, operator: Box<dyn OperatorContents>) -> Self {
        Self {
            id,
            node,
            params,
            operator: Some(operator.into()),
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

    pub(crate) fn check_params_len(&self, params_len: usize) -> Result<&Vec<usize>> {
        if self.params.len() != params_len {
            return Err(
                KdezeroError::SizeError(
                    "params".to_string(),
                    params_len,
                    self.params.len()
                ).into()
            );
        }
        Ok(&self.params)
    }

    pub(crate) fn change_variable_id(&mut self, old_id: usize, new_id: usize) -> Result<()> {
        for param in self.params.iter_mut() {
            if *param == old_id {
                *param = new_id;
            }
        }
        Ok(())
    }
}

impl Clone for Operator {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            node: self.node,
            params: self.params.clone(),
            operator: self.operator.clone(),
        }
    }
}
