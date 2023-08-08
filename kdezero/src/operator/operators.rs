use anyhow::Result;
use std::collections::HashMap;
use super::{Operator, OperatorContents};
use crate::error::KdezeroError;

pub struct Operators {
    operators: HashMap<usize, Operator>,
}

impl Operators {
    pub fn new() -> Self {
        Self {
            operators: HashMap::new(),
        }
    }

    pub fn get_operators(&self) -> &HashMap<usize, Operator> {
        &self.operators
    }

    pub fn add_operator(&mut self, operator: Operator) -> Option<Operator> {
        self.operators.insert(operator.get_id(), operator)
    }

    pub fn get_operator(&self, id: usize) -> Result<&Operator> {
        match self.operators.get(&id) {
            Some(operator) => Ok(operator),
            None => Err(
                KdezeroError::NotFoundError(
                    id.to_string(),
                    "Operators".to_string()
                ).into()
            ),
        }
    }

    pub fn add_new_operator(
        &mut self, id: usize, node: Option<usize>, params: Vec<usize>, operator: Box<dyn OperatorContents>
    ) -> Result<()> {
        if self.operators.contains_key(&id) {
            return Err(
                KdezeroError::ExistError(
                    id.to_string(),
                    "Operators".to_string()
                ).into()
            );
        }
        let operator = Operator::new(id, node, params, operator);
        self.operators.insert(id, operator);
        Ok(())
    }
}
