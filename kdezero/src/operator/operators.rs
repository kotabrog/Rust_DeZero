use anyhow::Result;
use std::collections::HashMap;
use super::{Operator, OperatorContents};
use crate::error::KdezeroError;

pub struct Operators {
    operators: HashMap<usize, Operator>,
    next_id: usize,
}

impl Operators {
    pub fn new() -> Self {
        Self {
            operators: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn get_operators(&self) -> &HashMap<usize, Operator> {
        &self.operators
    }

    pub fn get_next_id(&self) -> usize {
        self.next_id
    }

    pub fn add_operator(&mut self, operator: Operator) -> Option<Operator> {
        let next_id = operator.get_id().max(self.next_id) + 1;
        let result = self.operators.insert(operator.get_id(), operator);
        if result.is_some() {
            self.next_id = next_id;
        }
        result
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
        self.next_id = self.next_id.max(id) + 1;
        Ok(())
    }
}
