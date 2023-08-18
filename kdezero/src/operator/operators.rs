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

    fn check_id_in_operators(&self, id: usize) -> Result<()> {
        if self.operators.contains_key(&id) {
            return Err(
                KdezeroError::ExistError(
                    id.to_string(),
                    "Operators".to_string()
                ).into()
            );
        }
        Ok(())
    }

    fn check_id_not_in_operators(&self, id: usize) -> Result<()> {
        if !self.operators.contains_key(&id) {
            return Err(
                KdezeroError::NotFoundError(
                    id.to_string(),
                    "Operators".to_string()
                ).into()
            );
        }
        Ok(())
    }

    pub fn add_operator(&mut self, operator: Operator) -> Option<Operator> {
        let id = operator.get_id();
        let result = self.operators.insert(operator.get_id(), operator);
        if result.is_some() {
            self.update_next_id(id);
        }
        result
    }

    pub fn get_operator(&self, id: usize) -> Result<&Operator> {
        self.check_id_not_in_operators(id)?;
        Ok(self.operators.get(&id).unwrap())
    }

    pub fn get_operator_mut(&mut self, id: usize) -> Result<&mut Operator> {
        self.check_id_not_in_operators(id)?;
        Ok(self.operators.get_mut(&id).unwrap())
    }

    pub(crate) fn get_operators_mut(&mut self) -> &mut HashMap<usize, Operator> {
        &mut self.operators
    }

    pub(crate) fn get_node_id(&self, id: usize) -> Result<usize> {
        let operator = self.get_operator(id)?;
        operator.get_node_id()
    }

    pub fn add_new_operator(
        &mut self, id: usize, node: Option<usize>, params: Vec<usize>, operator: Box<dyn OperatorContents>
    ) -> Result<()> {
        self.check_id_in_operators(id)?;
        let operator = Operator::new(id, node, params, operator);
        self.operators.insert(id, operator);
        self.update_next_id(id);
        Ok(())
    }

    pub(crate) fn set_node(&mut self, id: usize, node: Option<usize>) -> Result<()> {
        let operator = self.get_operator_mut(id)?;
        operator.set_node(node);
        Ok(())
    }

    pub(crate) fn change_operator_id(&mut self, old_id: usize, new_id: usize) -> Result<()> {
        self.check_id_not_in_operators(old_id)?;
        self.check_id_in_operators(new_id)?;
        let operator = self.operators.remove(&old_id).unwrap();
        self.operators.insert(new_id, operator);
        self.update_next_id(new_id);
        Ok(())
    }

    pub(crate) fn move_all_operator(self) -> HashMap<usize, Operator> {
        self.operators
    }

    fn update_next_id(&mut self, id: usize) {
        self.next_id = self.next_id.max(id) + 1;
    }
}
