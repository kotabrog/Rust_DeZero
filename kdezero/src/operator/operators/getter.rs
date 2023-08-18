use std::collections::HashMap;
use anyhow::Result;
use super::{Operators, Operator};

impl Operators {
    pub fn get_operators(&self) -> &HashMap<usize, Operator> {
        &self.operators
    }

    pub fn get_next_id(&self) -> usize {
        self.next_id
    }

    pub fn get_operator(&self, id: usize) -> Result<&Operator> {
        self.check_id_not_in_operators(id)?;
        Ok(self.operators.get(&id).unwrap())
    }

    pub(crate) fn get_node_id(&self, id: usize) -> Result<usize> {
        let operator = self.get_operator(id)?;
        operator.get_node_id()
    }

    pub(crate) fn get_operator_mut(&mut self, id: usize) -> Result<&mut Operator> {
        self.check_id_not_in_operators(id)?;
        Ok(self.operators.get_mut(&id).unwrap())
    }

    pub(crate) fn get_operators_mut(&mut self) -> &mut HashMap<usize, Operator> {
        &mut self.operators
    }

    pub(crate) fn move_all_operator(self) -> HashMap<usize, Operator> {
        self.operators
    }
}
