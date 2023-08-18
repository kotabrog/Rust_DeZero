use anyhow::Result;
use std::collections::HashMap;
use super::{Variable, Variables};

impl Variables {
    pub fn get_variables(&self) -> &HashMap<usize, Variable> {
        &self.variables
    }

    pub fn get_next_id(&self) -> usize {
        self.next_id
    }

    pub fn get_variable(&self, id: usize) -> Result<&Variable> {
        self.check_id_not_in_variables(id)?;
        Ok(self.variables.get(&id).unwrap())
    }

    pub fn get_grad(&self, id: usize) -> Result<Option<usize>> {
        let variable = self.get_variable(id)?;
        Ok(variable.get_grad())
    }

    pub fn get_grad_id(&self, id: usize) -> Result<usize> {
        let variable = self.get_variable(id)?;
        variable.get_grad_id()
    }

    pub(crate) fn get_node_id(&self, id: usize) -> Result<Option<usize>> {
        let variable = self.get_variable(id)?;
        Ok(variable.get_node())
    }

    pub(crate) fn get_variable_mut(&mut self, id: usize) -> Result<&mut Variable> {
        self.check_id_not_in_variables(id)?;
        Ok(self.variables.get_mut(&id).unwrap())
    }

    pub(crate) fn get_variables_mut(&mut self) -> &mut HashMap<usize, Variable> {
        &mut self.variables
    }

    pub(crate) fn move_all_variable(self) -> HashMap<usize, Variable> {
        self.variables
    }
}
