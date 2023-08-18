use anyhow::Result;
use std::collections::HashMap;
use super::{Variable, VariableData};
use crate::error::KdezeroError;

#[derive(Debug, Clone)]
pub struct Variables {
    variables: HashMap<usize, Variable>,
    next_id: usize,
}

impl Variables {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn get_variables(&self) -> &HashMap<usize, Variable> {
        &self.variables
    }

    pub fn get_next_id(&self) -> usize {
        self.next_id
    }

    fn check_id_in_variables(&self, id: usize) -> Result<()> {
        if self.variables.contains_key(&id) {
            return Err(
                KdezeroError::ExistError(
                    id.to_string(),
                    "Variables".to_string()
                ).into()
            );
        }
        Ok(())
    }

    fn check_id_not_in_variables(&self, id: usize) -> Result<()> {
        if !self.variables.contains_key(&id) {
            return Err(
                KdezeroError::NotFoundError(
                    id.to_string(),
                    "Variables".to_string()
                ).into()
            );
        }
        Ok(())
    }

    pub fn add_variable(&mut self, variable: Variable) -> Option<Variable> {
        let id = variable.get_id();
        let result = self.variables.insert(variable.get_id(), variable);
        if result.is_some() {
            self.update_next_id(id);
        }
        result
    }

    pub fn get_variable(&self, id: usize) -> Result<&Variable> {
        self.check_id_not_in_variables(id)?;
        Ok(self.variables.get(&id).unwrap())
    }

    pub(crate) fn get_variable_mut(&mut self, id: usize) -> Result<&mut Variable> {
        self.check_id_not_in_variables(id)?;
        Ok(self.variables.get_mut(&id).unwrap())
    }

    pub(crate) fn get_variables_mut(&mut self) -> &mut HashMap<usize, Variable> {
        &mut self.variables
    }

    pub fn get_grad(&self, id: usize) -> Result<Option<usize>> {
        let variable = self.get_variable(id)?;
        Ok(variable.get_grad())
    }

    pub(crate) fn get_node_id(&self, id: usize) -> Result<Option<usize>> {
        let variable = self.get_variable(id)?;
        Ok(variable.get_node())
    }

    pub fn add_new_variable(
        &mut self, id: usize, node: Option<usize>, data: VariableData)
    -> Result<()> {
        if self.variables.contains_key(&id) {
            return Err(
                KdezeroError::ExistError(
                    id.to_string(),
                    "Variables".to_string()
                ).into()
            );
        }
        let variable = Variable::new(id, node, data);
        self.variables.insert(id, variable);
        self.update_next_id(id);
        Ok(())
    }

    pub(crate) fn set_node_id(&mut self, id: usize, node: Option<usize>) -> Result<()> {
        let variable = self.get_variable_mut(id)?;
        variable.set_node(node);
        Ok(())
    }

    pub fn set_grad(&mut self, id: usize, grad: Option<usize>) -> Result<()> {
        let variable = self.get_variable_mut(id)?;
        variable.set_grad(grad);
        Ok(())
    }

    pub(crate) fn change_variable_id(&mut self, id: usize, new_id: usize) -> Result<()> {
        self.check_id_not_in_variables(id)?;
        self.check_id_in_variables(new_id)?;
        let mut variable = self.variables.remove(&id).unwrap();
        variable.set_id(new_id);
        self.variables.insert(new_id, variable);
        Ok(())
    }

    pub(crate) fn move_all_variable(self) -> HashMap<usize, Variable> {
        self.variables
    }

    fn update_next_id(&mut self, id: usize) {
        self.next_id = self.next_id.max(id) + 1;
    }

    pub(crate) fn delete_variable(&mut self, id: usize) -> Result<()> {
        self.check_id_not_in_variables(id)?;
        self.variables.remove(&id);
        Ok(())
    }

    pub fn print_variables(&self) {
        println!("Variables:");
        for i in self.variables.keys() {
            println!("  {:?}: {:?}", i, self.variables[i]);
        }
    }
}
