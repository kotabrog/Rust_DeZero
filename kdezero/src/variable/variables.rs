use anyhow::Result;
use std::collections::HashMap;
use super::{Variable, VariableData};
use crate::error::KdezeroError;

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

    pub fn add_variable(&mut self, variable: Variable) -> Option<Variable> {
        let next_id = variable.get_id().max(self.next_id) + 1;
        let result = self.variables.insert(variable.get_id(), variable);
        if result.is_some() {
            self.next_id = next_id;
        }
        result
    }

    pub fn get_variable(&self, id: usize) -> Result<&Variable> {
        match self.variables.get(&id) {
            Some(variable) => Ok(variable),
            None => Err(
                KdezeroError::NotFoundError(
                    id.to_string(),
                    "Variables".to_string()
                ).into()
            ),
        }
    }

    pub fn get_mut_variable(&mut self, id: usize) -> Result<&mut Variable> {
        match self.variables.get_mut(&id) {
            Some(variable) => Ok(variable),
            None => Err(
                KdezeroError::NotFoundError(
                    id.to_string(),
                    "Variables".to_string()
                ).into()
            ),
        }
    }

    pub fn get_grad(&self, id: usize) -> Result<Option<usize>> {
        let variable = self.get_variable(id)?;
        Ok(variable.get_grad())
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
        self.next_id = self.next_id.max(id) + 1;
        Ok(())
    }

    pub fn set_grad(&mut self, id: usize, grad: Option<usize>) -> Result<()> {
        let variable = self.get_mut_variable(id)?;
        variable.set_grad(grad);
        Ok(())
    }
}
