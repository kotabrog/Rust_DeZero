use anyhow::Result;
use std::collections::HashMap;
use super::{Variable, VariableData};
use crate::error::KdezeroError;

pub struct Variables {
    variables: HashMap<usize, Variable>,
}

impl Variables {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    pub fn get_variables(&self) -> &HashMap<usize, Variable> {
        &self.variables
    }

    pub fn add_variable(&mut self, variable: Variable) -> Option<Variable> {
        self.variables.insert(variable.get_id(), variable)
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
        Ok(())
    }
}
