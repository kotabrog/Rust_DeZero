use anyhow::Result;
use super::Model;
use crate::variable::{Variable, Variables};
use crate::node::Graph;
use crate::operator::Operators;
use crate::error::KdezeroError;

impl Model {
    pub fn get_graph(&self) -> &Graph {
        &self.graph
    }

    pub fn get_variables(&self) -> &Variables {
        &self.variables
    }

    pub fn get_operators(&self) -> &Operators {
        &self.operators
    }

    pub fn get_inputs(&self) -> &Vec<usize> {
        &self.inputs
    }

    pub fn get_outputs(&self) -> &Vec<usize> {
        &self.outputs
    }

    pub fn get_grad_model(&self) -> &Option<Box<Model>> {
        &self.grad_model
    }

    pub fn get_variable_from_name(&self, name: &str) -> Result<&Variable> {
        let node = self.graph.get_node_from_name(name)?;
        let variable_id = node.get_data().get_variable_id()?;
        self.variables.get_variable(variable_id)
    }

    pub fn get_grad_from_variable_name(&self, name: &str) -> Result<&Variable> {
        let node = self.graph.get_node_from_name(name)?;
        let variable_id = node.get_data().get_variable_id()?;
        let grad_id = self.variables.get_grad(variable_id)?
            .ok_or_else(|| KdezeroError::NotFoundError(
                "Variable.grad".to_string(),
                "Variable".to_string()
            ))?;
        self.grad_model.as_ref()
            .ok_or_else(
                || KdezeroError::NotFoundError(
                    "grad_model".to_string(),
                    "Model".to_string()
                )
            )?.variables.get_variable(grad_id)
    }
}
