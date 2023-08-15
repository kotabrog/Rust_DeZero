use anyhow::Result;
use super::Model;
use crate::variable::{Variable, Variables, VariableData};
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

    // pub(crate) fn get_variables_mut(&mut self) -> &mut Variables {
    //     &mut self.variables
    // }

    pub(crate) fn get_grad_model_mut(&mut self) -> &mut Box<Model> {
        self.init_grad_model();
        self.grad_model.as_mut().unwrap()
    }

    // pub(crate) fn get_grad_model_unwrap(&self) -> &Box<Model> {
    //     self.grad_model.as_ref().unwrap()
    // }

    pub(crate) fn get_grad_model_result(&self) -> Result<&Box<Model>> {
        self.grad_model.as_ref()
            .ok_or_else(
                || KdezeroError::NotFoundError(
                    "grad_model".to_string(),
                    "Model".to_string()
                ).into()
            )
    }

    // pub(crate) fn get_and_init_grad_model_mut(&mut self) -> &Box<Model> {
    //     self.init_grad_model();
    //     self.grad_model.as_ref().unwrap()
    // }

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

    pub(crate) fn get_variable_data_from_node_id(&self, node_id: usize) -> Result<&VariableData> {
        let variable_id = self.get_variable_id_from_node_id(node_id)?;
        Ok(self.variables.get_variable(variable_id)?
            .get_data())
    }

    pub(crate) fn get_variable_id_from_node_id(&self, node_id: usize) -> Result<usize> {
        let variable_id = self.graph.get_node(node_id)?
            .get_variable_id()?;
        Ok(variable_id)
    }

    pub(crate) fn get_variable_from_node_id(&self, node_id: usize) -> Result<&Variable> {
        let variable_id = self.get_variable_id_from_node_id(node_id)?;
        Ok(self.variables.get_variable(variable_id)?)
    }

    pub(crate) fn get_variable_from_node_id_mut(&mut self, node_id: usize) -> Result<&mut Variable> {
        let variable_id = self.get_variable_id_from_node_id(node_id)?;
        Ok(self.variables.get_mut_variable(variable_id)?)
    }

    pub(crate) fn get_grad_id_from_node_id(&self, node_id: usize) -> Result<usize> {
        let variable_id = self.get_variable_id_from_node_id(node_id)?;
        let grad_id = self.get_variable_from_node_id(node_id)?.get_grad()
            .ok_or_else(|| KdezeroError::NotFoundError(
                "Variable.grad".to_string(),
                format!("Variable(id={})", variable_id)
            ))?;
        Ok(grad_id)
    }

    pub(crate) fn get_grad_data_from_node_id(&self, node_id: usize) -> Result<&VariableData> {
        let grad_id = self.get_grad_id_from_node_id(node_id)?;
        Ok(self.get_grad_model_result()?
            .get_variable_data_from_node_id(grad_id)?)
    }
}
