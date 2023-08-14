use super::Model;

use anyhow::Result;
use ktensor::Tensor;
use crate::node::NodeData;
use crate::variable::VariableData;
use crate::error::KdezeroError;

impl Model {
    pub fn forward(&mut self) -> Result<()> {
        if self.sorted_forward_nodes.is_empty() {
            self.sorted_forward_nodes = self.graph.topological_sort(false)?;
        }
        for id in self.sorted_forward_nodes.iter() {
            let node = self.graph.get_node(*id)?;
            match node.get_data() {
                NodeData::Operator(operator_id) => {
                    let operator = self.operators.get_operator(*operator_id)?;
                    operator.forward(&self.graph, &mut self.variables)?;
                },
                _ => (),
            }
        }
        Ok(())
    }

    pub fn backward(&mut self, name: &str) -> Result<()> {
        if self.sorted_backward_nodes.is_empty() {
            self.sorted_backward_nodes = self.graph.topological_sort(true)?;
        }
        let grad_model = match self.grad_model {
            Some(ref mut grad_model) => grad_model,
            None => {
                self.grad_model = Some(Box::new(Model::new()));
                self.grad_model.as_mut().unwrap()
            }
        };
        let node = self.graph.get_node_from_name(name)?;
        let variable_id = node.get_data().get_variable_id()?;
        let variable_data =
            self.variables.get_variable(variable_id)?.get_data();
        let grad_data = match variable_data {
            VariableData::F32(tensor) =>
                VariableData::F32(Box::new(Tensor::ones_like(tensor))),
            VariableData::F64(tensor) =>
                VariableData::F64(Box::new(Tensor::ones_like(tensor))),
            VariableData::USIZE(tensor) =>
                VariableData::USIZE(Box::new(Tensor::ones_like(tensor))),
            VariableData::I32(tensor) =>
                VariableData::I32(Box::new(Tensor::ones_like(tensor))),
            VariableData::I64(tensor) =>
                VariableData::I64(Box::new(Tensor::ones_like(tensor))),
            _ => return Err(KdezeroError::NotImplementedTypeError(
                variable_data.to_string(),
                "Model".to_string()
            ).into()),
        };
        let grad = self.variables.get_grad(variable_id)?;
        match grad {
            Some(_) => (),
            None => {
                let grad_node_id = grad_model.graph.get_next_id();
                let grad_variable_id = grad_model.variables.get_next_id();
                grad_model.add_new_node(
                    grad_node_id, "".to_string(),
                    NodeData::Variable(grad_variable_id),
                    Vec::new(), Vec::new()
                )?;
                grad_model.add_new_variable(
                    grad_variable_id, Some(grad_node_id), grad_data
                )?;
                self.set_grad(variable_id, Some(grad_node_id))?;
            },
        }
        let grad_model = match self.grad_model {
            Some(ref mut grad_model) => grad_model,
            None => {
                self.grad_model = Some(Box::new(Model::new()));
                self.grad_model.as_mut().unwrap()
            }
        };
        for id in self.sorted_backward_nodes.iter() {
            let node = self.graph.get_node(*id)?;
            match node.get_data() {
                NodeData::Operator(operator_id) => {
                    let operator = self.operators.get_operator(*operator_id)?;
                    operator.backward(
                        &self.graph, &mut self.variables, grad_model
                    )?;
                },
                _ => (),
            }
        }
        Ok(())
    }
}