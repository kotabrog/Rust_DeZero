use anyhow::Result;
use super::OperatorContents;
use crate::variable::{Variables, VariableData};
use crate::node::{Graph, NodeData};
use crate::model::Model;
use crate::error::KdezeroError;

pub struct Square {}

impl OperatorContents for Square {
    fn forward(
            &self, node_id: usize,
            graph: &Graph,
            variables: &mut Variables,
        ) -> Result<Vec<usize>> {
        let operator_node = graph.get_node(node_id)?;
        operator_node.check_inputs_len(1)?;
        operator_node.check_outputs_len(1)?;
        let input_id = operator_node.get_inputs()[0];
        let input_node = graph.get_node(input_id)?;
        let input_variable_id = input_node.get_variable_id()?;
        let input_variable = variables.get_variable(input_variable_id)?;
        let variable_data = input_variable.get_data();
        let output_data = variable_data.pow(2)?;
        let output_id = operator_node.get_outputs()[0];
        let output_node = graph.get_node(output_id)?;
        let output_variable_id = output_node.get_variable_id()?;
        let output_variable = variables.get_mut_variable(output_variable_id)?;
        output_variable.set_data(output_data);
        Ok(vec![output_id])
    }

    fn backward(
            &self, node_id: usize,
            graph: &Graph, variables: &mut Variables,
            grad_model: &mut Model,
        ) -> Result<Vec<usize>> {
        let operator_node = graph.get_node(node_id)?;
        operator_node.check_inputs_len(1)?;
        operator_node.check_outputs_len(1)?;
        let output_id = operator_node.get_outputs()[0];
        let output_node = graph.get_node(output_id)?;
        let output_variable_id = output_node.get_variable_id()?;
        let output_variable = variables.get_variable(output_variable_id)?;
        let output_grad_id = output_variable
            .get_grad()
            .ok_or_else(|| KdezeroError::NotFoundError(
                "Variable.grad".to_string(),
                format!("Variable(id={})", output_variable_id)
            ))?;
        let output_grad_node = grad_model.get_graph().get_node(output_grad_id)?;
        let output_grad_variable_id = output_grad_node.get_variable_id()?;
        let output_grad_variable = grad_model.get_variables().get_variable(output_grad_variable_id)?;
        let output_grad_data = output_grad_variable.get_data();
        let mut grad_data = match output_grad_data {
            VariableData::F32(tensor) =>
                VariableData::F32(Box::new(*tensor.clone() * 2.0)),
            VariableData::F64(tensor) =>
                VariableData::F64(Box::new(*tensor.clone() * 2.0)),
            VariableData::USIZE(tensor) =>
                VariableData::USIZE(Box::new(*tensor.clone() * 2)),
            VariableData::I32(tensor) =>
                VariableData::I32(Box::new(*tensor.clone() * 2)),
            VariableData::I64(tensor) =>
                VariableData::I64(Box::new(*tensor.clone() * 2)),
            _ => return Err(KdezeroError::NotImplementedTypeError(
                output_grad_data.to_string(),
                "Square backward".to_string()
            ).into()),
        };
        let input_id = operator_node.get_inputs()[0];
        let input_node = graph.get_node(input_id)?;
        let input_variable_id = input_node.get_variable_id()?;
        let input_variable = variables.get_variable(input_variable_id)?;
        let input_data = input_variable.get_data();
        if input_data.to_string() != grad_data.to_string() {
            return Err(KdezeroError::NotCollectTypeError(
                input_data.to_string(),
                grad_data.to_string(),
            ).into());
        }
        grad_data = input_data.mul(&grad_data)?;
        let grad_variable_id = grad_model.get_variables().get_next_id();
        let grad_node_id = grad_model.get_graph().get_next_id();
        grad_model.add_new_variable(
            grad_variable_id, Some(grad_node_id), grad_data
        )?;
        grad_model.add_new_node(
            grad_node_id, "".to_string(),
            NodeData::Variable(grad_variable_id),
            vec![], vec![]
        )?;
        variables.set_grad(input_variable_id, Some(grad_node_id))?;
        Ok(vec![input_id])
    }
}
