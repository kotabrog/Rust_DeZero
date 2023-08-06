use std::collections::HashMap;
use anyhow::Result;
use super::super::OperatorContents;
use crate::variable::{Variable, VariableData};
use crate::node::{Node, NodeData};
use crate::error::KdezeroError;

pub struct Square {}

impl OperatorContents for Square {
    fn forward(
            &self, node_id: usize,
            nodes: &HashMap<usize, Node>,
            variables: &mut HashMap<usize, Variable>,
        ) -> Result<Vec<usize>> {
        let operator_node = nodes
            .get(&node_id)
            .ok_or_else(
                || KdezeroError::NotFoundError(
                    node_id.to_string(),
                    "nodes".to_string()
                ))?;
        let inputs = operator_node.get_inputs();
        if inputs.len() != 1 {
            return Err(KdezeroError::SizeError(
                "inputs".to_string(),
                1,
                inputs.len()
            ).into());
        }
        let input_id = inputs[0];
        let input_node = nodes
            .get(&input_id)
            .ok_or_else(
                || KdezeroError::NotFoundError(
                    input_id.to_string(),
                    "nodes".to_string()
                ))?;
        let input_variable_id = match input_node.get_data() {
            NodeData::Variable(id) => *id,
            _ => return Err(KdezeroError::NotCollectTypeError(
                input_node.get_data().to_string(),
                "Variable".to_string()
            ).into()),
        };
        let input_variable = variables
            .get(&input_variable_id)
            .ok_or_else(
                || KdezeroError::NotFoundError(
                    input_id.to_string(),
                    "variables".to_string()
                ))?;
        let variable_data = input_variable.get_data();
        let output_data = match variable_data {
            VariableData::F32(tensor) =>
                VariableData::F32(Box::new(tensor.clone().powi(2))),
            VariableData::F64(tensor) =>
                VariableData::F64(Box::new(tensor.clone().powi(2))),
            VariableData::USIZE(tensor) =>
                VariableData::USIZE(Box::new(tensor.clone().pow(2))),
            VariableData::I32(tensor) =>
                VariableData::I32(Box::new(tensor.clone().pow(2))),
            VariableData::I64(tensor) =>
                VariableData::I64(Box::new(tensor.clone().pow(2))),
            _ => return Err(KdezeroError::NotImplementedTypeError(
                variable_data.to_string(),
                "Square".to_string()
            ).into()),
        };
        let outputs = operator_node.get_outputs();
        if outputs.len() != 1 {
            return Err(KdezeroError::SizeError(
                "outputs".to_string(),
                1,
                outputs.len()
            ).into());
        }
        let output_id: usize = outputs[0];
        let output_node = nodes
            .get(&output_id)
            .ok_or_else(
                || KdezeroError::NotFoundError(
                    output_id.to_string(),
                    "nodes".to_string()
                ))?;
        let output_variable_id = match output_node.get_data() {
            NodeData::Variable(id) => *id,
            _ => return Err(KdezeroError::NotCollectTypeError(
                output_node.get_data().to_string(),
                "Variable".to_string()
            ).into()),
        };
        let output_variable = variables
            .get_mut(&output_variable_id)
            .ok_or_else(
                || KdezeroError::NotFoundError(
                    output_id.to_string(),
                    "variables".to_string()
                ))?;
        output_variable.set_data(output_data);
        Ok(vec![output_id])
    }
}
