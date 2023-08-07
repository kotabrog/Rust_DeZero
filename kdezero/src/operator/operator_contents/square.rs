use std::collections::HashMap;
use anyhow::Result;
use super::OperatorContents;
use crate::variable::{Variables, VariableData};
use crate::node::Graph;
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
        let input_variable_id = input_node.get_data().get_variable_id()?;
        let input_variable = variables.get_variable(input_variable_id)?;
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
        let output_id = operator_node.get_outputs()[0];
        let output_node = graph.get_node(output_id)?;
        let output_variable_id = output_node.get_data().get_variable_id()?;
        let output_variable = variables.get_mut_variable(output_variable_id)?;
        output_variable.set_data(output_data);
        Ok(vec![output_id])
    }
}
