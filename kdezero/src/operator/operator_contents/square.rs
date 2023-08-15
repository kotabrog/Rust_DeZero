use anyhow::Result;
use super::OperatorContents;
use crate::node::NodeData;
use crate::model::Model;
use crate::error::KdezeroError;

#[derive(Clone)]
pub struct Square {}

impl OperatorContents for Square {
    fn forward(
            &self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        let (inputs, outputs) =
            model.check_inputs_outputs_len(node_id, 1, 1)?;
        let input_id = inputs[0];
        let output_id = outputs[0];
        let variable_data = model.get_variable_data_from_node_id(input_id)?;
        let output_data = variable_data.pow(2)?;
        model.set_variable_data_from_node_id(output_id, output_data)?;
        Ok(vec![output_id])
    }

    fn backward(
            &self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        let (inputs, outputs) =
            model.check_inputs_outputs_len(node_id, 1, 1)?;
        let input_id = inputs[0];
        let output_id = outputs[0];
        let output_node = model.get_graph().get_node(output_id)?;
        let output_variable_id = output_node.get_variable_id()?;
        let output_variable = model.get_variables().get_variable(output_variable_id)?;
        let output_grad_id = output_variable
            .get_grad()
            .ok_or_else(|| KdezeroError::NotFoundError(
                "Variable.grad".to_string(),
                format!("Variable(id={})", output_variable_id)
            ))?;
        let grad_model = model.get_grad_model_unwrap();
        let output_grad_node = grad_model.get_graph().get_node(output_grad_id)?;
        let output_grad_variable_id = output_grad_node.get_variable_id()?;
        let output_grad_variable = grad_model.get_variables().get_variable(output_grad_variable_id)?;
        let output_grad_data = output_grad_variable.get_data();
        let mut grad_data = output_grad_data.scalar_mul(2.0)?;
        let input_node = model.get_graph().get_node(input_id)?;
        let input_variable_id = input_node.get_variable_id()?;
        let input_data = model.get_variable_data_from_node_id(input_id)?;
        input_data.check_type(&grad_data)?;
        grad_data = input_data.mul(&grad_data)?;
        let grad_variable_id = grad_model.get_variables().get_next_id();
        let grad_node_id = grad_model.get_graph().get_next_id();
        let grad_model = model.get_grad_model_mut();
        grad_model.add_new_variable(
            grad_variable_id, Some(grad_node_id), grad_data
        )?;
        grad_model.add_new_node(
            grad_node_id, "".to_string(),
            NodeData::Variable(grad_variable_id),
            vec![], vec![]
        )?;
        model.set_grad(input_variable_id, Some(grad_node_id))?;
        Ok(vec![input_id])
    }
}
