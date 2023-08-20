use std::vec;

use anyhow::Result;
use super::{OperatorContents, Mul, Sin};
use crate::model::{Model, ModelVariable, ModelOperator};
use crate::variable::VariableData;

#[derive(Clone)]
pub struct Cos {}

impl OperatorContents for Cos {
    fn forward(
            &self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        let (inputs, outputs) =
            model.check_inputs_outputs_len(node_id, 1, 1)?;
        let input_id = inputs[0];
        let output_id = outputs[0];
        let variable_data = model.get_variable_data_from_node_id(input_id)?;
        let output_data = variable_data.cos()?;
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
        if !model.get_grad_model_result()?.is_in_node_id(input_id) {
            model.clone_node_to_grad_model(input_id)?;
        }
        let input_data = model.get_variable_data_from_node_id(input_id)?;
        let output_grad_id = model.get_grad_id_from_node_id(output_id)?;
        let insert_model = Model::make_model(
            vec![
                ModelVariable::new("in", VariableData::None),
                ModelVariable::new("x", VariableData::None),
            ], vec![ModelVariable::new("out", VariableData::None)],
            vec![
                ModelOperator::new(
                    "op", Box::new(Sin {}),
                    vec!["x"], vec!["sin"], vec![]
                ), ModelOperator::new(
                    "op", Box::new(Mul {}),
                    vec!["init", "sin"], vec!["minus"], vec![]
                ), ModelOperator::new(
                    "op", Box::new(Mul {}),
                    vec!["in", "minus"], vec!["out"], vec![]
                )],
            vec![
                ModelVariable::new("init", VariableData::scalar(&input_data, -1.0)?),
            ]
        )?;
        let grad_outputs = model.get_grad_model_mut()
            .insert_structure_model(insert_model, &vec![output_grad_id, input_id])?;
        model.set_or_add_grad(input_id, grad_outputs[0])?;
        Ok(vec![input_id])
    }
}
