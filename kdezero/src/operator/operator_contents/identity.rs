use anyhow::Result;
use super::{OperatorContents, Add};
use crate::variable::VariableData;
use crate::model::{Model, ModelVariable, ModelOperator};

#[derive(Clone)]
pub struct Identity {}

impl OperatorContents for Identity {
    fn forward(
            &mut self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        model.check_inputs_len(node_id, 1)?;
        model.check_outputs_len_at_least(node_id, 1)?;
        let input = model.get_node_inputs_from_node_id(node_id)?[0];
        let outputs = model.get_node_outputs_from_node_id(node_id)?.clone();
        for output in outputs.clone() {
            let variable_data = model.get_variable_data_from_node_id(input)?;
            model.set_variable_data_from_node_id(output, variable_data.clone())?;
        }
        Ok(outputs)
    }

    fn backward(
            &mut self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        model.check_inputs_len(node_id, 1)?;
        model.check_outputs_len_at_least(node_id, 1)?;
        let input = model.get_node_inputs_from_node_id(node_id)?[0];
        let outputs = model.get_node_outputs_from_node_id(node_id)?.clone();
        let output_grad_ids = model.get_grad_ids_from_node_ids(&outputs)?;
        let insert_model = if outputs.len() == 1 {
            Model::make_model(
                vec![ModelVariable::new("in", VariableData::None)],
                vec![ModelVariable::new("out", VariableData::None)],
                vec![ModelOperator::new(
                        "op", Box::new(Identity {}),
                        vec!["in"], vec!["out"], vec![]
                )],
                vec![]
            )?
        } else {
            let len = outputs.len();
            let input_names: Vec<String> = (0..len).into_iter()
                .map(|i| format!("in{}", i)).collect();
            let model_inputs = input_names.iter()
                .map(|name| ModelVariable::new(
                        name,
                        VariableData::None
                )).collect();
            Model::make_model(
                model_inputs,
                vec![ModelVariable::new("out", VariableData::None)],
                vec![ModelOperator::new(
                        "op", Box::new(Add {}),
                        input_names.iter().map(|s| s.as_str()).collect(),
                        vec!["out"], vec![]
                )], vec![]
            )?
        };
        let grad_outputs = model.get_grad_model_mut()
            .insert_structure_model(insert_model, &output_grad_ids)?;
        model.set_or_add_grad(input, grad_outputs[0])?;
        Ok(vec![input])
    }
}
