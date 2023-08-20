use anyhow::Result;
use super::OperatorContents;
use crate::model::{Model, ModelVariable, ModelOperator};
use crate::variable::VariableData;

#[derive(Clone)]
pub struct Mul {}

impl OperatorContents for Mul {
    fn forward(
            &self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        let (inputs, outputs) =
            model.check_inputs_outputs_len(node_id, 2, 1)?;
        let output = outputs[0];
        let variable_data0 = model.get_variable_data_from_node_id(inputs[0])?;
        let variable_data1 = model.get_variable_data_from_node_id(inputs[1])?;
        let output_data = variable_data0.mul(variable_data1)?;
        model.set_variable_data_from_node_id(output, output_data)?;
        Ok(vec![output])
    }

    fn backward(
            &self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        let (inputs, outputs) =
            model.check_inputs_outputs_len(node_id, 2, 1)?;
        let inputs = inputs.clone();
        let variable_data0 = model.get_variable_data_from_node_id(inputs[0])?;
        let variable_data1 = model.get_variable_data_from_node_id(inputs[1])?;
        let output_grad_id = model.get_grad_id_from_node_id(outputs[0])?;
        let insert_model = Model::make_model(
            vec![ModelVariable::new("in", VariableData::None)],
            vec![
                ModelVariable::new("out0", VariableData::None),
                ModelVariable::new("out1", VariableData::None),
            ],
            vec![
                ModelOperator::new(
                    "op0", Box::new(Mul {}),
                    vec!["in", "x1"], vec!["out0"], vec![]
                ),
                ModelOperator::new(
                    "op1", Box::new(Mul {}),
                    vec!["in", "x0"], vec!["out1"], vec![]
                ),
            ],
            vec![
                ModelVariable::new("x0", variable_data0.clone()),
                ModelVariable::new("x1", variable_data1.clone()),
            ]
        )?;
        let grad_outputs = model.get_grad_model_mut()
            .insert_structure_model(insert_model, &vec![output_grad_id])?;
        for (grad, node_id) in grad_outputs.into_iter().zip(inputs.clone()) {
            model.set_or_add_grad(node_id, grad)?;
        }
        Ok(inputs)
    }
}
