use std::vec;

use anyhow::Result;
use ktensor::Tensor;
use super::OperatorContents;
use crate::model::{Model, ModelVariable, ModelOperator};
use crate::variable::VariableData;
use crate::error::KdezeroError;

#[derive(Clone)]
pub struct Reshape {}

impl OperatorContents for Reshape {
    fn forward(
            &self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        let (inputs, outputs) =
            model.check_inputs_outputs_len(node_id, 1, 1)?;
        let params = model.check_params_len(node_id, 1)?;
        let input_id = inputs[0];
        let output_id = outputs[0];
        let shape = model.get_variable_data_from_variable_id(params[0])?
            .to_usize_tensor()?;
        if shape.ndim() != 1 {
            return Err(KdezeroError::ParameterError(
                format!("{:?}", shape),
                "Reshape".to_string(),
                "shape.ndim() == 1".to_string(),
            ).into())
        }
        let shape = shape.get_data();
        let variable_data = model.get_variable_data_from_node_id(input_id)?;
        let output_data = variable_data.reshape(shape)?;
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
        let input_data = model.get_variable_data_from_node_id(input_id)?;
        let shape = input_data.get_shape()?;
        let shape = Tensor::new(shape.clone(), vec![shape.len()])?;
        let output_grad_id = model.get_grad_id_from_node_id(output_id)?;
        model.clone_node_to_grad_model_if_needed(input_id)?;
        let insert_model = Model::make_model(
            vec![ModelVariable::new("in", VariableData::None)],
            vec![ModelVariable::new("out", VariableData::None)],
            vec![
                ModelOperator::new(
                    "op", Box::new(Reshape {}),
                    vec!["in"], vec!["out"],
                    vec![shape.into()]
                ),
            ],
            vec![]
        )?;
        let grad_outputs = model.get_grad_model_mut()
            .insert_structure_model(insert_model, &vec![output_grad_id])?;
        model.set_or_add_grad(input_id, grad_outputs[0])?;
        Ok(vec![input_id])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_normal() {
        use ktensor::Tensor;

        let tensor = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]
        ).unwrap();
        let shape = Tensor::<usize>::new(vec![3, 2], vec![2]).unwrap();
        let mut model = Model::make_model(
            vec![ModelVariable::new(
                    "in", tensor.into()
            )],
            vec![ModelVariable::new(
                    "out", VariableData::None
            )],
            vec![ModelOperator::new(
                    "op", Box::new(Reshape {}),
                    vec!["in"], vec!["out"], vec![shape.into()]
            )], vec![]
        ).unwrap();
        model.forward().unwrap();
        let output_variable = model.get_variable_from_name("out").unwrap();
        assert_eq!(output_variable.get_type(), "F64");
        assert_eq!(output_variable.get_data(),
            &Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap().into());
    }

    #[test]
    fn backward_normal() {
        use ktensor::Tensor;

        let tensor = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]
        ).unwrap();
        let shape = Tensor::<usize>::new(vec![3, 2], vec![2]).unwrap();
        let mut model = Model::make_model(
            vec![ModelVariable::new(
                    "in", tensor.into()
            )],
            vec![ModelVariable::new(
                    "out", VariableData::None
            )],
            vec![ModelOperator::new(
                    "op", Box::new(Reshape {}),
                    vec!["in"], vec!["out"], vec![shape.into()]
            )], vec![]
        ).unwrap();
        model.forward().unwrap();
        let output_id = model.get_node_id_from_name("out").unwrap();
        model.backward(output_id).unwrap();
        let input_grad = model.get_grad_from_variable_name("in").unwrap();
        assert_eq!(input_grad.get_type(), "F64");
        assert_eq!(input_grad.get_data(),
            &Tensor::new(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![2, 3]).unwrap().into());
    }
}
