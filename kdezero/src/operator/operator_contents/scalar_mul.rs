use anyhow::Result;
use super::OperatorContents;
use crate::variable::VariableData;
use crate::model::{Model, ModelVariable, ModelOperator};

#[derive(Clone)]
pub struct ScalarMul {}

impl OperatorContents for ScalarMul {
    fn forward(
            &self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        let (inputs, outputs) =
            model.check_inputs_outputs_len(node_id, 1, 1)?;
        let params = model.check_params_len(node_id, 1)?;
        let input = inputs[0];
        let output = outputs[0];
        let c = model.get_variable_data_from_variable_id(params[0])?
            .to_f64_tensor()?
            .to_scalar()?;
        let input_data = model.get_variable_data_from_node_id(input)?;
        let output_data = input_data.scalar_mul(c)?;
        model.set_variable_data_from_node_id(output, output_data)?;
        Ok(vec![output])
    }

    fn backward(
            &self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        let (inputs, outputs) =
            model.check_inputs_outputs_len(node_id, 1, 1)?;
        let params = model.check_params_len(node_id, 1)?;
        let input = inputs[0];
        let output = outputs[0];
        let c = model.get_variable_data_from_variable_id(params[0])?
            .to_f64_tensor()?.clone();
        let output_grad_id = model.get_grad_id_from_node_id(output)?;
        let insert_model = Model::make_model(
            vec![ModelVariable::new("in", VariableData::None)],
            vec![ModelVariable::new("out", VariableData::None)],
            vec![ModelOperator::new(
                "op", Box::new(ScalarMul {}),
                vec!["in"], vec!["out"],
                vec![c.into()]
            )],
            vec![]
        )?;
        let grad_outputs = model.get_grad_model_mut()
            .insert_structure_model(insert_model, &vec![output_grad_id])?;
        model.set_or_add_grad(input, grad_outputs[0])?;
        Ok(vec![input])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_normal() {
        use ktensor::Tensor;

        let tensor = Tensor::new(vec![2.0], vec![])
            .unwrap();

        let mut model = Model::make_model(
            vec![ModelVariable::new(
                    "in", tensor.into()
            )],
            vec![ModelVariable::new(
                    "out", VariableData::None
            )],
            vec![ModelOperator::new(
                    "op", Box::new(ScalarMul {}),
                    vec!["in"], vec!["out"],
                    vec![Tensor::scalar(3.0).into()]
            )],
            vec![]
        ).unwrap();
        model.forward().unwrap();
        let output_variable = model.get_variable_from_name("out").unwrap();
        assert_eq!(output_variable.get_type(), "F64");
        assert_eq!(output_variable.get_data(), &Tensor::new(vec![6.0], vec![]).unwrap().into());
    }

    #[test]
    fn backward_normal() {
        use ktensor::Tensor;

        let tensor = Tensor::new(vec![3.0], vec![])
            .unwrap();

        let mut model = Model::make_model(
            vec![ModelVariable::new(
                    "in", tensor.into()
            )],
            vec![ModelVariable::new(
                    "out", VariableData::None
            )],
            vec![ModelOperator::new(
                    "op", Box::new(ScalarMul {}),
                    vec!["in"], vec!["out"],
                    vec![Tensor::scalar(3.0).into()]
            )],
            vec![]
        ).unwrap();
        model.forward().unwrap();
        let output_id = model.get_node_id_from_name("out").unwrap();
        model.backward(output_id).unwrap();
        let input_grad_variable = model.get_grad_from_variable_name("in").unwrap();
        assert_eq!(input_grad_variable.get_type(), "F64");
        assert_eq!(input_grad_variable.get_data(), &Tensor::new(vec![3.0], vec![]).unwrap().into());
    }
}