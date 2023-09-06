use anyhow::Result;
use ktensor::Tensor;
use super::{OperatorContents, Mul, ScalarMul};
use crate::variable::VariableData;
use crate::model::{Model, ModelVariable, ModelOperator};
use crate::error::KdezeroError;

#[derive(Clone)]
pub struct Pow {}

impl OperatorContents for Pow {
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
            .to_usize_tensor()?
            .to_scalar()?;
        let input_data = model.get_variable_data_from_node_id(input)?;
        let output_data = input_data.pow(c as u32)?;
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
            .to_usize_tensor()?;
        let c0 = c.to_scalar()?.checked_sub(1)
            .ok_or_else(|| KdezeroError::OverflowError(
                "Pow c".to_string()
            ))?;
        let input_data = model.get_variable_data_from_node_id(input)?;
        let c1 = VariableData::as_type_from_other(c.clone(), input_data)?;
        // let c1 = match input_data {
        //     VariableData::F32(_) => Tensor::scalar(c as f32).into(),
        //     VariableData::F64(_) => Tensor::scalar(c as f64).into(),
        //     VariableData::I32(_) => Tensor::scalar(c as i32).into(),
        //     VariableData::I64(_) => Tensor::scalar(c as i64).into(),
        //     VariableData::USIZE(_) => Tensor::scalar(c as usize).into(),
        //     _ => return Err(KdezeroError::NotImplementedTypeError(
        //         input_data.to_string(), "Pow".to_string()
        //     ).into()),
        // };
        let output_grad_id = model.get_grad_id_from_node_id(output)?;
        model.clone_node_to_grad_model_if_needed(input)?;
        let insert_model = Model::make_model(
            vec![
                ModelVariable::new("in", VariableData::None),
                ModelVariable::new("x", VariableData::None),
            ],
            vec![ModelVariable::new("out", VariableData::None)],
            vec![
                ModelOperator::new(
                    "op0", Box::new(Pow {}),
                    vec!["x"], vec!["pow"],
                    vec![Tensor::scalar(c0).into()]
                ), ModelOperator::new(
                    "op1", Box::new(ScalarMul {}),
                    vec!["pow"], vec!["scalar_mul"],
                    vec![c1]
                ), ModelOperator::new(
                    "op2", Box::new(Mul {}),
                    vec!["in", "scalar_mul"], vec!["out"], vec![]
                ),
            ], vec![],
        )?;
        let grad_outputs = model.get_grad_model_mut()
            .insert_structure_model(insert_model, &vec![output_grad_id, input])?;
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
        use num_traits::*;

        let tensor = Tensor::<f64>::new(vec![2.0], vec![])
            .unwrap();

        let mut model = Model::make_model(
            vec![ModelVariable::new(
                    "in", tensor.into()
            )],
            vec![ModelVariable::new(
                    "out", VariableData::None
            )],
            vec![ModelOperator::new(
                    "op", Box::new(super::Pow {}),
                    vec!["in"], vec!["out"],
                    vec![Tensor::scalar(3usize).into()]
            )],
            vec![]
        ).unwrap();
        model.forward().unwrap();
        let output_variable = model.get_variable_from_name("out").unwrap();
        assert_eq!(output_variable.get_type(), "F64");
        assert_eq!(output_variable.get_data(), &Tensor::new(vec![2.0.powi(3)], vec![]).unwrap().into());
    }

    #[test]
    fn backward_normal() {
        use ktensor::Tensor;

        let tensor = Tensor::<f64>::new(vec![2.0], vec![])
            .unwrap();

        let mut model = Model::make_model(
            vec![ModelVariable::new(
                    "in", tensor.into()
            )],
            vec![ModelVariable::new(
                    "out", VariableData::None
            )],
            vec![ModelOperator::new(
                    "op", Box::new(Pow {}),
                    vec!["in"], vec!["out"],
                    vec![Tensor::scalar(3usize).into()]
            )],
            vec![]
        ).unwrap();
        model.forward().unwrap();
        let output_id = model.get_node_id_from_name("out").unwrap();
        model.backward(output_id).unwrap();
        let input_grad_variable = model.get_grad_from_variable_name("in").unwrap();
        assert_eq!(input_grad_variable.get_type(), "F64");
        assert_eq!(input_grad_variable.get_data(), &Tensor::new(vec![12.0], vec![]).unwrap().into());
    }
}
