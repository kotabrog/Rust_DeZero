use anyhow::Result;
use ktensor::Tensor;
use super::{OperatorContents, Sub, BroadcastTo, Mul, ScalarMul, Neg};
use crate::model::{Model, ModelVariable, ModelOperator};
use crate::variable::VariableData;

#[derive(Clone)]
pub struct MeanSquaredError {}

impl OperatorContents for MeanSquaredError {
    fn forward(
            &mut self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        let (inputs, outputs) =
            model.check_inputs_outputs_len(node_id, 2, 1)?;
        let output = outputs[0];
        let variable_data0 = model.get_variable_data_from_node_id(inputs[0])?;
        let variable_data1 = model.get_variable_data_from_node_id(inputs[1])?;
        let diff = variable_data0.sub(variable_data1)?;
        let output_data = diff
            .pow(2)?
            .sum::<&[usize]>(None, false)?
            .scalar_mul(1.0 / diff.size()? as f64)?;
        model.set_variable_data_from_node_id(output, output_data)?;
        Ok(vec![output])
    }

    fn backward(
            &mut self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        let (inputs, outputs) =
            model.check_inputs_outputs_len(node_id, 2, 1)?;
        let inputs = inputs.clone();
        let output_grad_id = model.get_grad_id_from_node_id(outputs[0])?;
        let input = model.get_variable_data_from_variable_id(inputs[0])?;
        let shape = input.get_shape()?.clone();
        let size = input.size()?;
        model.clone_node_to_grad_model_if_needed(inputs[0])?;
        model.clone_node_to_grad_model_if_needed(inputs[1])?;
        let insert_model = Model::make_model(
            vec![
                ModelVariable::new("in", VariableData::None),
                ModelVariable::new("x0", VariableData::None),
                ModelVariable::new("x1", VariableData::None),
            ],
            vec![
                ModelVariable::new("out0", VariableData::None),
                ModelVariable::new("out1", VariableData::None),
            ],
            vec![
                ModelOperator::new(
                    "op0", Box::new(Sub {}),
                    vec!["x0", "x1"], vec!["diff"], vec![]
                ),
                ModelOperator::new(
                    "op1", Box::new(BroadcastTo {}),
                    vec!["in"], vec!["gy"],
                    vec![Tensor::new(shape.clone(), vec![shape.len()])?.into()]
                ),
                ModelOperator::new(
                    "op2", Box::new(Mul {}),
                    vec!["gy", "diff"], vec!["mul"], vec![]
                ),
                ModelOperator::new(
                    "op3", Box::new(ScalarMul {}),
                    vec!["mul"], vec!["out0"],
                    vec![Tensor::scalar(2.0 / size as f64).into()]
                ),
                ModelOperator::new(
                    "op4", Box::new(Neg {}),
                    vec!["out0"], vec!["out1"], vec![]
                ),
            ],
            vec![]
        )?;
        let grad_outputs = model.get_grad_model_mut()
            .insert_structure_model(insert_model, &vec![
                output_grad_id, inputs[0], inputs[1]])?;
        for (grad, node_id) in grad_outputs.into_iter().zip(inputs.clone()) {
            model.set_or_add_grad(node_id, grad)?;
        }
        Ok(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_normal() {
        use ktensor::Tensor;

        let tensor0 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])
            .unwrap();
        let tensor1 = Tensor::new(vec![2.0, 3.0, 5.0], vec![3])
            .unwrap();
        let mut model = Model::make_model(
            vec![
                ModelVariable::new("in0", tensor0.into()),
                ModelVariable::new("in1", tensor1.into()),
            ],
            vec![ModelVariable::new("out", VariableData::None)],
            vec![
                ModelOperator::new(
                    "op0", Box::new(MeanSquaredError {}),
                    vec!["in0", "in1"], vec!["out"], vec![]
                ),
            ],
            vec![]
        ).unwrap();
        model.forward().unwrap();
        let output_variable = model.get_variable_from_name("out").unwrap();
        assert_eq!(output_variable.get_type(), "F64");
        assert_eq!(output_variable.get_data(), &Tensor::new(vec![2.0], vec![]).unwrap().into());
    }

    #[test]
    fn backward_normal() {
        use ktensor::Tensor;

        let tensor0 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])
            .unwrap();
        let tensor1 = Tensor::new(vec![2.0, 3.0, 5.0], vec![3])
            .unwrap();
        let mut model = Model::make_model(
            vec![
                ModelVariable::new("in0", tensor0.into()),
                ModelVariable::new("in1", tensor1.into()),
            ],
            vec![ModelVariable::new("out", VariableData::None)],
            vec![
                ModelOperator::new(
                    "op0", Box::new(MeanSquaredError {}),
                    vec!["in0", "in1"], vec!["out"], vec![]
                ),
            ],
            vec![]
        ).unwrap();
        model.forward().unwrap();
        let output_id = model.get_node_id_from_name("out").unwrap();
        model.backward(output_id).unwrap();
        let input_grad0 = model.get_grad_from_variable_name("in0").unwrap();
        let input_grad1 = model.get_grad_from_variable_name("in1").unwrap();
        assert_eq!(input_grad0.get_type(), "F64");
        assert_eq!(input_grad0.get_data(),
            &Tensor::new(vec![-2.0 / 3.0, -2.0 / 3.0, -4.0 / 3.0], vec![3]).unwrap().into());
        assert_eq!(input_grad1.get_type(), "F64");
        assert_eq!(input_grad1.get_data(),
            &Tensor::new(vec![2.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0], vec![3]).unwrap().into());
    }
}
