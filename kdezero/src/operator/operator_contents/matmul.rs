use anyhow::Result;
use super::{OperatorContents, Transpose};
use crate::model::{Model, ModelVariable, ModelOperator};
use crate::variable::VariableData;

#[derive(Clone)]
pub struct MatMul {}

impl OperatorContents for MatMul {
    fn forward(
            &self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        let (inputs, outputs) =
            model.check_inputs_outputs_len(node_id, 2, 1)?;
        let output = outputs[0];
        let variable_data0 = model.get_variable_data_from_node_id(inputs[0])?;
        let variable_data1 = model.get_variable_data_from_node_id(inputs[1])?;
        let output_data = variable_data0.matmul(variable_data1)?;
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
        let output_grad_id = model.get_grad_id_from_node_id(outputs[0])?;
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
                    "op0", Box::new(Transpose {}),
                    vec!["x1"], vec!["x1_T"], vec![]
                ),
                ModelOperator::new(
                    "op1", Box::new(Transpose {}),
                    vec!["x0"], vec!["x0_T"], vec![]
                ),
                ModelOperator::new(
                    "op2", Box::new(MatMul {}),
                    vec!["in", "x1_T"], vec!["out0"], vec![]
                ),
                ModelOperator::new(
                    "op3", Box::new(MatMul {}),
                    vec!["x0_T", "in"], vec!["out1"], vec![]
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
        use ktensor::tensor::TensorRng;

        let mut rng = TensorRng::new();
        let tensor0 = rng.gen::<f64, _>(vec![2, 3]);
        let tensor1 = rng.gen::<f64, _>(vec![3, 4]);
        let mut model = Model::make_model(
            vec![
                ModelVariable::new("in0", tensor0.clone().into()),
                ModelVariable::new("in1", tensor1.clone().into()),
            ],
            vec![ModelVariable::new("out", VariableData::None)],
            vec![
                ModelOperator::new(
                    "op0", Box::new(MatMul {}),
                    vec!["in0", "in1"], vec!["out"], vec![]
                ),
            ],
            vec![]
        ).unwrap();
        model.forward().unwrap();
        let output_variable = model.get_variable_from_name("out").unwrap();
        assert_eq!(output_variable.get_type(), "F64");
        assert_eq!(output_variable.get_data(),
            &tensor0.matmul(&tensor1).unwrap().into());
    }

    #[test]
    fn backward_normal() {
        use ktensor::{Tensor, tensor::TensorRng};

        let mut rng = TensorRng::new();
        let tensor0 = rng.gen::<f64, _>(vec![2, 3]);
        let tensor1 = rng.gen::<f64, _>(vec![3, 4]);
        let mut model = Model::make_model(
            vec![
                ModelVariable::new("in0", tensor0.clone().into()),
                ModelVariable::new("in1", tensor1.clone().into()),
            ],
            vec![ModelVariable::new("out", VariableData::None)],
            vec![
                ModelOperator::new(
                    "op0", Box::new(MatMul {}),
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
        let gy = Tensor::full(1.0, vec![2, 4]);
        assert_eq!(input_grad0.get_type(), "F64");
        assert_eq!(input_grad0.get_data(),
            &gy.matmul(&tensor1.transpose()).unwrap().into());
        assert_eq!(input_grad1.get_type(), "F64");
        assert_eq!(input_grad1.get_data(),
            &tensor0.transpose().matmul(&gy).unwrap().into());
    }
}