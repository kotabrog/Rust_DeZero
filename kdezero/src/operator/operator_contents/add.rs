use anyhow::Result;
use super::{OperatorContents, Identity};
use crate::model::{Model, ModelVariable, ModelOperator};
use crate::variable::VariableData;

#[derive(Clone)]
pub struct Add {}

impl OperatorContents for Add {
    fn forward(
            &mut self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        model.check_inputs_len_at_least(node_id, 2)?;
        model.check_outputs_len(node_id, 1)?;
        let inputs = model.get_node_inputs_from_node_id(node_id)?.clone();
        let output = model.get_node_outputs_from_node_id(node_id)?[0];
        let variable_data0 = model.get_variable_data_from_node_id(inputs[0])?;
        let variable_data1 = model.get_variable_data_from_node_id(inputs[1])?;
        let output_data = variable_data0.add(variable_data1)?;
        let output_data = inputs.iter().skip(2).fold(
            Ok(output_data), |acc, input| {
                let input_data = model.get_variable_data_from_node_id(*input)?;
                acc?.add(input_data)
            }
        )?;
        model.set_variable_data_from_node_id(output, output_data)?;
        Ok(vec![output])
    }

    fn backward(
            &mut self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        model.check_inputs_len_at_least(node_id, 2)?;
        model.check_outputs_len(node_id, 1)?;
        let inputs = model.get_node_inputs_from_node_id(node_id)?.clone();
        let output = model.get_node_outputs_from_node_id(node_id)?[0];
        let output_grad_id = model.get_grad_id_from_node_id(output)?;
        let output_names: Vec<String> = (0..inputs.len()).into_iter()
            .map(|i| format!("out{}", i)).collect();
        let model_outputs = output_names.iter()
            .map(|name| ModelVariable::new(
                    name,
                    VariableData::None
            )).collect();
        let insert_model = Model::make_model(
            vec![ModelVariable::new(
                    "in", VariableData::None
            )],
            model_outputs,
            vec![ModelOperator::new(
                    "op", Box::new(Identity {}), vec!["in"],
                    output_names.iter().map(|s| s.as_str()).collect()
            )], vec![]
        )?;
        let grad_outputs = model.get_grad_model_mut()
            .insert_structure_model(insert_model, &vec![output_grad_id])?;
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

        let tensor1 = Tensor::new(vec![10.0], vec![])
            .unwrap();
        let tensor2 = Tensor::new(vec![20.0], vec![])
            .unwrap();
        let mut model = Model::make_model(
            vec![
                ModelVariable::new("in1", tensor1.into()),
                ModelVariable::new("in2", tensor2.into()),
            ],
            vec![ModelVariable::new(
                    "out", VariableData::None
            )],
            vec![ModelOperator::new(
                    "op", Box::new(Add {}),
                    vec!["in1", "in2"], vec!["out"]
            )],
            vec![]
        ).unwrap();
        model.forward().unwrap();
        let output_variable = model.get_variable_from_name("out").unwrap();
        assert_eq!(output_variable.get_data().to_string(), "F64");
        assert_eq!(output_variable.get_data(), &Tensor::new(vec![30.0], vec![]).unwrap().into());
    }

    #[test]
    fn backward_normal() {
        use ktensor::Tensor;

        let tensor1 = Tensor::new(vec![10.0], vec![])
            .unwrap();
        let tensor2 = Tensor::new(vec![20.0], vec![])
            .unwrap();
        let mut model = Model::make_model(
            vec![
                ModelVariable::new("in1", tensor1.into()),
                ModelVariable::new("in2", tensor2.into()),
            ],
            vec![ModelVariable::new(
                    "out", VariableData::None
            )],
            vec![ModelOperator::new(
                    "op", Box::new(Add {}),
                    vec!["in1", "in2"], vec!["out"]
            )],
            vec![]
        ).unwrap();
        model.forward().unwrap();
        let output_id = model.get_node_id_from_name("out").unwrap();
        model.backward(output_id).unwrap();
        let input1_grad_variable = model.get_grad_from_variable_name("in1").unwrap();
        let input2_grad_variable = model.get_grad_from_variable_name("in2").unwrap();
        assert_eq!(input1_grad_variable.get_data().to_string(), "F64");
        assert_eq!(input1_grad_variable.get_data(), &Tensor::new(vec![1.0], vec![]).unwrap().into());
        assert_eq!(input2_grad_variable.get_data().to_string(), "F64");
        assert_eq!(input2_grad_variable.get_data(), &Tensor::new(vec![1.0], vec![]).unwrap().into());
    }
}
