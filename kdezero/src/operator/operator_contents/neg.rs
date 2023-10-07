use anyhow::Result;
use super::OperatorContents;
use crate::model::{Model, ModelVariable, ModelOperator};
use crate::variable::VariableData;

#[derive(Clone)]
pub struct Neg {}

impl OperatorContents for Neg {
    fn forward(
            &mut self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        let (inputs, outputs) =
            model.check_inputs_outputs_len(node_id, 1, 1)?;
        let input_id = inputs[0];
        let output_id = outputs[0];
        let variable_data = model.get_variable_data_from_node_id(input_id)?;
        let output_data = variable_data.neg()?;
        model.set_variable_data_from_node_id(output_id, output_data)?;
        Ok(vec![output_id])
    }

    fn backward(
            &mut self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        let (inputs, outputs) =
            model.check_inputs_outputs_len(node_id, 1, 1)?;
        let input_id = inputs[0];
        let output_id = outputs[0];
        let output_grad_id = model.get_grad_id_from_node_id(output_id)?;
        let insert_model = Model::make_model(
            vec![ModelVariable::new("in", VariableData::None)],
            vec![ModelVariable::new("out", VariableData::None)],
            vec![ModelOperator::new(
                    "op", Box::new(Neg {}),
                    vec!["in"], vec!["out"], vec![])],
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
                    "op", Box::new(Neg {}),
                    vec!["in"], vec!["out"], vec![]
            )], vec![]
        ).unwrap();
        model.forward().unwrap();
        let output_variable = model.get_variable_from_name("out").unwrap();
        assert_eq!(output_variable.get_type(), "F64");
        assert_eq!(output_variable.get_data(), &Tensor::new(vec![-2.0], vec![]).unwrap().into());
    }

    #[test]
    fn backward_normal() {
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
                    "op", Box::new(Neg {}),
                    vec!["in"], vec!["out"], vec![]
            )], vec![]
        ).unwrap();
        model.forward().unwrap();
        let output_id = model.get_node_id_from_name("out").unwrap();
        model.backward(output_id).unwrap();
        let input_grad = model.get_grad_from_variable_name("in").unwrap();
        assert_eq!(input_grad.get_type(), "F64");
        assert_eq!(input_grad.get_data(), &Tensor::new(vec![-1.0], vec![]).unwrap().into());
    }

    /// Test backward of backward.
    /// This test is expected to panic
    /// because it doesn't get through to x
    #[test]
    #[should_panic]
    fn backward_backward_normal() {
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
                    "op", Box::new(Neg {}),
                    vec!["in"], vec!["out"], vec![]
            )], vec![]
        ).unwrap();
        model.forward().unwrap();
        let output_id = model.get_node_id_from_name("out").unwrap();
        model.backward(output_id).unwrap();
        let input_id = model.get_node_id_from_name("in").unwrap();
        let grad_id = model.get_grad_id_from_node_id(input_id).unwrap();
        let grad_model = model.get_grad_model_mut();
        grad_model.backward(grad_id).unwrap();
        let grad = grad_model.get_grad_variable_from_node_id(input_id).unwrap();
        assert_eq!(grad.get_type(), "F64");
        assert_eq!(grad.get_data(), &Tensor::new(vec![-1.0], vec![]).unwrap().into());
    }
}
