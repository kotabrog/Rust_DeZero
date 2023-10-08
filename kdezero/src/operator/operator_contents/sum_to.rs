use anyhow::Result;
use super::{OperatorContents, BroadcastTo};
use crate::model::{Model, ModelVariable, ModelOperator};
use crate::variable::VariableData;

#[derive(Clone)]
pub struct SumTo {
    pub shape: Vec<usize>,
}

impl OperatorContents for SumTo {
    fn forward(
            &mut self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        let (inputs, outputs) =
            model.check_inputs_outputs_len(node_id, 1, 1)?;
        let input_id = inputs[0];
        let output_id = outputs[0];
        let variable_data = model.get_variable_data_from_node_id(input_id)?;
        let output_data = variable_data.sum_to(&self.shape)?;
        model.set_variable_data_from_node_id(output_id, output_data)?;
        Ok(vec![output_id])
    }

    fn backward(
            &mut self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        let (inputs, outputs) =
            model.check_inputs_outputs_len(node_id, 1, 1)?;
        let input = inputs[0];
        let output = outputs[0];
        let input_shape = model.get_variable_data_from_node_id(input)?
            .get_shape()?;
        let output_grad_id = model.get_grad_id_from_node_id(output)?;
        let insert_model = Model::make_model(
            vec![ModelVariable::new("in", VariableData::None)],
            vec![ModelVariable::new("out", VariableData::None)],
            vec![ModelOperator::new(
                "op", Box::new(BroadcastTo {
                    shape: input_shape.clone()
                }),
                vec!["in"], vec!["out"]
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

        let tensor = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]).unwrap();
        let mut model = Model::make_model(
            vec![ModelVariable::new(
                "in", tensor.into()
            )],
            vec![ModelVariable::new(
                "out", VariableData::None
            )],
            vec![ModelOperator::new(
                "op", Box::new(SumTo {
                    shape: vec![3]
                }),
                vec!["in"], vec!["out"]
            )], vec![]
        ).unwrap();
        model.forward().unwrap();
        let output_variable = model.get_variable_from_name("out").unwrap();
        assert_eq!(output_variable.get_type(), "F64");
        assert_eq!(output_variable.get_data(),
            &Tensor::new(vec![3.0, 5.0, 7.0], vec![3]).unwrap().into());
    }

    #[test]
    fn backward_normal() {
        use ktensor::Tensor;

        let tensor = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]).unwrap();
        let mut model = Model::make_model(
            vec![ModelVariable::new(
                "in", tensor.clone().into()
            )],
            vec![ModelVariable::new(
                "out", VariableData::None
            )],
            vec![ModelOperator::new(
                "op", Box::new(SumTo {
                    shape: vec![3]
                }),
                vec!["in"], vec!["out"]
            )], vec![]
        ).unwrap();
        model.forward().unwrap();
        let output_id = model.get_node_id_from_name("out").unwrap();
        model.backward(output_id).unwrap();
        let input_grad = model.get_grad_from_variable_name("in").unwrap();
        assert_eq!(input_grad.get_type(), "F64");
        assert_eq!(input_grad.get_data(),
            &Tensor::full_like(1.0, &tensor).into());
    }
}
