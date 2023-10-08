use anyhow::Result;
use super::{OperatorContents, Reshape, BroadcastTo};
use crate::model::{Model, ModelVariable, ModelOperator};
use crate::variable::VariableData;

#[derive(Clone)]
pub struct Sum {
    pub axis: Option<Vec<usize>>,
    pub keepdims: bool,
}

impl OperatorContents for Sum {
    fn forward(
            &mut self, node_id: usize,
            model: &mut Model,
        ) -> Result<Vec<usize>> {
        let (inputs, outputs) =
            model.check_inputs_outputs_len(node_id, 1, 1)?;
        let input_id = inputs[0];
        let output_id = outputs[0];
        let variable_data = model.get_variable_data_from_node_id(input_id)?;
        let output_data = variable_data.sum(self.axis.as_ref(), self.keepdims)?;
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
        let input_data = model.get_variable_data_from_node_id(input_id)?;
        let input_shape = input_data.get_shape()?;
        let output_data = model.get_variable_data_from_node_id(output_id)?;
        let output_shape = output_data.get_shape()?;
        let output_grad_id = model.get_grad_id_from_node_id(output_id)?;
        let mut operators = Vec::new();
        if !self.keepdims && input_shape.len() != 0 {
            let mut axis = match self.axis.clone() {
                Some(axis) => axis,
                None => (0..input_shape.len()).collect(),
            };
            axis.sort();
            let mut output_shape = output_shape.clone();
            for i in axis.iter() {
                output_shape.insert(*i, 1);
            }
            operators.push(
                ModelOperator::new(
                    "op0", Box::new(Reshape {
                        shape: output_shape,
                    }),
                    vec!["in"], vec!["reshape"], vec![]
                )
            );
            operators.push(
                ModelOperator::new(
                    "op1", Box::new(BroadcastTo {
                        shape: input_shape.clone()
                    }),
                    vec!["reshape"], vec!["out"], vec![]
                )
            );
        } else {
            operators.push(
                ModelOperator::new(
                    "op", Box::new(BroadcastTo {
                        shape: input_shape.clone()
                    }),
                    vec!["in"], vec!["out"], vec![]
                )
            );
        }
        let insert_model = Model::make_model(
            vec![ModelVariable::new("in", VariableData::None)],
            vec![ModelVariable::new("out", VariableData::None)],
            operators, vec![]
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

        let tensor = Tensor::<f64>::arrange([2, 3]).unwrap();
        let mut model = Model::make_model(
            vec![ModelVariable::new(
                "in", tensor.into()
            )],
            vec![ModelVariable::new(
                "out", VariableData::None
            )],
            vec![ModelOperator::new(
                "op", Box::new(Sum {
                    axis: Some(vec![0]),
                    keepdims: false,
                }),
                vec!["in"], vec!["out"], vec![]
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

        let tensor = Tensor::<f64>::arrange([2, 3]).unwrap();
        let mut model = Model::make_model(
            vec![ModelVariable::new(
                "in", tensor.clone().into()
            )],
            vec![ModelVariable::new(
                "out", VariableData::None
            )],
            vec![ModelOperator::new(
                "op", Box::new(Sum {
                    axis: Some(vec![0]),
                    keepdims: true,
                }),
                vec!["in"], vec!["out"], vec![]
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

    #[test]
    fn backward_keepdims_false() {
        use ktensor::Tensor;

        let tensor = Tensor::<f64>::arrange([2, 3]).unwrap();
        let mut model = Model::make_model(
            vec![ModelVariable::new(
                "in", tensor.clone().into()
            )],
            vec![ModelVariable::new(
                "out", VariableData::None
            )],
            vec![ModelOperator::new(
                "op", Box::new(Sum {
                    axis: Some(vec![0]),
                    keepdims: false,
                }),
                vec!["in"], vec!["out"], vec![]
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
