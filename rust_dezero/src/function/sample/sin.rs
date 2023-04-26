use crate::variable::{Variable, VariableTable, VariableType, VariableWrapper};
use super::super::{Function, FunctionInfo};

#[derive(Debug, Clone)]
pub struct Sin {}

/// Sin function
impl Sin {
    pub fn new() -> Self {
        Self { }
    }
}

impl Function for Sin {
    fn name(&self) -> String {
        "Sin".to_string()
    }

    fn forward(&self, _info: &FunctionInfo, inputs: &Vec<usize>, variables: &mut VariableTable) -> Vec<usize> {
        if inputs.len() != 1 {
            panic!("Sin error: inputs.len() != 1");
        }
        let input = inputs[0];
        let input = variables.get_variable_type(input).expect("input is None");
        let input = match input {
            VariableType::F64(x) => x.data(),
        };

        let output = input.sin();

        let output = VariableWrapper::from_variable_f64(Variable::new(output), None);
        let id = variables.add(Box::new(output));
        vec![id]
    }

    fn backward(&self, info: &FunctionInfo, variables: &mut VariableTable) -> Vec<usize> {
        let outputs = info.get_outputs_unchecked();
        if outputs.len() != 1 {
            panic!("Sin error: outputs.len() != 1");
        }
        let output = outputs[0];
        let inputs = info.get_inputs_unchecked();
        if inputs.len() != 1 {
            panic!("Sin error: inputs.len() != 1");
        }
        let input_id = inputs[0];
        let input = variables.get_variable_type(input_id).expect("input is None");
        let input = match input {
            VariableType::F64(x) => x.data(),
        };
        let grad = variables.get_variable_type(output).expect("grad is None");
        let grad = match grad {
            VariableType::F64(x) =>
                x.grad()
                .expect("grad is None"),
        };
        let gx = &input.cos() * grad;

        let input = variables.get_variable_type_mut(input_id).expect("input is None");
        let input_variable = match input {
            VariableType::F64(x) => x,
        };
        input_variable.update_grad(gx);
        vec![input_id]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::FunctionWrapper;
    use crate::Tensor;
    use crate::function::FunctionTable;

    #[test]
    fn sin_forward() {
        let data = vec![1.0, 2.0, 3.0];
        let mut variables = VariableTable::new();
        let mut functions = FunctionTable::new();
        let x = Variable::<f64>::new(Tensor::new_from_num_vec(data.clone(), vec![3]));
        let x = VariableWrapper::from_variable_f64(x, None);
        let x_id = variables.add(Box::new(x));
        let f = Sin::new();
        let f = FunctionWrapper::new(Box::new(f));
        let f_id = functions.add(f);
        let f = functions.get_mut(f_id).expect("f is None");
        let y_ids = f.call_mut(vec![x_id], &mut variables, false);
        let y = variables.get_variable_type(y_ids[0]).expect("y is None");
        let y = match y {
            VariableType::F64(x) => x,
        };
        assert_eq!(*y.data(), Tensor::new_from_num_vec(data.iter().map(|x| x.sin()), vec![3]));
    }

    #[test]
    fn sin_backward() {
        let data = vec![1.0, 2.0, 3.0];
        let mut variables = VariableTable::new();
        let mut functions = FunctionTable::new();
        let x = Variable::<f64>::new(Tensor::new_from_num_vec(data.clone(), vec![3]));
        let x = VariableWrapper::from_variable_f64(x, None);
        let x_id = variables.add(Box::new(x));
        let f = Sin::new();
        let f = FunctionWrapper::new(Box::new(f));
        let f_id = functions.add(f);
        let f = functions.get_mut(f_id).expect("f is None");
        let y_id = f.call_mut(vec![x_id], &mut variables, false);
        variables.backward(y_id, &mut functions, true);
        let x_grad = variables.get_variable_type(x_id).expect("x is None");
        let x_grad = match x_grad {
            VariableType::F64(x) => x.grad().unwrap(),
        };
        assert_eq!(x_grad, &Tensor::new_from_num_vec(data.iter().map(|x| x.cos()), vec![3]));
    }
}
