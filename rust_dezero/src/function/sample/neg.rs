use crate::variable::{Variable, VariableTable, VariableType, VariableWrapper};
use super::super::{Function, FunctionInfo};

#[derive(Debug, Clone)]
pub struct Neg {}

/// Neg function
impl Neg {
    pub fn new() -> Self {
        Self { }
    }
}

impl Function for Neg {
    fn forward(&self, _info: &FunctionInfo, inputs: &Vec<usize>, variables: &mut VariableTable) -> Vec<usize> {
        if inputs.len() != 1 {
            panic!("Neg error: inputs.len() != 1");
        }
        let input = inputs[0];
        let input = variables.get_variable_type(input).expect("input is None");
        let input = match input {
            VariableType::F64(x) => x.data(),
        };

        let output = - input;

        let output = VariableWrapper::from_variable_f64(Variable::new(output), None);
        let id = variables.add(Box::new(output));
        vec![id]
    }

    fn backward(&self, info: &FunctionInfo, variables: &mut VariableTable) -> Vec<usize> {
        let outputs = info.get_outputs_unchecked();
        if outputs.len() != 1 {
            panic!("Neg error: outputs.len() != 1");
        }
        let output = outputs[0];
        let grad = variables.get_variable_type(output).expect("outputs is None");
        let grad = match grad {
            VariableType::F64(x) =>
                x.grad()
                .expect("grad is None"),
        };

        let gx = - grad;

        let inputs = info.get_inputs_unchecked();
        if inputs.len() != 1 {
            panic!("Neg error: inputs.len() != 1");
        }
        let input = variables.get_variable_type_mut(inputs[0]).expect("input is None");
        let input_variable = match input {
            VariableType::F64(x) => x,
        };
        input_variable.update_grad(gx);
        vec![inputs[0]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::FunctionWrapper;
    use crate::Tensor;
    use crate::function::FunctionTable;

    #[test]
    fn neg_forward() {
        let mut functions = FunctionTable::new();
        let mut variables = VariableTable::new();
        let x = Variable::new(Tensor::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]));
        let x_id = variables.add(Box::new(VariableWrapper::from_variable_f64(x, None)));
        let neg = Neg::new();
        let neg_id = functions.add(FunctionWrapper::new(Box::new(neg)));
        let neg = functions.get_mut(neg_id).expect("neg is None");
        let outputs = neg.call_mut(vec![x_id], &mut variables, false);
        let output = variables.get_variable_type(outputs[0]).expect("outputs[0] is None");
        let output = match output {
            VariableType::F64(x) => x.data(),
        };
        assert_eq!(*output, Tensor::new_from_num_vec(vec![-1.0, -2.0, -3.0], vec![3]));
    }

    #[test]
    fn neg_backward() {
        let mut functions = FunctionTable::new();
        let mut variables = VariableTable::new();
        let x = Variable::new(Tensor::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]));
        let x_id = variables.add(Box::new(VariableWrapper::from_variable_f64(x, None)));
        let neg = Neg::new();
        let neg_id = functions.add(FunctionWrapper::new(Box::new(neg)));
        let neg = functions.get_mut(neg_id).expect("neg is None");
        let outputs = neg.call_mut(vec![x_id], &mut variables, false);
        variables.backward(outputs, &mut functions, true);
        let x = variables.get_variable_type(x_id).expect("x is None");
        let x_grad = match x {
            VariableType::F64(x) => x.grad().unwrap(),
        };
        assert_eq!(*x_grad, Tensor::new_from_num_vec(vec![-1.0, -1.0, -1.0], vec![3]));
    }
}
