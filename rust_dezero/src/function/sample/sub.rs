use crate::variable::{Variable, VariableTable, VariableType, VariableWrapper};
use super::super::{Function, FunctionInfo};

#[derive(Debug, Clone)]
pub struct Sub {}

/// Sub function
impl Sub {
    pub fn new() -> Self {
        Self { }
    }
}

impl Function for Sub {
    fn forward(&self, _info: &FunctionInfo, inputs: &Vec<usize>, variables: &mut VariableTable) -> Vec<usize> {
        if inputs.len() != 2 {
            panic!("Sub error: inputs.len() != 2");
        }
        let input1 = inputs[0];
        let input2 = inputs[1];
        let input1 = variables.get_variable_type(input1).expect("input1 is None");
        let input1 = match input1 {
            VariableType::F64(x) => x.data(),
        };
        let input2 = variables.get_variable_type(input2).expect("input2 is None");
        let input2 = match input2 {
            VariableType::F64(x) => x.data(),
        };

        let output = input1 - input2;

        let output = VariableWrapper::from_variable_f64(Variable::new(output), None);
        let id = variables.add(Box::new(output));
        vec![id]
    }

    fn backward(&self, info: &FunctionInfo, variables: &mut VariableTable) -> Vec<usize> {
        let outputs = info.get_outputs_unchecked();
        if outputs.len() != 1 {
            panic!("Sub error: outputs.len() != 1");
        }
        let output = outputs[0];
        let grad = variables.get_variable_type(output).expect("outputs is None");
        let grad = match grad {
            VariableType::F64(x) =>
                x.grad()
                .expect("grad is None"),
        };

        let gx1 = grad.clone();
        let gx2 = - grad;

        let inputs = info.get_inputs_unchecked();
        if inputs.len() != 2 {
            panic!("Sub error: inputs.len() != 2");
        }
        let input_id1 = inputs[0];
        let input1 = variables.get_variable_type_mut(input_id1).expect("input1 is None");
        let input1_variable = match input1 {
            VariableType::F64(x) => x,
        };
        input1_variable.update_grad(gx1);
        let input_id2 = inputs[1];
        let input2 = variables.get_variable_type_mut(input_id2).expect("input2 is None");
        let input2_variable = match input2 {
            VariableType::F64(x) => x,
        };
        input2_variable.update_grad(gx2);
        vec![input_id1, input_id2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::FunctionWrapper;
    use crate::Tensor;
    use crate::function::FunctionTable;

    #[test]
    fn sub_forward() {
        let mut functions = FunctionTable::new();
        let mut variables = VariableTable::new();
        let x = Variable::new(Tensor::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]));
        let y = Variable::new(Tensor::new_from_num_vec(vec![4.0, 5.0, 6.0], vec![3]));
        let x_id = variables.add(Box::new(VariableWrapper::from_variable_f64(x, None)));
        let y_id = variables.add(Box::new(VariableWrapper::from_variable_f64(y, None)));
        let sub = Sub::new();
        let sub_id = functions.add(FunctionWrapper::new(Box::new(sub)));
        let sub = functions.get_mut(sub_id).expect("sub is None");
        let outputs = sub.call_mut(vec![x_id, y_id], &mut variables, false);
        let output = variables.get_variable_type(outputs[0]).expect("outputs[0] is None");
        let output = match output {
            VariableType::F64(x) => x.data(),
        };
        assert_eq!(*output, Tensor::new_from_num_vec(vec![-3.0, -3.0, -3.0], vec![3]));
    }

    #[test]
    fn sub_backward() {
        let mut functions = FunctionTable::new();
        let mut variables = VariableTable::new();
        let x = Variable::new(Tensor::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]));
        let y = Variable::new(Tensor::new_from_num_vec(vec![4.0, 5.0, 6.0], vec![3]));
        let x_id = variables.add(Box::new(VariableWrapper::from_variable_f64(x, None)));
        let y_id = variables.add(Box::new(VariableWrapper::from_variable_f64(y, None)));
        let sub = Sub::new();
        let sub_id = functions.add(FunctionWrapper::new(Box::new(sub)));
        let sub = functions.get_mut(sub_id).expect("sub is None");
        let outputs = sub.call_mut(vec![x_id, y_id], &mut variables, false);
        variables.backward(outputs, &mut functions, true);
        let x = variables.get_variable_type(x_id).expect("x is None");
        let x_grad = match x {
            VariableType::F64(x) => x.grad().unwrap(),
        };
        assert_eq!(*x_grad, Tensor::new_from_num_vec(vec![1.0, 1.0, 1.0], vec![3]));
        let y = variables.get_variable_type(y_id).expect("y is None");
        let y_grad = match y {
            VariableType::F64(y) => y.grad().unwrap(),
        };
        assert_eq!(*y_grad, Tensor::new_from_num_vec(vec![-1.0, -1.0, -1.0], vec![3]));
    }
}
