use crate::variable::{Variable, VariableTable, VariableType, VariableWrapper};
use super::super::{Function, FunctionInfo};

#[derive(Debug, Clone)]
pub struct Add {}

/// Add function
impl Add {
    pub fn new() -> Self {
        Self { }
    }
}

impl Function for Add {
    fn forward(&self, _info: &FunctionInfo, inputs: &Vec<usize>, variables: &mut VariableTable) -> Vec<usize> {
        if inputs.len() != 2 {
            panic!("Add error: inputs.len() != 2");
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

        let output = input1 + input2;

        let output = VariableWrapper::from_variable_f64(Variable::new(output));
        let id = variables.add(Box::new(output));
        vec![id]
    }

    fn backward(&self, info: &FunctionInfo, variables: &mut VariableTable) -> Vec<usize> {
        let outputs = info.outputs.as_ref().expect("outputs is None");
        if outputs.len() != 1 {
            panic!("Add error: outputs.len() != 1");
        }
        let output = outputs[0];
        let grad = variables.get_variable_type(output).expect("outputs is None");
        let grad = match grad {
            VariableType::F64(x) =>
                x.grad()
                .expect("grad is None"),
        };

        let gx1 = grad.clone();
        let gx2 = grad.clone();

        let input_id1 = info.inputs.as_ref().expect("input is None").get(0).expect("input size is 0");
        let input1 = variables.get_variable_type_mut(*input_id1).expect("input1 is None");
        let input1_variable = match input1 {
            VariableType::F64(x) => x,
        };
        input1_variable.set_grad(gx1);
        let input_id2 = info.inputs.as_ref().expect("input is None").get(1).expect("input size is 1");
        let input2 = variables.get_variable_type_mut(*input_id2).expect("input2 is None");
        let input2_variable = match input2 {
            VariableType::F64(x) => x,
        };
        input2_variable.set_grad(gx2);
        vec![*input_id1, *input_id2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::FunctionWrapper;
    use crate::Tensor;
    use crate::function::FunctionTable;

    #[test]
    fn add_forward() {
        let mut functions = FunctionTable::new();
        let mut variables = VariableTable::new();
        let x = Variable::new(Tensor::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]));
        let y = Variable::new(Tensor::new_from_num_vec(vec![4.0, 5.0, 6.0], vec![3]));
        let x_id = variables.add(Box::new(VariableWrapper::from_variable_f64(x)));
        let y_id = variables.add(Box::new(VariableWrapper::from_variable_f64(y)));
        let add = Add::new();
        let add_id = functions.add(FunctionWrapper::new(Box::new(add)));
        let add = functions.get_mut(add_id).expect("add is None");
        let outputs = add.call_mut(vec![x_id, y_id], &mut variables);
        let output = variables.get_variable_type(outputs[0]).expect("outputs[0] is None");
        let output = match output {
            VariableType::F64(x) => x.data(),
        };
        assert_eq!(*output, Tensor::new_from_num_vec(vec![5.0, 7.0, 9.0], vec![3]));
    }
}
