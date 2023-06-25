use std::any::Any;
use super::Neg;
use super::super::{FunctionContents, FunctionTable};
use crate::variable::VariableTable;

#[derive(Debug, Clone)]
pub struct Sub {}

impl Sub {
    pub fn new() -> Self {
        Self {}
    }

    fn input_check(inputs: &Vec<usize>) {
        if inputs.len() != 2 {
            panic!("Sub function must have only 2 input, but got {} inputs.", inputs.len());
        }
    }

    fn output_check(outputs: &Vec<usize>) {
        if outputs.len() != 1 {
            panic!("Sub function must have only one output, but got {} outputs.", outputs.len());
        }
    }
}

impl FunctionContents for Sub {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "Sub"
    }

    fn forward(&self, _info: &crate::function::FunctionInfo, inputs: &Vec<usize>, variable_table: &mut VariableTable) -> Vec<usize> {
        Sub::input_check(inputs);
        let input0 = variable_table.get_variable_contents_f64(inputs[0]).expect("Invalid variable id");
        let input1 = variable_table.get_variable_contents_f64(inputs[1]).expect("Invalid variable id");

        let output = input0 - input1;

        let output_id = variable_table.generate_variable_from_f64_tensor(output, "");
        vec![output_id]
    }

    fn get_backward(&self) -> fn(usize, &mut FunctionTable, &mut VariableTable) -> Vec<usize> {
        |function_id, function_table, variable_table| {
            let function = function_table.get(function_id).expect("Invalid function id");
            let inputs = function.get_inputs().expect("Invalid inputs");
            let outputs = function.get_outputs().expect("Invalid outputs");
            Sub::input_check(inputs);
            Sub::output_check(outputs);
            let input_ids = inputs.clone();
            let output_id = outputs[0];
            let output_grad_id = variable_table.get_variable_grad_id(output_id).expect("Output grad id not found");

            let neg_id = function_table.generate_function_from_function_contents(Box::new(Neg::new()));
            let grad_id1 = function_table.forward(neg_id, vec![output_grad_id], variable_table, false)[0];

            variable_table.update_grad(input_ids[0], output_grad_id, function_table);
            variable_table.update_grad(input_ids[1], grad_id1, function_table);

            input_ids
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Tensor, variable::VariableTable, function::FunctionTable};

    #[test]
    fn forward_normal() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data0 = vec![1.0, 2.0, 3.0];
        let data1 = vec![4.0, 5.0, 6.0];
        let sub_id = function_table.generate_function_from_function_contents(Box::new(Sub::new()));
        let id0 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data0.clone(), vec![3]), "x");
        let id1 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data1.clone(), vec![3]), "y");

        let output_ids = function_table.forward(sub_id, vec![id0, id1], &mut variable_table, false);

        let output = variable_table.get_variable_contents_f64(output_ids[0]).unwrap();
        assert_eq!(output, &Tensor::new_from_num_vec(vec![-3.0, -3.0, -3.0], vec![3]));
    }

    #[test]
    fn backward_normal() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data0 = vec![1.0, 2.0, 3.0];
        let data1 = vec![4.0, 5.0, 6.0];
        let sub_id = function_table.generate_function_from_function_contents(Box::new(Sub::new()));
        let id0 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data0.clone(), vec![3]), "x");
        let id1 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data1.clone(), vec![3]), "y");

        let output_ids = function_table.forward(sub_id, vec![id0, id1], &mut variable_table, false);

        variable_table.backward(output_ids, &mut function_table, false);

        let grad0 = variable_table.get_variable_grad_contents_f64(id0).unwrap();
        let grad1 = variable_table.get_variable_grad_contents_f64(id1).unwrap();
        assert_eq!(grad0, &Tensor::new_from_num_vec(vec![1.0, 1.0, 1.0], vec![3]));
        assert_eq!(grad1, &Tensor::new_from_num_vec(vec![-1.0, -1.0, -1.0], vec![3]));
    }

    /// Test for backward of backward
    /// This test is expected to panic
    /// because it doesn't get through to x (probably)
    #[test]
    #[should_panic]
    fn backward_backward_normal() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data = vec![1.0, 2.0, 3.0];
        let sub_id = function_table.generate_function_from_function_contents(Box::new(Sub::new()));
        let id = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data.clone(), vec![3]), "x");

        let output_ids = function_table.forward(sub_id, vec![id, id], &mut variable_table, false);

        variable_table.backward(output_ids, &mut function_table, false);

        let grad0_id = variable_table.get_variable_grad_id(id).unwrap();
        variable_table.clear_grad(id);

        variable_table.backward(vec![grad0_id], &mut function_table, false);

        variable_table.get_variable_grad_contents_f64(id).unwrap();
    }
}
