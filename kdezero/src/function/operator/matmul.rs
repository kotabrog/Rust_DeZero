use std::any::Any;
use super::Transpose;
use super::super::{FunctionContents, FunctionTable};
use crate::variable::VariableTable;

#[derive(Debug, Clone)]
pub struct MatMul {}

impl MatMul {
    pub fn new() -> Self {
        Self {}
    }

    fn input_check(inputs: &Vec<usize>) {
        if inputs.len() != 2 {
            panic!("MatMul function must have only 2 input, but got {} inputs.", inputs.len());
        }
    }

    fn output_check(outputs: &Vec<usize>) {
        if outputs.len() != 1 {
            panic!("MatMul function must have only one output, but got {} outputs.", outputs.len());
        }
    }
}

impl FunctionContents for MatMul {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "MatMul"
    }

    fn forward(&self, _info: &crate::function::FunctionInfo, inputs: &Vec<usize>, variable_table: &mut VariableTable) -> Vec<usize> {
        MatMul::input_check(inputs);
        let input0 = variable_table.get_variable_contents_f64(inputs[0]).expect("Invalid variable id");
        let input1 = variable_table.get_variable_contents_f64(inputs[1]).expect("Invalid variable id");

        let output = input0.matmul(input1);

        let output_id = variable_table.generate_variable_from_f64_tensor(output, "");
        vec![output_id]
    }

    fn get_backward(&self) -> fn(usize, &mut FunctionTable, &mut VariableTable) -> Vec<usize> {
        |function_id, function_table, variable_table| {
            let function = function_table.get(function_id).expect("Invalid function id");
            let inputs = function.get_inputs().expect("Invalid inputs");
            let outputs = function.get_outputs().expect("Invalid outputs");
            MatMul::input_check(inputs);
            MatMul::output_check(outputs);
            let input_ids = inputs.clone();
            let output_id = outputs[0];
            let output_grad_id = variable_table.get_variable_grad_id(output_id).expect("Output grad id not found");

            let transpose_id0 = function_table.generate_function_from_function_contents(Box::new(Transpose::new()));
            let input1_t_id = function_table.forward(transpose_id0, vec![input_ids[1]], variable_table, false)[0];
            let matmul_id0 = function_table.generate_function_from_function_contents(Box::new(MatMul::new()));
            let grad_id0 = function_table.forward(matmul_id0, vec![output_grad_id, input1_t_id], variable_table, false)[0];

            let transpose_id1 = function_table.generate_function_from_function_contents(Box::new(Transpose::new()));
            let input0_t_id = function_table.forward(transpose_id1, vec![input_ids[0]], variable_table, false)[0];
            let matmul_id1 = function_table.generate_function_from_function_contents(Box::new(MatMul::new()));
            let grad_id1 = function_table.forward(matmul_id1, vec![input0_t_id, output_grad_id], variable_table, false)[0];

            variable_table.update_grad(input_ids[0], grad_id0, function_table);
            variable_table.update_grad(input_ids[1], grad_id1, function_table);

            input_ids
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktensor::Tensor;
    use crate::{variable::VariableTable, function::FunctionTable};

    #[test]
    fn forward_normal() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data0 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let data1 = vec![0.0, 1.0, 2.0, 3.0];
        let matmul_id = function_table.generate_function_from_function_contents(Box::new(MatMul::new()));
        let id0 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data0.clone(), vec![3, 2]), "x");
        let id1 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data1.clone(), vec![2, 2]), "y");
        
        let output_ids = function_table.forward(matmul_id, vec![id0, id1], &mut variable_table, false);

        let output = variable_table.get_variable_contents_f64(output_ids[0]).unwrap();
        assert_eq!(output.data(), Tensor::new_from_num_vec(vec![2.0, 3.0, 6.0, 11.0, 10.0, 19.0], vec![3, 2]).data());
    }

    #[test]
    fn backward_normal() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data0 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let data1 = vec![0.0, 1.0, 2.0, 3.0];
        let matmul_id = function_table.generate_function_from_function_contents(Box::new(MatMul::new()));
        let id0 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data0.clone(), vec![3, 2]), "x");
        let id1 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data1.clone(), vec![2, 2]), "y");

        let output_ids = function_table.forward(matmul_id, vec![id0, id1], &mut variable_table, false);

        variable_table.backward(output_ids, &mut function_table, false);

        let grad0 = variable_table.get_variable_grad_contents_f64(id0).unwrap();
        let grad1 = variable_table.get_variable_grad_contents_f64(id1).unwrap();
        assert_eq!(grad0, &Tensor::new_from_num_vec(vec![1.0, 5.0, 1.0, 5.0, 1.0, 5.0], vec![3, 2]));
        assert_eq!(grad1, &Tensor::new_from_num_vec(vec![6.0, 6.0, 9.0, 9.0], vec![2, 2]));
    }

    #[test]
    fn backward_backward_x_2() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data = vec![0.0, 1.0, 2.0, 3.0];
        let matmul_id = function_table.generate_function_from_function_contents(Box::new(MatMul::new()));
        let id = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data.clone(), vec![2, 2]), "x");

        let output_ids = function_table.forward(matmul_id, vec![id, id], &mut variable_table, false);

        variable_table.backward(output_ids, &mut function_table, false);

        let grad0_id = variable_table.get_variable_grad_id(id).unwrap();
        variable_table.clear_grad(id);

        variable_table.backward(vec![grad0_id], &mut function_table, false);

        let grad = variable_table.get_variable_grad_contents_f64(id).unwrap();
        assert_eq!(grad, &Tensor::new_from_num_vec(vec![4.0, 4.0, 4.0, 4.0], vec![2, 2]));
    }

    /// Test for backward of backward
    /// This test is expected to panic
    /// because it doesn't get through to x (probably)
    #[test]
    #[should_panic]
    fn backward_backward_x_y() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data0 = vec![1.0, 2.0, 3.0];
        let data1 = vec![4.0, 5.0, 6.0];
        let matmul_id = function_table.generate_function_from_function_contents(Box::new(MatMul::new()));
        let id0 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data0.clone(), vec![3]), "x");
        let id1 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data1.clone(), vec![3]), "y");

        let output_ids = function_table.forward(matmul_id, vec![id0, id1], &mut variable_table, false);

        variable_table.backward(output_ids, &mut function_table, false);

        let grad0_id = variable_table.get_variable_grad_id(id0).unwrap();
        variable_table.clear_grad(id0);

        variable_table.backward(vec![grad0_id], &mut function_table, false);

        variable_table.get_variable_grad_contents_f64(id0).unwrap();
    }
}
