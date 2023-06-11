use std::any::Any;
use super::{Sub, BroadcastTo, Mul, Neg};
use super::super::{FunctionContents, FunctionTable};
use crate::variable::VariableTable;
use crate::Tensor;

#[derive(Debug, Clone)]
pub struct MeanSquaredError {}

impl MeanSquaredError {
    pub fn new() -> Self {
        Self {}
    }

    fn input_check(inputs: &Vec<usize>) {
        if inputs.len() != 2 {
            panic!("MeanSquaredError function must have only 2 input, but got {} inputs.", inputs.len());
        }
    }

    fn output_check(outputs: &Vec<usize>) {
        if outputs.len() != 1 {
            panic!("MeanSquaredError function must have only one output, but got {} outputs.", outputs.len());
        }
    }
}

impl FunctionContents for MeanSquaredError {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "MeanSquaredError"
    }

    fn forward(&self, _info: &crate::function::FunctionInfo, inputs: &Vec<usize>, variable_table: &mut VariableTable) -> Vec<usize> {
        MeanSquaredError::input_check(inputs);
        let input0 = variable_table.get_variable_contents_f64(inputs[0]).expect("Invalid variable id");
        let input1 = variable_table.get_variable_contents_f64(inputs[1]).expect("Invalid variable id");

        let output = (input0 - input1)
            .powi(2)
            .sum([], false)
            .scalar_div((input0.size() as f64).into());

        let output_id = variable_table.generate_variable_from_f64_tensor(output, "");
        vec![output_id]
    }

    fn get_backward(&self) -> fn(usize, &mut FunctionTable, &mut VariableTable) -> Vec<usize> {
        |function_id, function_table, variable_table| {
            let function = function_table.get(function_id).expect("Invalid function id");
            let inputs = function.get_inputs().expect("Invalid inputs");
            let outputs = function.get_outputs().expect("Invalid outputs");
            MeanSquaredError::input_check(inputs);
            MeanSquaredError::output_check(outputs);
            let input_ids = inputs.clone();
            let input0 = variable_table
                .get(input_ids[0]).expect("Invalid variable id");
            let input_shape = input0.shape().clone();
            let input_size = input0.size();
            let output_id = outputs[0];
            let output_grad_id = variable_table.get_variable_grad_id(output_id).expect("Output grad id not found");

            let sub_id = function_table.generate_function_from_function_contents(Box::new(Sub::new()));
            let diff_id = function_table.forward(sub_id, input_ids.clone(), variable_table, false)[0];
            let broadcast_to_id = function_table.generate_function_from_function_contents(Box::new(BroadcastTo::new(input_shape)));
            let broadcast_gy_id = function_table.forward(broadcast_to_id, vec![output_grad_id], variable_table, false)[0];
            let mul_id0 = function_table.generate_function_from_function_contents(Box::new(Mul::new()));
            let gx0_id = function_table.forward(mul_id0, vec![broadcast_gy_id, diff_id], variable_table, false)[0];
            let gx0 = variable_table.get_variable_contents_f64(gx0_id).expect("Invalid variable id");
            let constant_tensor = Tensor::full_like(gx0, 2.0 / input_size as f64);
            let constant_id = variable_table.generate_variable_from_f64_tensor(constant_tensor, "");
            let mul_id1 = function_table.generate_function_from_function_contents(Box::new(Mul::new()));
            let gx0_id = function_table.forward(mul_id1, vec![gx0_id, constant_id], variable_table, false)[0];
            let neg_id = function_table.generate_function_from_function_contents(Box::new(Neg::new()));
            let gx1_id = function_table.forward(neg_id, vec![gx0_id], variable_table, false)[0];

            variable_table.update_grad(input_ids[0], gx0_id, function_table);
            variable_table.update_grad(input_ids[1], gx1_id, function_table);

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
        let data1 = vec![3.0, 2.0, 2.0];
        let mse_id = function_table.generate_function_from_function_contents(Box::new(MeanSquaredError::new()));
        let id0 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data0.clone(), vec![3]), "x");
        let id1 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data1.clone(), vec![3]), "y");

        let output_ids = function_table.forward(mse_id, vec![id0, id1], &mut variable_table, false);

        let output = variable_table.get_variable_contents_f64(output_ids[0]).unwrap();
        assert_eq!(output, &Tensor::new_from_num_vec(vec![5.0 / 3.0], vec![]));
    }

    #[test]
    fn backward_normal() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data0 = vec![1.0, 2.0, 3.0];
        let data1 = vec![3.0, 2.0, 2.0];
        let mse_id = function_table.generate_function_from_function_contents(Box::new(MeanSquaredError::new()));
        let id0 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data0.clone(), vec![3]), "x");
        let id1 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data1.clone(), vec![3]), "y");

        let output_ids = function_table.forward(mse_id, vec![id0, id1], &mut variable_table, false);

        variable_table.backward(output_ids, &mut function_table, false);

        let grad0 = variable_table.get_variable_grad_contents_f64(id0).unwrap();
        let grad1 = variable_table.get_variable_grad_contents_f64(id1).unwrap();
        assert_eq!(grad0, &Tensor::new_from_num_vec(vec![-4.0 / 3.0, 0.0, 2.0 / 3.0], vec![3]));
        assert_eq!(grad1, &Tensor::new_from_num_vec(vec![4.0 / 3.0, 0.0, -2.0 / 3.0], vec![3]));
    }

    #[test]
    fn backward_backward_x_2() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data = vec![1.0, 2.0, 3.0];
        let mse_id = function_table.generate_function_from_function_contents(Box::new(MeanSquaredError::new()));
        let id = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data.clone(), vec![3]), "x");

        let output_ids = function_table.forward(mse_id, vec![id, id], &mut variable_table, false);

        variable_table.backward(output_ids, &mut function_table, false);

        let grad0_id = variable_table.get_variable_grad_id(id).unwrap();
        variable_table.clear_grad(id);

        variable_table.backward(vec![grad0_id], &mut function_table, false);

        let grad = variable_table.get_variable_grad_contents_f64(id).unwrap();
        assert_eq!(grad, &Tensor::new_from_num_vec(vec![0.0, 0.0, 0.0], vec![3]));
    }

    #[test]
    fn backward_backward_x_y() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data0 = vec![1.0, 2.0, 3.0];
        let data1 = vec![3.0, 2.0, 2.0];
        let mse_id = function_table.generate_function_from_function_contents(Box::new(MeanSquaredError::new()));
        let id0 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data0.clone(), vec![3]), "x");
        let id1 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data1.clone(), vec![3]), "y");

        let output_ids = function_table.forward(mse_id, vec![id0, id1], &mut variable_table, false);

        variable_table.backward(output_ids, &mut function_table, false);

        let grad0_id = variable_table.get_variable_grad_id(id0).unwrap();
        variable_table.clear_grad(id0);

        variable_table.backward(vec![grad0_id], &mut function_table, false);

        variable_table.get_variable_grad_contents_f64(id0).unwrap();
    }
}
