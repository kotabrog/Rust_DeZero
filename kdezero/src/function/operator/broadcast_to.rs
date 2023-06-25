use std::any::Any;
use super::SumTo;
use super::super::{FunctionContents, FunctionTable};
use crate::variable::VariableTable;

#[derive(Debug, Clone)]
pub struct BroadcastTo {
    shape: Vec<usize>,
}

impl BroadcastTo {
    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }

    fn input_check(inputs: &Vec<usize>) {
        if inputs.len() != 1 {
            panic!("BroadcastTo function must have only one input, but got {} inputs.", inputs.len());
        }
    }

    fn output_check(outputs: &Vec<usize>) {
        if outputs.len() != 1 {
            panic!("BroadcastTo function must have only one output, but got {} outputs.", outputs.len());
        }
    }
}

impl FunctionContents for BroadcastTo {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "BroadcastTo"
    }

    fn forward(&self, _info: &crate::function::FunctionInfo, inputs: &Vec<usize>, variable_table: &mut VariableTable) -> Vec<usize> {
        BroadcastTo::input_check(inputs);
        let x = variable_table.get_variable_contents_f64(inputs[0]).expect("Invalid variable id");

        let output = x.broadcast_to(&self.shape);

        let output_id = variable_table.generate_variable_from_f64_tensor(output, "");
        vec![output_id]
    }

    fn get_backward(&self) -> fn(usize, &mut FunctionTable, &mut VariableTable) -> Vec<usize> {
        |function_id, function_table, variable_table| {
            let function = function_table.get(function_id).expect("Invalid function id");
            let inputs = function.get_inputs().expect("Invalid inputs");
            let outputs = function.get_outputs().expect("Invalid outputs");
            BroadcastTo::input_check(inputs);
            BroadcastTo::output_check(outputs);
            let input_id = inputs[0];
            let input_shape = variable_table.get(input_id).expect("Invalid variable id").shape();
            let output_id = outputs[0];
            let output_grad_id = variable_table.get_variable_grad_id(output_id).expect("Output grad id not found");

            let sum_to_id = function_table.generate_function_from_function_contents(Box::new(SumTo::new(input_shape.clone())));
            let grad_id = function_table.forward(sum_to_id, vec![output_grad_id], variable_table, false)[0];

            variable_table.update_grad(input_id, grad_id, function_table);

            vec![input_id]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktensor::tensor::Tensor;
    use crate::{variable::VariableTable, function::FunctionTable};

    #[test]
    fn forward_normal() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data = vec![0.0, 1.0];
        let broadcast_to_id = function_table.generate_function_from_function_contents(Box::new(BroadcastTo::new(vec![2, 2])));
        let x_id = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data.clone(), vec![2,]), "x");
        let y_id = function_table.forward(broadcast_to_id, vec![x_id], &mut variable_table, false);

        let y = variable_table.get_variable_contents_f64(y_id[0]).unwrap();
        assert_eq!(y, &Tensor::new_from_num_vec(vec![0.0, 1.0, 0.0, 1.0], vec![2, 2]));
    }

    #[test]
    fn backward_normal() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data = vec![0.0, 1.0];
        let broadcast_to_id = function_table.generate_function_from_function_contents(Box::new(BroadcastTo::new(vec![2, 2])));
        let x_id = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data.clone(), vec![2,]), "x");
        let y_id = function_table.forward(broadcast_to_id, vec![x_id], &mut variable_table, false);

        variable_table.backward(y_id, &mut function_table, false);

        let x_grad = variable_table.get_variable_grad_contents_f64(x_id).unwrap();
        assert_eq!(x_grad, &Tensor::new_from_num_vec(vec![2.0, 2.0], vec![2,]));
    }

    #[test]
    #[should_panic]
    fn backward_backward_normal() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data = vec![0.0, 1.0];
        let broadcast_to_id = function_table.generate_function_from_function_contents(Box::new(BroadcastTo::new(vec![2, 2])));
        let x_id = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data.clone(), vec![2,]), "x");
        let y_id = function_table.forward(broadcast_to_id, vec![x_id], &mut variable_table, false);

        variable_table.backward(y_id, &mut function_table, false);

        let x_grad_id = variable_table.get_variable_grad_id(x_id).unwrap();
        variable_table.clear_grad(x_id);

        variable_table.backward(vec![x_grad_id], &mut function_table, false);

        let _ = variable_table.get_variable_grad_contents_f64(x_id).unwrap();
    }
}
