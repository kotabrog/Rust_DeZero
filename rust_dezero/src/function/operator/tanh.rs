use super::super::{FunctionContents, FunctionTable};
use crate::variable::VariableTable;

#[derive(Debug, Clone)]
pub struct Tanh {}

impl Tanh {
    pub fn new() -> Self {
        Self {}
    }

    fn input_check(inputs: &Vec<usize>) {
        if inputs.len() != 1 {
            panic!("Tanh function must have only one input, but got {} inputs.", inputs.len());
        }
    }

    fn output_check(outputs: &Vec<usize>) {
        if outputs.len() != 1 {
            panic!("Tanh function must have only one output, but got {} outputs.", outputs.len());
        }
    }
}

impl FunctionContents for Tanh {
    fn name(&self) -> &str {
        "Tanh"
    }

    fn forward(&self, _info: &crate::function::FunctionInfo, inputs: &Vec<usize>, variable_table: &mut VariableTable) -> Vec<usize> {
        Tanh::input_check(inputs);
        let x = variable_table.get_variable_contents_f64(inputs[0]).expect("Invalid variable id");

        let output = x.tanh();

        let output_id = variable_table.generate_variable_from_f64_tensor(output, "");
        vec![output_id]
    }

    fn get_backward(&self) -> fn(usize, &mut FunctionTable, &mut VariableTable) -> Vec<usize> {
        todo!()
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

        let data = vec![1.0, 2.0, 3.0];
        let tanh_id = function_table.generate_function_from_function_contents(Box::new(Tanh::new()));
        let x_id = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data.clone(), vec![3]), "x");
        let y_id = function_table.forward(tanh_id, vec![x_id], &mut variable_table, false);

        let y = variable_table.get_variable_contents_f64(y_id[0]).unwrap();
        assert_eq!(y, &Tensor::new_from_num_vec(data.iter().map(|x| x.tanh()), vec![3]));
    }

    // #[test]
    // fn backward_normal() {
    //     let mut variable_table = VariableTable::new();
    //     let mut function_table = FunctionTable::new();

    //     let data = vec![1.0, 2.0, 3.0];
    //     let tanh_id = function_table.generate_function_from_function_contents(Box::new(Tanh::new()));
    //     let x_id = variable_table.generate_variable_from_f64_tensor(
    //         Tensor::new_from_num_vec(data.clone(), vec![3]), "x");
    //     let y_ids = function_table.forward(tanh_id, vec![x_id], &mut variable_table, false);

    //     variable_table.backward(y_ids, &mut function_table, false);

    //     let x_grad = variable_table.get_variable_grad_contents_f64(x_id).unwrap();
    //     assert_eq!(x_grad, &Tensor::new_from_num_vec(data.iter().map(|x| x.tanh()), vec![3]));
    // }

    // #[test]
    // fn backward_backward_normal() {
    //     let mut variable_table = VariableTable::new();
    //     let mut function_table = FunctionTable::new();

    //     let data = vec![1.0, 2.0, 3.0];
    //     let tanh_id = function_table.generate_function_from_function_contents(Box::new(Tanh::new()));
    //     let x_id = variable_table.generate_variable_from_f64_tensor(
    //         Tensor::new_from_num_vec(data.clone(), vec![3]), "x");
    //     let y_ids = function_table.forward(tanh_id, vec![x_id], &mut variable_table, false);

    //     variable_table.backward(y_ids, &mut function_table, false);

    //     let x_grad_id = variable_table.get_variable_grad_id(x_id).unwrap();
    //     variable_table.clear_grad(x_id);

    //     variable_table.backward(vec![x_grad_id], &mut function_table, false);

    //     let x_grad = variable_table.get_variable_grad_contents_f64(x_id).unwrap();
    //     assert_eq!(x_grad, &Tensor::new_from_num_vec(data.iter().map(|x| x.tanh()), vec![3]));
    // }
}
