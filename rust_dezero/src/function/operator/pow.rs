use std::any::Any;
use super::Mul;
use super::super::{FunctionContents, FunctionTable};
use crate::Tensor;
use crate::variable::VariableTable;

#[derive(Debug, Clone)]
pub struct Pow<T> {
    c: T,
}

impl<T> Pow<T> {
    pub fn new(c: T) -> Self {
        Self { c }
    }

    pub fn get_c(&self) -> &T {
        &self.c
    }

    fn input_check(inputs: &Vec<usize>) {
        if inputs.len() != 1 {
            panic!("Pow function must have only one input, but got {} inputs.", inputs.len());
        }
    }

    fn output_check(outputs: &Vec<usize>) {
        if outputs.len() != 1 {
            panic!("Pow function must have only one output, but got {} outputs.", outputs.len());
        }
    }
}

impl FunctionContents for Pow<f64> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "Pow"
    }

    fn forward(&self, _info: &crate::function::FunctionInfo, inputs: &Vec<usize>, variable_table: &mut VariableTable) -> Vec<usize> {
        Pow::<f64>::input_check(inputs);
        let x = variable_table.get_variable_contents_f64(inputs[0]).expect("Invalid variable id");

        let output = x.powf(self.c);

        let output_id = variable_table.generate_variable_from_f64_tensor(output, "");
        vec![output_id]
    }

    fn get_backward(&self) -> fn(usize, &mut FunctionTable, &mut VariableTable) -> Vec<usize> {
        |function_id, function_table, variable_table| {
            let function = function_table.get(function_id).expect("Invalid function id");
            let function_contents = function.get_function_contents::<Pow<f64>>().expect("Invalid function contents");
            let c = function_contents.get_c();

            let inputs = function.get_inputs().expect("Invalid inputs");
            let outputs = function.get_outputs().expect("Invalid outputs");
            Pow::<f64>::input_check(inputs);
            Pow::<f64>::output_check(outputs);
            let input_id = inputs[0];
            let output_id = outputs[0];
            let output_grad_id = variable_table.get_variable_grad_id(output_id).expect("Output grad id not found");

            let variable = variable_table.get_variable_contents_f64(output_id).expect("Invalid variable id");
            let const_id = variable_table.generate_variable_from_f64_tensor(
                Tensor::full_like(variable, *c), ""
            );

            let pow_id = function_table.generate_function_from_function_contents(Box::new(Pow::<f64>::new(*c - 1.0)));
            let temp_id = function_table.forward(pow_id, vec![input_id], variable_table, false)[0];
            let mul_id = function_table.generate_function_from_function_contents(Box::new(Mul::new()));
            let temp_id = function_table.forward(mul_id, vec![const_id, temp_id], variable_table, false)[0];
            let mul_id = function_table.generate_function_from_function_contents(Box::new(Mul::new()));
            let grad_id = function_table.forward(mul_id, vec![temp_id, output_grad_id], variable_table, false)[0];

            variable_table.update_grad(input_id, grad_id, function_table);

            vec![input_id]
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

        let data = vec![1.0, 2.0, 3.0];
        let pow_id = function_table.generate_function_from_function_contents(Box::new(Pow::new(3.0)));
        let x_id = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data.clone(), vec![3]), "x");
        let y_id = function_table.forward(pow_id, vec![x_id], &mut variable_table, false);

        let y = variable_table.get_variable_contents_f64(y_id[0]).unwrap();
        assert_eq!(y.data(), Tensor::new_from_num_vec(data.iter().map(|x| x.powf(3.0)), vec![3]).data());
    }

    #[test]
    fn backward_normal() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data = vec![1.0, 2.0, 3.0];
        let pow_id = function_table.generate_function_from_function_contents(Box::new(Pow::new(2.0)));
        let x_id = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data.clone(), vec![3]), "x");
        let y_ids = function_table.forward(pow_id, vec![x_id], &mut variable_table, false);

        variable_table.backward(y_ids, &mut function_table, false);

        let x_grad = variable_table.get_variable_grad_contents_f64(x_id).unwrap();
        assert_eq!(x_grad, &Tensor::new_from_num_vec(data.iter().map(|x| 2.0 * x), vec![3]));
    }

    #[test]
    fn backward_backward_normal() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data = vec![1.0, 2.0, 3.0];
        let pos_id = function_table.generate_function_from_function_contents(Box::new(Pow::new(3.0)));
        let x_id = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data.clone(), vec![3]), "x");
        let y_ids = function_table.forward(pos_id, vec![x_id], &mut variable_table, false);

        variable_table.backward(y_ids, &mut function_table, false);

        let x_grad_id = variable_table.get_variable_grad_id(x_id).unwrap();
        variable_table.clear_grad(x_id);

        variable_table.backward(vec![x_grad_id], &mut function_table, false);

        let x_grad = variable_table.get_variable_grad_contents_f64(x_id).unwrap();
        assert_eq!(x_grad, &Tensor::new_from_num_vec(data.iter().map(|x| 6.0 * x), vec![3]));
    }
}
