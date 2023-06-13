use super::super::{FunctionTable, operator::{Div, Add, Neg, Exp}};
use crate::variable::VariableTable;
use crate::Tensor;

pub fn sigmoid(x_id: usize, variable_table: &mut VariableTable, function_table: &mut FunctionTable) -> usize {
    let neg_id = function_table.generate_function_from_function_contents(Box::new(Neg::new()));
    let temp_id0 = function_table.forward(neg_id, vec![x_id], variable_table, false)[0];
    let exp_id = function_table.generate_function_from_function_contents(Box::new(Exp::new()));
    let temp_id1 = function_table.forward(exp_id, vec![temp_id0], variable_table, false)[0];
    let shape = variable_table
        .get(temp_id0).expect("Invalid variable id")
        .shape().clone();
    let one_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::full(1.0, shape), "");
    let add_id = function_table.generate_function_from_function_contents(Box::new(Add::new()));
    let temp_id2 = function_table.forward(add_id, vec![temp_id1, one_id], variable_table, false)[0];
    let div_id = function_table.generate_function_from_function_contents(Box::new(Div::new()));
    let ret_id = function_table.forward(div_id, vec![one_id, temp_id2], variable_table, false)[0];
    ret_id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utility::assert_approx_eq;

    #[test]
    fn forward_normal() {
        fn sigmoid_f64(x: f64) -> f64 {
            1.0 / (1.0 + (-x).exp())
        }

        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data0 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let id0 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data0.clone(), vec![3, 2]), "x");

        let output_id = sigmoid(
            id0, &mut variable_table, &mut function_table);

        let output = variable_table.get_variable_contents_f64(output_id).unwrap();
        assert_eq!(output.data(), Tensor::new_from_num_vec(data0.iter().map(|&x| sigmoid_f64(x)), vec![3, 2]).data());
    }

    #[test]
    fn backward_normal() {
        fn sigmoid_f64(x: f64) -> f64 {
            1.0 / (1.0 + (-x).exp())
        }

        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data0 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let id0 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data0.clone(), vec![3, 2]), "x");

        let output_id = sigmoid(
            id0, &mut variable_table, &mut function_table);

        variable_table.backward(vec![output_id], &mut function_table, false);

        let grad = variable_table.get_variable_grad_contents_f64(id0).unwrap();

        data0.iter().map(|&x| sigmoid_f64(x) * (1.0 - sigmoid_f64(x)))
            .zip(grad.data().iter()).for_each(|(x, y)| assert_approx_eq(x, *y.data(), 1e-6));
    }
}
