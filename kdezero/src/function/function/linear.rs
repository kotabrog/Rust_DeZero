use super::super::{FunctionTable, operator::{Add, MatMul, BroadcastTo}};
use crate::variable::VariableTable;

pub fn linear(x_id: usize, w_id: usize, b_id: Option<usize>,
              variable_table: &mut VariableTable, function_table: &mut FunctionTable) -> usize {
    let matmul_id = function_table.generate_function_from_function_contents(Box::new(MatMul::new()));
    let temp_id0 = function_table.forward(matmul_id, vec![x_id, w_id], variable_table, false)[0];
    let b_id = match b_id {
        Some(b_id) => b_id,
        None => return temp_id0
    };
    let shape = variable_table
        .get(temp_id0).expect("Invalid variable id")
        .shape().clone();
    let broadcast_to_id = function_table.generate_function_from_function_contents(Box::new(BroadcastTo::new(shape)));
    let temp_id1 = function_table.forward(broadcast_to_id, vec![b_id], variable_table, false)[0];
    let add_id = function_table.generate_function_from_function_contents(Box::new(Add::new()));
    let ret_id = function_table.forward(add_id, vec![temp_id0, temp_id1], variable_table, false)[0];
    ret_id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn forward_normal() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data0 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let data1 = vec![0.0, 1.0, 2.0, 3.0];
        let id0 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data0.clone(), vec![3, 2]), "x");
        let id1 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data1.clone(), vec![2, 2]), "y");

        let output_id = linear(
            id0, id1, None, &mut variable_table, &mut function_table);

        let output = variable_table.get_variable_contents_f64(output_id).unwrap();
        assert_eq!(output.data(), Tensor::new_from_num_vec(vec![2.0, 3.0, 6.0, 11.0, 10.0, 19.0], vec![3, 2]).data());
    }

    #[test]
    fn forward_add_b() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data0 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let data1 = vec![0.0, 1.0, 2.0, 3.0];
        let data2 = vec![0.0, 1.0];
        let id0 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data0.clone(), vec![3, 2]), "x");
        let id1 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data1.clone(), vec![2, 2]), "y");
        let id2 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data2.clone(), vec![2,]), "b");

        let output_id = linear(
            id0, id1, Some(id2), &mut variable_table, &mut function_table);

        let output = variable_table.get_variable_contents_f64(output_id).unwrap();
        assert_eq!(output.data(), Tensor::new_from_num_vec(vec![2.0, 4.0, 6.0, 12.0, 10.0, 20.0], vec![3, 2]).data());
    }

    #[test]
    fn backward_normal() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data0 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let data1 = vec![0.0, 1.0, 2.0, 3.0];
        let id0 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data0.clone(), vec![3, 2]), "x");
        let id1 = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data1.clone(), vec![2, 2]), "y");

        let output_id = linear(
            id0, id1, None, &mut variable_table, &mut function_table);

        variable_table.backward(vec![output_id], &mut function_table, false);

        let grad0 = variable_table.get_variable_grad_contents_f64(id0).unwrap();
        let grad1 = variable_table.get_variable_grad_contents_f64(id1).unwrap();
        assert_eq!(grad0, &Tensor::new_from_num_vec(vec![1.0, 5.0, 1.0, 5.0, 1.0, 5.0], vec![3, 2]));
        assert_eq!(grad1, &Tensor::new_from_num_vec(vec![6.0, 6.0, 9.0, 9.0], vec![2, 2]));
    }
}
