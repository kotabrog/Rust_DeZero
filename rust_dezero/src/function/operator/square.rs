use super::super::FunctionContents;
use crate::variable::VariableTable;

#[derive(Debug, Clone)]
pub struct Square {}

impl Square {
    pub fn new() -> Self {
        Self {}
    }

    fn input_check(&self, inputs: &Vec<usize>) {
        if inputs.len() != 1 {
            panic!("Square function must have only one input, but got {} inputs.", inputs.len());
        }
    }
}

impl FunctionContents for Square {
    fn name(&self) -> &str {
        "Square"
    }

    fn forward(&self, _info: &crate::function::FunctionInfo, inputs: &Vec<usize>, variable_table: &mut VariableTable) -> Vec<usize> {
        self.input_check(inputs);
        let x = variable_table.get_variable_contents_f64(inputs[0]).expect("Invalid variable id");

        let output = x.powi(2);

        let output_id = variable_table.generate_variable_from_f64_tensor(output, "");
        vec![output_id]
    }

    fn backward(&self, _info: &crate::function::FunctionInfo, _variable_table: &mut VariableTable) -> Vec<usize> {
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
        let square_id = function_table.generate_function_from_function_contents(Box::new(Square::new()));
        let x_id = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data.clone(), vec![3]), "x");
        let y_id = function_table.forward(square_id, vec![x_id], &mut variable_table, false);

        let y = variable_table.get_variable_contents_f64(y_id[0]).unwrap();
        assert_eq!(y.data(), Tensor::new_from_num_vec(vec![1.0, 4.0, 9.0], vec![3]).data());
    }
}
