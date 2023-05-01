use super::super::{FunctionContents, FunctionTable};
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
        let pow_id = function_table.generate_function_from_function_contents(Box::new(Pow::new(3.0)));
        let x_id = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data.clone(), vec![3]), "x");
        let y_id = function_table.forward(pow_id, vec![x_id], &mut variable_table, false);

        let y = variable_table.get_variable_contents_f64(y_id[0]).unwrap();
        assert_eq!(y.data(), Tensor::new_from_num_vec(data.iter().map(|x| x.powf(3.0)), vec![3]).data());
    }
}
