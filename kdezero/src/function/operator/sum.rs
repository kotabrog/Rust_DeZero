use std::any::Any;
use super::{BroadcastTo, Reshape};
use super::super::{FunctionContents, FunctionTable};
use crate::variable::VariableTable;

#[derive(Debug, Clone)]
pub struct Sum {
    axis: Option<Vec<usize>>,
    keepdims: bool,
}

impl Sum {
    pub fn new<T: AsRef<[usize]>>(axis: Option<T>, keepdims: bool) -> Self {
        Self { axis: axis.map(|x| x.as_ref().to_vec()), keepdims }
    }

    pub fn get_axis(&self) -> Option<&Vec<usize>> {
        self.axis.as_ref()
    }

    pub fn get_keepdims(&self) -> bool {
        self.keepdims
    }

    fn input_check(inputs: &Vec<usize>) {
        if inputs.len() != 1 {
            panic!("Sum function must have only 1 input, but got {} inputs.", inputs.len());
        }
    }

    fn output_check(outputs: &Vec<usize>) {
        if outputs.len() != 1 {
            panic!("Sum function must have only one output, but got {} outputs.", outputs.len());
        }
    }
}

impl FunctionContents for Sum {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "Sum"
    }

    fn forward(&self, _info: &crate::function::FunctionInfo, inputs: &Vec<usize>, variable_table: &mut VariableTable) -> Vec<usize> {
        Sum::input_check(inputs);
        let input = variable_table.get_variable_contents_f64(inputs[0]).expect("Invalid variable id");

        let output = input.sum(
            match &self.axis {
                Some(axis) => axis.as_slice(),
                None => &[],
            },
            self.keepdims,
        );

        let output_id = variable_table.generate_variable_from_f64_tensor(output, "");
        vec![output_id]
    }

    fn get_backward(&self) -> fn(usize, &mut FunctionTable, &mut VariableTable) -> Vec<usize> {
        |function_id, function_table, variable_table| {
            let function = function_table.get(function_id).expect("Invalid function id");
            let function_contents = function.get_function_contents::<Sum>().expect("Invalid function contents");
            let axis = function_contents.get_axis();
            let keepdims = function_contents.get_keepdims();

            let inputs = function.get_inputs().expect("Invalid inputs");
            let outputs = function.get_outputs().expect("Invalid outputs");
            Sum::input_check(inputs);
            Sum::output_check(outputs);
            let input_id = inputs[0];
            let input_shape = variable_table.get(input_id).expect("Invalid variable id").shape().clone();
            let output_id = outputs[0];
            let mut output_grad_id = variable_table.get_variable_grad_id(output_id).expect("Output grad id not found");

            if !keepdims && input_shape.len() != 0 {
                let mut axis = match axis {
                    Some(axis) => axis.clone(),
                    None => (0..input_shape.len()).collect::<Vec<usize>>(),
                };
                axis.sort();
                let mut output_shape = variable_table.get(output_id).expect("Invalid variable id").shape().clone();
                for axis in axis.iter() {
                    output_shape.insert(*axis, 1);
                }
                let reshape_id = function_table.generate_function_from_function_contents(Box::new(Reshape::new(output_shape)));
                output_grad_id = function_table.forward(reshape_id, vec![output_grad_id], variable_table, false)[0];
            }

            let broadcast_to_id = function_table.generate_function_from_function_contents(Box::new(BroadcastTo::new(input_shape)));
            let grad_id = function_table.forward(broadcast_to_id, vec![output_grad_id], variable_table, false)[0];

            variable_table.update_grad(input_id, grad_id, function_table);

            vec![input_id]
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

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let sum_id = function_table.generate_function_from_function_contents(Box::new(Sum::new(Some([0]), false)));
        let x_id = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data.clone(), vec![2, 3]), "x");
        let y_id = function_table.forward(sum_id, vec![x_id], &mut variable_table, false);

        let y = variable_table.get_variable_contents_f64(y_id[0]).unwrap();
        assert_eq!(y, &Tensor::new_from_num_vec(vec![5.0, 7.0, 9.0], vec![3]));
    }

    #[test]
    fn backward_normal() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let sum_id = function_table.generate_function_from_function_contents(Box::new(Sum::new(Some([0]), false)));
        let x_id = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data.clone(), vec![2, 3]), "x");
        let y_id = function_table.forward(sum_id, vec![x_id], &mut variable_table, false);

        variable_table.backward(y_id, &mut function_table, false);

        let x_grad = variable_table.get_variable_grad_contents_f64(x_id).unwrap();
        assert_eq!(x_grad, &Tensor::new_from_num_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![2, 3]));
    }

    #[test]
    fn backward_reshape() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let sum_id = function_table.generate_function_from_function_contents(Box::new(Sum::new::<&[usize]>(None, true)));
        let x_id = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data.clone(), vec![2, 3]), "x");
        let y_id = function_table.forward(sum_id, vec![x_id], &mut variable_table, false);

        variable_table.backward(y_id, &mut function_table, false);

        let x_grad = variable_table.get_variable_grad_contents_f64(x_id).unwrap();
        assert_eq!(x_grad, &Tensor::new_from_num_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![2, 3]));
    }

    #[test]
    #[should_panic]
    fn backward_backward_normal() {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let sum_id = function_table.generate_function_from_function_contents(Box::new(Sum::new(Some([0]), false)));
        let x_id = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(data.clone(), vec![2, 3]), "x");
        let y_id = function_table.forward(sum_id, vec![x_id], &mut variable_table, false);

        variable_table.backward(y_id, &mut function_table, false);

        let x_grad_id = variable_table.get_variable_grad_id(x_id).unwrap();
        variable_table.clear_grad(x_id);

        variable_table.backward(vec![x_grad_id], &mut function_table, false);

        let _ = variable_table.get_variable_grad_contents_f64(x_id).unwrap();
    }
}
