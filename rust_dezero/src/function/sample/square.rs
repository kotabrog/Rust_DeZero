use crate::variable::{Variable, VariableTable, VariableType, VariableWrapper};
use super::super::{Function, FunctionInfo};

#[derive(Debug, Clone)]
pub struct Square {}

/// Square function
impl Square {
    pub fn new() -> Self {
        Self { }
    }
}

impl Function for Square {
    fn forward(&self, _info: &FunctionInfo, input: &usize, variables: &mut VariableTable) -> usize {
        let input = variables.get_variable_type(*input).expect("input is None");
        let input = match input {
            VariableType::F64(x) => x.data(),
        };

        let output = input.powi(2);

        let output = VariableWrapper::from_variable_f64(Variable::new(output));
        variables.add(Box::new(output))
    }

    fn backward(&self, info: &FunctionInfo, grad_id: &usize, variables: &mut VariableTable) -> usize {
        let input = variables.get_variable_type(info.id).expect("input is None");
        let input = match input {
            VariableType::F64(x) => x.data(),
        };
        let grad = variables.get_variable_type(*grad_id).expect("grad is None");
        let grad = match grad {
            VariableType::F64(x) =>
                x.grad()
                .expect("grad is None"),
        };

        let gx = (input * grad).scalar_mul(2.0.into());

        let input = variables.get_variable_type_mut(info.id).expect("input is None");
        let input_variable = match input {
            VariableType::F64(x) => x,
        };
        input_variable.set_grad(gx);
        *grad_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::FunctionWrapper;
    use crate::Tensor;
    use crate::function::FunctionTable;

    #[test]
    fn square_forward() {
        let data = vec![1.0, 2.0, 3.0];
        let mut variables = VariableTable::new();
        let mut functions = FunctionTable::new();
        let x = Variable::<f64>::new(Tensor::new_from_num_vec(data.clone(), vec![3]));
        let x = VariableWrapper::from_variable_f64(x);
        let x_id = variables.add(Box::new(x));
        let f = Square::new();
        let f = FunctionWrapper::new(Box::new(f));
        let f_id = functions.add(f);
        let f = functions.get_mut(f_id).expect("f is None");
        let y_id = f.call_mut(x_id, &mut variables);
        let y = variables.get_variable_type(y_id).expect("y is None");
        let y = match y {
            VariableType::F64(x) => x,
        };
        assert_eq!(*y.data(), Tensor::new_from_num_vec(vec![1.0, 4.0, 9.0], vec![3]));
    }

    #[test]
    fn square_backward() {
        let data = vec![1.0, 2.0, 3.0];
        let mut variables = VariableTable::new();
        let mut functions = FunctionTable::new();
        let x = Variable::<f64>::new(Tensor::new_from_num_vec(data.clone(), vec![3]));
        let x = VariableWrapper::from_variable_f64(x);
        let x_id = variables.add(Box::new(x));
        let f = Square::new();
        let f = FunctionWrapper::new(Box::new(f));
        let f_id = functions.add(f);
        let f = functions.get_mut(f_id).expect("f is None");
        let y_id = f.call_mut(x_id, &mut variables);
        variables.backward(y_id, &mut functions);
        let x_grad = variables.get_variable_type(x_id).expect("x is None");
        let x_grad = match x_grad {
            VariableType::F64(x) => x.grad().unwrap(),
        };
        assert_eq!(x_grad, &Tensor::new_from_num_vec(vec![2.0, 4.0, 6.0], vec![3]));
    }
}
