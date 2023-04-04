use std::cell::{RefCell, Ref};
use std::rc::Rc;
use super::super::{Function, FunctionInternal};
use crate::{Tensor, Variable};

#[derive(Debug, Clone)]
pub struct Square<T> {
    internal: FunctionInternal<T>,
}

/// Square function
impl<T> Square<T> {
    pub fn new() -> Self {
        Self { internal: FunctionInternal::new() }
    }
}

impl Function<f64> for Square<f64> {
    fn get_internal(&self) -> &FunctionInternal<f64> {
        &self.internal
    }

    fn get_internal_mut(&mut self) -> &mut FunctionInternal<f64> {
        &mut self.internal
    }

    fn get_input(&self) -> Option<Ref<'_, Variable<f64>>> {
        self.internal.get_input()
    }

    fn set_input(&mut self, input: Rc<RefCell<Variable<f64>>>) {
        self.internal.set_input(input);
    }

    fn get_output(&self) -> Option<Ref<'_, Variable<f64>>> {
        self.internal.get_output()
    }

    fn set_output(&mut self, output: Rc<RefCell<Variable<f64>>>) {
        self.internal.set_output(output);
    }

    fn forward(&self, input: &Tensor<f64>) -> Tensor<f64> {
        input.powi(2)
    }

    fn backward(&self, grad: &Tensor<f64>) -> Tensor<f64> {
        let input_borrowed =
            self.get_input()
            .expect("input is None");
        let input = input_borrowed.data();
        let gx = (input * grad).scalar_mul(2.0.into());
        gx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Variable;

    #[test]
    fn square_forward() {
        let x = Variable::new(Tensor::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]));
        let y = Square::new().call_mut(&x);
        assert_eq!(*y.data(), Tensor::new_from_num_vec(vec![1.0, 4.0, 9.0], vec![3]));
    }

    #[test]
    fn square_backward() {
        let x = Variable::new(Tensor::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]));
        let mut f = Square::new();
        let y = f.call_mut(&x);
        let x_grad = f.backward(&Tensor::new_from_num_vec(vec![1.0, 1.0, 1.0], vec![3]));
        assert_eq!(*y.data(), Tensor::new_from_num_vec(vec![1.0, 4.0, 9.0], vec![3]));
        assert_eq!(x_grad, Tensor::new_from_num_vec(vec![2.0, 4.0, 6.0], vec![3]));
    }
}
