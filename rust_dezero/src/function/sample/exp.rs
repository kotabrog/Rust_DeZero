use std::cell::{RefCell, Ref};
use std::rc::Rc;
use super::super::{Function, FunctionInternal};
use crate::{Tensor, Variable};

#[derive(Debug, Clone)]
pub struct Exp<T> {
    internal: FunctionInternal<T>,
}

/// Exponential function
impl<T> Exp<T> {
    pub fn new() -> Self {
        Self { internal: FunctionInternal::new() }
    }
}

impl Function<f64> for Exp<f64> {
    fn get_internal(&self) -> &FunctionInternal<f64> {
        &self.internal
    }

    fn get_internal_mut(&mut self) -> &mut FunctionInternal<f64> {
        &mut self.internal
    }

    fn forward(&self, input: &Tensor<f64>) -> Tensor<f64> {
        input.exp()
    }

    fn backward(&self, grad: &Tensor<f64>) -> Tensor<f64> {
        let input_borrowed =
            self.get_input()
            .expect("input is None");
        let input = input_borrowed.data();
        let gx = &input.exp() * grad;
        gx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Variable;

    #[test]
    fn exp_forward() {
        let data = vec![1.0, 2.0, 3.0];
        let x = Variable::<f64>::new(Tensor::new_from_num_vec(data.clone(), vec![3]));
        let y = Exp::new().call_mut(&x);
        assert_eq!(*y.data(), Tensor::new_from_num_vec(data.iter().map(|x| x.exp()), vec![3]));
    }

    #[test]
    fn exp_backward() {
        let data = vec![1.0, 2.0, 3.0];
        let x = Variable::<f64>::new(Tensor::new_from_num_vec(data.clone(), vec![3]));
        let mut f = Exp::new();
        let y = f.call_mut(&x);
        let x_grad = f.backward(&Tensor::new_from_num_vec(vec![1.0, 1.0, 1.0], vec![3]));
        assert_eq!(*y.data(), Tensor::new_from_num_vec(data.iter().map(|x| x.exp()), vec![3]));
        assert_eq!(x_grad, Tensor::new_from_num_vec(data.iter().map(|x| x.exp()), vec![3]));
    }
}
