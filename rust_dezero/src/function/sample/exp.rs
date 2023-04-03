use super::super::Function;
use crate::{Tensor, Variable};

pub struct Exp<T> {
    input: Option<Variable<T>>,
}

/// Exponential function
impl<T> Exp<T> {
    pub fn new() -> Self {
        Self { input: None }
    }
}

impl Function<f64> for Exp<f64> {
    fn get_input(&self) -> Option<&Variable<f64>> {
        self.input.as_ref()
    }

    fn set_input(&mut self, input: &Variable<f64>) {
        self.input = Some(input.clone());
    }

    fn forward(&self, input: &Tensor<f64>) -> Tensor<f64> {
        input.exp()
    }

    fn backward(&self, grad: &Tensor<f64>) -> Tensor<f64> {
        let input = self.get_input()
            .expect("input is None")
            .data();
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
