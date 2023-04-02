use super::super::Function;
use crate::Tensor;
pub struct Exp {}

/// Exponential function
impl Exp {
    pub fn new() -> Self {
        Self {}
    }
}

impl Function<f32> for Exp {
    fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        input.exp()
    }
}

impl Function<f64> for Exp {
    fn forward(&self, input: &Tensor<f64>) -> Tensor<f64> {
        input.exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Variable;

    #[test]
    fn exp_normal_f32() {
        let data = vec![1.0, 2.0, 3.0];
        let x = Variable::<f32>::new(Tensor::new_from_num_vec(data.clone(), vec![3]));
        let y = Exp::new().call(&x);
        assert_eq!(*y.data(), Tensor::new_from_num_vec(data.iter().map(|x| x.exp()), vec![3]));
    }

    #[test]
    fn exp_normal_f64() {
        let data = vec![1.0, 2.0, 3.0];
        let x = Variable::<f64>::new(Tensor::new_from_num_vec(data.clone(), vec![3]));
        let y = Exp::new().call(&x);
        assert_eq!(*y.data(), Tensor::new_from_num_vec(data.iter().map(|x| x.exp()), vec![3]));
    }
}
