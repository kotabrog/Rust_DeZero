use super::super::Function;
use crate::Tensor;
pub struct Square {}

impl Square {
    pub fn new() -> Self {
        Self {}
    }
}

impl Function<f32> for Square {
    fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        input.powi(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Variable;

    #[test]
    fn square_normal() {
        let x = Variable::new(Tensor::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]));
        let y = Square::new().call(x);
        assert_eq!(*y.data(), Tensor::new_from_num_vec(vec![1.0, 4.0, 9.0], vec![3]));
    }
}
