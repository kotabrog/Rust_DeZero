pub mod sample;

use crate::{Variable, Tensor};

pub trait Function<T> {
    fn call(&self, input: Variable<T>) -> Variable<T> {
        let x = input.data();
        let y = self.forward(x);
        Variable::new(y)
    }

    fn forward(&self, input: &Tensor<T>) -> Tensor<T>;
}
