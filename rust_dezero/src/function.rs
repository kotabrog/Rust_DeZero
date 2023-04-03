pub mod sample;

use crate::{Variable, Tensor};

pub trait Function<T> {
    fn call_mut(&mut self, input: &Variable<T>) -> Variable<T> {
        let x = input.data();
        let y = self.forward(x);
        let output = Variable::new(y);
        self.set_input(input);
        output
    }

    fn get_input(&self) -> Option<&Variable<T>>;
    fn set_input(&mut self, input: &Variable<T>);

    fn forward(&self, input: &Tensor<T>) -> Tensor<T>;
    fn backward(&self, grad: &Tensor<T>) -> Tensor<T>;
}
