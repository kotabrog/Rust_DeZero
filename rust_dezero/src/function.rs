pub mod sample;

use std::cell::{RefCell, Ref};
use std::rc::Rc;
use crate::{Variable, Tensor};

#[derive(Debug, Clone)]
pub struct FunctionInternal<T> {
    input: Option<Rc<RefCell<Variable<T>>>>,
    output: Option<Rc<RefCell<Variable<T>>>>,
}

impl<T> FunctionInternal<T> {
    pub fn new() -> Self {
        Self { input: None, output: None }
    }

    /// Get the input
    pub fn get_input(&self) -> Option<Ref<'_, Variable<T>>> {
        self.input.as_ref().map(|x| x.borrow())
    }

    /// Get the input as Rc<RefCell<Variable<T>>>
    pub fn get_input_as_rc(&self) -> Option<&Rc<RefCell<Variable<T>>>> {
        self.input.as_ref()
    }

    /// Set the input
    /// 
    /// # Arguments
    /// 
    /// * `input` - Input Variable
    pub fn set_input(&mut self, input: Rc<RefCell<Variable<T>>>) {
        self.input = Some(input);
    }

    /// Get the output
    pub fn get_output(&self) -> Option<Ref<'_, Variable<T>>> {
        self.output.as_ref().map(|x| x.borrow())
    }

    /// Get the input as Rc<RefCell<Variable<T>>>
    pub fn get_output_as_rc(&self) -> Option<&Rc<RefCell<Variable<T>>>> {
        self.output.as_ref()
    }

    /// Set the output
    /// 
    /// # Arguments
    /// 
    /// * `output` - Output Variable
    pub fn set_output(&mut self, output: Rc<RefCell<Variable<T>>>) {
        self.output = Some(output);
    }
}


pub trait Function<T>: std::fmt::Debug {
    fn call_mut(&mut self, input: Rc<RefCell<Variable<T>>>) -> Rc<RefCell<Variable<T>>> {
        let y = {
            let input_borrowed = input.borrow();
            let x = input_borrowed.data();
            self.forward(x)
        };
        let output = Variable::new(y);
        let output = Rc::new(RefCell::new(output));
        self.set_input(input);
        self.set_output(output.clone());
        output
    }

    fn get_input(&self) -> Option<Ref<'_, Variable<T>>> {
        self.get_internal().get_input()
    }

    fn set_input(&mut self, input: Rc<RefCell<Variable<T>>>) {
        self.get_internal_mut().set_input(input);
    }

    fn get_output(&self) -> Option<Ref<'_, Variable<T>>> {
        self.get_internal().get_output()
    }

    fn set_output(&mut self, output: Rc<RefCell<Variable<T>>>) {
        self.get_internal_mut().set_output(output);
    }

    fn get_internal(&self) -> &FunctionInternal<T>;
    fn get_internal_mut(&mut self) -> &mut FunctionInternal<T>;

    fn forward(&self, input: &Tensor<T>) -> Tensor<T>;
    fn backward(&self, grad: &Tensor<T>) -> Tensor<T>;
}
