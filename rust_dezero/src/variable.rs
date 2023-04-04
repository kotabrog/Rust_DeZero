use std::cell::RefCell;
use std::rc::Rc;
use crate::{Tensor, Function};

/// Variable
/// 
/// # Fields
/// 
/// * `data` - Contents of Variable
/// * `grad` - Gradient of Variable
#[derive(Debug, Clone)]
pub struct Variable<T>
{
    data: Tensor<T>,
    grad: Option<Tensor<T>>,
    creator: Option<Rc<RefCell<dyn Function<T>>>>,
}

impl<T> Variable<T>
{
    /// Create a new Variable
    /// 
    /// # Arguments
    /// 
    /// * `data` - Contents of Variable
    pub fn new(data: Tensor<T>) -> Self {
        Self { data, grad: None, creator: None }
    }

    /// Get the data
    pub fn data(&self) -> &Tensor<T> {
        &self.data
    }

    /// Get the shape
    pub fn shape(&self) -> &Vec<usize> {
        self.data.shape()
    }

    /// Get the data type
    pub fn data_type(&self) -> &str {
        self.data.data_type()
    }

    /// Set the data
    /// 
    /// # Arguments
    /// 
    /// * `data` - Contents of Variable
    pub fn set_data(&mut self, data: Tensor<T>) {
        self.data = data;
    }

    /// Get the grad
    pub fn grad(&self) -> Option<&Tensor<T>> {
        self.grad.as_ref()
    }

    /// Get the grad shape
    pub fn grad_shape(&self) -> Option<&Vec<usize>> {
        match &self.grad {
            Some(grad) => Some(grad.shape()),
            None => None,
        }
    }

    /// Set the grad
    /// 
    /// # Arguments
    /// 
    /// * `grad` - Gradient of Variable
    pub fn set_grad(&mut self, grad: Tensor<T>) {
        self.grad = Some(grad);
    }

    /// Get the creator
    pub fn creator(&self) -> Option<&Rc<RefCell<dyn Function<T>>>> {
        self.creator.as_ref()
    }

    /// Set the creator
    /// 
    /// # Arguments
    /// 
    /// * `creator` - Creator of Variable
    pub fn set_creator(&mut self, creator: Rc<RefCell<dyn Function<T>>>) {
        self.creator = Some(creator);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_normal() {
        let tensor = Tensor::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let x = Variable::<f32>::new(tensor.clone());
        assert_eq!(*x.data(), tensor);
        assert_eq!(*x.shape(), vec![3]);
        assert_eq!(x.data_type(), "f32");
    }

    #[test]
    fn set_data_normal() {
        let tensor = Tensor::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let mut x = Variable::<f32>::new(tensor.clone());
        let tensor = Tensor::new_from_num_vec(vec![4.0, 5.0, 6.0], vec![3]);
        x.set_data(tensor.clone());
        assert_eq!(*x.data(), tensor);
        assert_eq!(*x.shape(), vec![3]);
        assert_eq!(x.data_type(), "f32");
    }

    #[test]
    fn set_grad_normal() {
        let tensor = Tensor::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let mut x = Variable::<f32>::new(tensor);
        let tensor = Tensor::new_from_num_vec(vec![4.0, 5.0, 6.0], vec![3]);
        x.set_grad(tensor.clone());
        assert_eq!(*x.grad().unwrap(), tensor);
        assert_eq!(*x.grad_shape().unwrap(), vec![3]);
    }
}
