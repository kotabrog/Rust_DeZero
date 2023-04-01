use crate::Tensor;

/// Variable
/// 
/// # Fields
/// 
/// * `data` - Contents of Tensor
#[derive(Debug, Clone)]
pub struct Variable<T>
{
    data: Tensor<T>,
}

impl<T> Variable<T>
{
    /// Create a new Variable
    /// 
    /// # Arguments
    /// 
    /// * `data` - Contents of Tensor
    pub fn new(data: Tensor<T>) -> Self {
        Self { data }
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
    /// * `data` - Contents of Tensor
    pub fn set_data(&mut self, data: Tensor<T>) {
        self.data = data;
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
}
