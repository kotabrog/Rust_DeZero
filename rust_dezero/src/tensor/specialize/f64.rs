use super::super::Tensor;

impl Tensor<f64> {
    /// Returns the result of performing an integer power over the value of each element
    pub fn powi(&self, n: i32) -> Self {
        Self {
            data: self.data
                .iter()
                .map(|x| x.powi(n))
                .collect(),
            shape: self.shape.clone(),
        }
    }

    /// Returns the result of performing a floating point power over the value of each element
    pub fn powf(&self, n: f64) -> Self {
        Self {
            data: self.data
                .iter()
                .map(|x| x.powf(n))
                .collect(),
            shape: self.shape.clone(),
        }
    }

    /// Returns the exponential of each element
    pub fn exp(&self) -> Self {
        Self {
            data: self.data
                .iter()
                .map(|x| x.exp())
                .collect(),
            shape: self.shape.clone(),
        }
    }

    /// Returns the sin of each element
    pub fn sin(&self) -> Self {
        Self {
            data: self.data
                .iter()
                .map(|x| x.sin())
                .collect(),
            shape: self.shape.clone(),
        }
    }

    /// Returns the cos of each element
    pub fn cos(&self) -> Self {
        Self {
            data: self.data
                .iter()
                .map(|x| x.cos())
                .collect(),
            shape: self.shape.clone(),
        }
    }

    /// Returns the tanh of each element
    pub fn tanh(&self) -> Self {
        Self {
            data: self.data
                .iter()
                .map(|x| x.tanh())
                .collect(),
            shape: self.shape.clone(),
        }
    }

    /// Returns a tensor with a value of 1 and the same shape as tensor
    pub fn ones_like(tensor: &Self) -> Self {
        Self {
            data: vec![1.0.into(); tensor.data.len()],
            shape: tensor.shape.clone(),
        }
    }

    /// Returns a tensor with a value and shape from the arguments
    pub fn full(value: f64, shape: Vec<usize>) -> Self {
        Self {
            data: vec![value.into(); shape.iter().product()],
            shape,
        }
    }

    /// Returns a tensor with a value from the argument and the same shape as tensor
    /// 
    /// # Arguments
    /// 
    /// * `tensor` - The tensor to be used as a reference for the shape
    /// * `value` - The value to be used for the tensor
    pub fn full_like(tensor: &Self, value: f64) -> Self {
        Self {
            data: vec![value.into(); tensor.data.len()],
            shape: tensor.shape.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powi_normal() {
        let x = Tensor::<f64>::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(x.powi(2), Tensor::<f64>::new_from_num_vec(vec![1.0, 4.0, 9.0], vec![3]));
    }

    #[test]
    fn powf_normal() {
        let x = Tensor::<f64>::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(x.powf(2.0), Tensor::<f64>::new_from_num_vec(vec![1.0, 4.0, 9.0], vec![3]));
    }

    #[test]
    fn exp_normal() {
        let data = vec![1.0, 2.0, 3.0];
        let x = Tensor::<f64>::new_from_num_vec(data.clone(), vec![3]);
        assert_eq!(x.exp(), Tensor::<f64>::new_from_num_vec(data.iter().map(|x| x.exp()), vec![3]));
    }

    #[test]
    fn sin_normal() {
        let data = vec![1.0, 2.0, 3.0];
        let x = Tensor::<f64>::new_from_num_vec(data.clone(), vec![3]);
        assert_eq!(x.sin(), Tensor::<f64>::new_from_num_vec(data.iter().map(|x| x.sin()), vec![3]));
    }

    #[test]
    fn cos_normal() {
        let data = vec![1.0, 2.0, 3.0];
        let x = Tensor::<f64>::new_from_num_vec(data.clone(), vec![3]);
        assert_eq!(x.cos(), Tensor::<f64>::new_from_num_vec(data.iter().map(|x| x.cos()), vec![3]));
    }

    #[test]
    fn tanh_normal() {
        let data = vec![1.0, 2.0, 3.0];
        let x = Tensor::<f64>::new_from_num_vec(data.clone(), vec![3]);
        assert_eq!(x.tanh(), Tensor::<f64>::new_from_num_vec(data.iter().map(|x| x.tanh()), vec![3]));
    }

    #[test]
    fn ones_like_normal() {
        let x = Tensor::<f64>::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(Tensor::ones_like(&x), Tensor::<f64>::new_from_num_vec(vec![1.0, 1.0, 1.0], vec![3]));
    }

    #[test]
    fn full_normal() {
        assert_eq!(Tensor::<f64>::full(1.0, vec![3]), Tensor::<f64>::new_from_num_vec(vec![1.0, 1.0, 1.0], vec![3]));
    }

    #[test]
    fn full_like_normal() {
        let x = Tensor::<f64>::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(Tensor::full_like(&x, 1.0), Tensor::<f64>::new_from_num_vec(vec![1.0, 1.0, 1.0], vec![3]));
    }
}
