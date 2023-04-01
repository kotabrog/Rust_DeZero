use super::super::Tensor;

impl Tensor<f32> {
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
    pub fn powf(&self, n: f32) -> Self {
        Self {
            data: self.data
                .iter()
                .map(|x| x.powf(n))
                .collect(),
            shape: self.shape.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powi_normal() {
        let x = Tensor::<f32>::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(x.powi(2), Tensor::<f32>::new_from_num_vec(vec![1.0, 4.0, 9.0], vec![3]));
    }

    #[test]
    fn powf_normal() {
        let x = Tensor::<f32>::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(x.powf(2.0), Tensor::<f32>::new_from_num_vec(vec![1.0, 4.0, 9.0], vec![3]));
    }
}
