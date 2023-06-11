extern crate rand;

use rand::Rng;
use rand::distributions::{Distribution, Standard};

use super::{Tensor, Scaler};

/// Random number generator for Tensor.
/// 
/// # Fields
/// 
/// * `rng` - Random number generator.
pub struct TensorRng {
    rng: rand::rngs::ThreadRng,
}

impl TensorRng {
    /// Create a new TensorRng.
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }

    /// Generate a random Tensor.
    /// 
    /// # Arguments
    /// 
    /// * `shape` - Shape of the Tensor.
    pub fn gen<T, U>(&mut self, shape: U) -> Tensor<T>
    where
        Standard: Distribution<T>,
        T: Clone,
        U: AsRef<[usize]>
    {
        let mut data = Vec::new();
        for _ in 0..shape.as_ref().iter().product::<usize>() {
            data.push(Scaler::new(self.rng.gen::<T>()));
        }
        Tensor::new(data, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gen_normal() {
        let mut rng = TensorRng::new();
        let x = rng.gen::<f32, _>([2, 3]);
        assert_eq!(x.data_type(), "f32");
        assert_eq!(x.shape(), &[2, 3]);
    }
}
