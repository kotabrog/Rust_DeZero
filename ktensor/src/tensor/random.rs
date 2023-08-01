extern crate rand;

use rand::Rng;
use rand::distributions::{Distribution, Standard};
use super::Tensor;

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
        T: Clone + Default,
        U: Into<Vec<usize>>
    {
        let shape = shape.into();
        let size: usize = shape.iter().product();
        let mut data = vec![T::default(); size];
        for i in 0..size {
            data[i] = self.rng.gen::<T>();
        }
        Tensor { data, shape: shape }
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
        assert_eq!(x.get_shape(), &[2, 3]);
    }
}
