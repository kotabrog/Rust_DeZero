pub mod scalar;

pub use self::scalar::Scaler;

/// Struct to perform calculations
/// 
/// # Fields
/// 
/// * `data` - Contents of Tensor
/// * `shape` - Tensor shape
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T>
{
    data: Vec<Scaler<T>>,
    shape: Vec<usize>,
}

impl<T> Tensor<T>
{
    /// Check the shape
    /// 
    /// # Arguments
    /// 
    /// * `data` - Contents of Tensor
    /// * `shape` - Tensor shape
    /// 
    /// # Panics
    /// 
    /// Panics if the shape is not correct.
    fn check_shape(data: &Vec<Scaler<T>>, shape: &Vec<usize>) {
        if shape.len() == 0 {
            assert_eq!(data.len(), 1, "Shape mismatch");
            return;
        }
        let mut size = 1;
        for s in shape {
            size *= s;
        }
        assert_eq!(data.len(), size, "Shape mismatch");
    }

    /// Get the data
    pub fn data(&self) -> &Vec<Scaler<T>> {
        &self.data
    }

    /// Get the shape
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the data type
    pub fn data_type(&self) -> &str {
        std::any::type_name::<T>()
    }
}

impl<T> Tensor<T>
where
    T: Clone
{
    /// Create a new Tensor
    /// 
    /// # Arguments
    /// 
    /// * `data` - Contents of Tensor
    /// * `shape` - Tensor shape
    /// 
    /// # Panics
    /// 
    /// Panics if the shape is not correct.
    pub fn new<U: AsRef<[Scaler<T>]>, V: AsRef<[usize]>>(data: U, shape: V) -> Self {
        let data = data.as_ref().to_vec();
        let shape = shape.as_ref().to_owned();
        Self::check_shape(&data, &shape);
        Self { data, shape }
    }

    /// Create a new Tensor from numbers
    /// 
    /// # Arguments
    /// 
    /// * `data` - Contents of Tensor
    /// * `shape` - Tensor shape
    /// 
    /// # Panics
    /// 
    /// Panics if the shape is not correct.
    pub fn new_from_num_vec<U: IntoIterator<Item = T>, V: AsRef<[usize]>>(data: U, shape: V) -> Self {
        let data: Vec<Scaler<T>> = data.into_iter().map(Scaler::from).collect();
        Tensor::new(data, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_shape_normal() {
        Tensor::check_shape(
            &vec![1.0.into(), 2.0.into(), 3.0.into(), 4.0.into(), 5.0.into(), 6.0.into()],
            &vec![2, 3]
        );
    }

    #[test]
    #[should_panic]
    fn check_shape_error_mismatch() {
        Tensor::check_shape(
            &vec![1.0.into(), 2.0.into(), 3.0.into(), 4.0.into(), 5.0.into(), 6.0.into()],
            &vec![2, 2]
        );
    }

    #[test]
    fn check_shape_zero_dim() {
        Tensor::<f32>::check_shape(&vec![1.0.into()], &vec![]);
    }

    #[test]
    fn check_shape_empty() {
        Tensor::<f32>::check_shape(&vec![], &vec![1, 0, 2]);
    }

    #[test]
    #[should_panic]
    fn check_shape_error_empty_mismatch() {
        Tensor::<f32>::check_shape(&vec![], &vec![]);
    }

    #[test]
    fn new_normal() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        assert_eq!(x.data(), &vec![0.0.into(), 1.0.into(), 2.0.into()]);
        assert_eq!(x.shape(), &vec![3]);
    }

    #[test]
    fn new_zero_dim() {
        let x = Tensor::new([1.0.into()], []);
        assert_eq!(x.data(), &vec![1.0.into()]);
        assert_eq!(x.shape(), &vec![]);
    }

    #[test]
    #[should_panic]
    fn new_error_mismatch_shape() {
        let _ = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [2,]);
    }

    #[test]
    fn new_from_num_vec_normal() {
        let x = Tensor::new_from_num_vec([0.0, 1.0, 2.0], [3,]);
        assert_eq!(x.data(), &vec![0.0.into(), 1.0.into(), 2.0.into()]);
        assert_eq!(x.shape(), &vec![3]);
    }

    #[test]
    fn new_from_num_vec_zero_dim() {
        let x = Tensor::new_from_num_vec([1.0], []);
        assert_eq!(x.data(), &vec![1.0.into()]);
        assert_eq!(x.shape(), &vec![]);
    }

    #[test]
    #[should_panic]
    fn new_from_num_vec_error_mismatch_shape() {
        let _ = Tensor::new_from_num_vec([0.0, 1.0, 2.0], [2,]);
    }

    #[test]
    fn ndim_normal() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        assert_eq!(x.ndim(), 1);
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [1, 3]);
        assert_eq!(x.ndim(), 2);
        let x = Tensor::new([1.0.into()], []);
        assert_eq!(x.ndim(), 0);
    }
}
