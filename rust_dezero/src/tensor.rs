mod scaler;
mod specialize;

use crate::num::FromUsize;

pub use self::scaler::Scaler;

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
        let size = shape.iter().product();
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

    /// Check if the Tensor is a scalar
    pub fn is_scalar(&self) -> bool {
        self.ndim() == 0
    }

    fn calc_at_index(&self, indexes: &[usize]) -> usize {
        if self.is_scalar() {
            assert!(indexes.len() == 0, "Shape mismatch");
            return 0;
        }
        assert_eq!(indexes.len(), self.ndim(), "Shape mismatch");
        for i in 0..self.ndim() {
            assert!(indexes[i] < self.shape[i], "Index out of range");
        }
        let mut index = 0;
        let mut size = 1;
        for i in (0..self.ndim()).rev() {
            index += indexes[i] * size;
            size *= self.shape[i];
        }
        index
    }

    /// Get the value of the Tensor
    /// 
    /// # Arguments
    /// 
    /// * `indexes` - Indexes of the Tensor
    /// 
    /// # Panics
    /// 
    /// Panics if the shape is not correct.
    pub fn at(&self, indexes: &[usize]) -> &Scaler<T> {
        let index = self.calc_at_index(indexes);
        &self.data[index]
    }

    /// Get the value of a Tensor that can be changed
    /// 
    /// # Arguments
    /// 
    /// * `indexes` - Indexes of the Tensor
    /// 
    /// # Panics
    /// 
    /// Panics if the shape is not correct.
    pub fn at_mut(&mut self, indexes: &[usize]) -> &mut Scaler<T> {
        let index = self.calc_at_index(indexes);
        &mut self.data[index]
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

    /// Reshape the Tensor
    /// 
    /// # Arguments
    /// 
    /// * `shape` - Tensor shape
    /// 
    /// # Panics
    /// 
    /// Panics if the shape is not correct.
    pub fn reshape<U: AsRef<[usize]>>(&self, shape: U) -> Self {
        let shape = shape.as_ref().to_owned();
        Self::check_shape(&self.data, &shape);
        Self { data: self.data.clone(), shape }
    }

    /// Transpose the Tensor
    pub fn transpose(&self) -> Self {
        let mut shape = self.shape.clone();
        shape.reverse();
        Self::new(self.data.clone(), shape)
    }
}

impl<T> Tensor<T>
where
    T: FromUsize,
{
    /// Create a tensor with values in order from 0 in the given shape.
    /// 
    /// # Arguments
    /// 
    /// * `shape` - Tensor shape
    /// 
    /// # Panics
    /// 
    /// panic when the conversion from usize to T is not possible.
    pub fn arrange<U: AsRef<[usize]>>(shape: U) -> Self {
        let shape = shape.as_ref().to_owned();
        let size = shape.iter().product();
        let data = (0..size)
            .map(|x| Scaler::from(T::from_usize(x)))
            .collect();
        Self { data, shape }
    }
}

impl<T> std::ops::Add for Tensor<T>
where
    T: std::ops::Add<Output = T> + Clone
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        let data =
            self.data
            .into_iter()
            .zip(other.data.into_iter()).map(|(x, y)| x + y)
            .collect();
        Self { data, shape: self.shape }
    }
}

impl<T> std::ops::Add for &Tensor<T>
where
    T: std::ops::Add<Output = T> + Copy
{
    type Output = Tensor<T>;

    fn add(self, other: Self) -> Self::Output {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        let data =
            self.data
            .iter()
            .zip(other.data.iter()).map(|(x, y)| *x + *y)
            .collect();
        Tensor { data, shape: self.shape.clone() }
    }
}

impl<T> std::ops::Sub for Tensor<T>
where
    T: std::ops::Sub<Output = T> + Clone
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        let data =
            self.data
            .into_iter()
            .zip(other.data.into_iter()).map(|(x, y)| x - y)
            .collect();
        Self { data, shape: self.shape }
    }
}

impl<T> std::ops::Sub for &Tensor<T>
where
    T: std::ops::Sub<Output = T> + Copy
{
    type Output = Tensor<T>;

    fn sub(self, other: Self) -> Self::Output {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        let data =
            self.data
            .iter()
            .zip(other.data.iter()).map(|(x, y)| *x - *y)
            .collect();
        Tensor { data, shape: self.shape.clone() }
    }
}

impl<T> std::ops::Mul for Tensor<T>
where
    T: std::ops::Mul<Output = T> + Clone
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        let data =
            self.data
            .into_iter()
            .zip(other.data.into_iter()).map(|(x, y)| x * y)
            .collect();
        Self { data, shape: self.shape }
    }
}

impl<T> std::ops::Mul for &Tensor<T>
where
    T: std::ops::Mul<Output = T> + Copy
{
    type Output = Tensor<T>;

    fn mul(self, other: Self) -> Self::Output {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        let data =
            self.data
            .iter()
            .zip(other.data.iter()).map(|(x, y)| *x * *y)
            .collect();
        Tensor { data, shape: self.shape.clone() }
    }
}

impl<T> std::ops::Div for Tensor<T>
where
    T: std::ops::Div<Output = T> + Clone
{
    type Output = Self;

    fn div(self, other: Self) -> Self {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        let data =
            self.data
            .into_iter()
            .zip(other.data.into_iter()).map(|(x, y)| x / y)
            .collect();
        Self { data, shape: self.shape }
    }
}

impl<T> std::ops::Div for &Tensor<T>
where
    T: std::ops::Div<Output = T> + Copy
{
    type Output = Tensor<T>;

    fn div(self, other: Self) -> Self::Output {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        let data =
            self.data
            .iter()
            .zip(other.data.iter()).map(|(x, y)| *x / *y)
            .collect();
        Tensor { data, shape: self.shape.clone() }
    }
}

impl<T> std::ops::Neg for Tensor<T>
where
    T: std::ops::Neg<Output = T> + Clone
{
    type Output = Self;

    fn neg(self) -> Self {
        let data = self.data.into_iter().map(|x| -x).collect();
        Self { data, shape: self.shape }
    }
}

impl<T> std::ops::Neg for &Tensor<T>
where
    T: std::ops::Neg<Output = T> + Copy
{
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        let data = self.data.iter().map(|x| -*x).collect();
        Tensor { data, shape: self.shape.clone() }
    }
}

impl<T> std::ops::AddAssign<&Tensor<T>> for Tensor<T>
where
    T: std::ops::AddAssign + Copy
{
    fn add_assign(&mut self, other: &Self) {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        for (x, y) in self.data.iter_mut().zip(other.data.iter()) {
            *x += *y;
        }
    }
}

impl<T> std::ops::SubAssign<&Tensor<T>> for Tensor<T>
where
    T: std::ops::SubAssign + Copy
{
    fn sub_assign(&mut self, other: &Self) {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        for (x, y) in self.data.iter_mut().zip(other.data.iter()) {
            *x -= *y;
        }
    }
}

impl<T> std::ops::MulAssign<&Tensor<T>> for Tensor<T>
where
    T: std::ops::MulAssign + Copy
{
    fn mul_assign(&mut self, other: &Self) {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        for (x, y) in self.data.iter_mut().zip(other.data.iter()) {
            *x *= *y;
        }
    }
}

impl<T> std::ops::DivAssign<&Tensor<T>> for Tensor<T>
where
    T: std::ops::DivAssign + Copy
{
    fn div_assign(&mut self, other: &Tensor<T>) {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        for (x, y) in self.data.iter_mut().zip(other.data.iter()) {
            *x /= *y;
        }
    }
}

impl<T> Tensor<T>
where
    T: std::ops::Add<Output = T> + Copy
{
    /// Add a Tensor to a scalar
    /// 
    /// # Arguments
    /// 
    /// * `scalar` - Scalar to add
    pub fn scalar_add(&self, scalar: Scaler<T>) -> Self {
        let data = self.data.iter().map(|x| *x + scalar).collect();
        Self { data, shape: self.shape.clone() }
    }
}

impl<T> Tensor<T>
where
    T: std::ops::Sub<Output = T> + Copy
{
    /// Subtract a Tensor from a scalar
    /// 
    /// # Arguments
    /// 
    /// * `scalar` - Scalar to subtract
    pub fn scalar_sub(&self, scalar: Scaler<T>) -> Self {
        let data = self.data.iter().map(|x| *x - scalar).collect();
        Self { data, shape: self.shape.clone() }
    }
}

impl<T> Tensor<T>
where
    T: std::ops::Mul<Output = T> + Copy
{
    /// Multiply a Tensor by a scalar
    /// 
    /// # Arguments
    /// 
    /// * `scalar` - Scalar to multiply by
    pub fn scalar_mul(&self, scalar: Scaler<T>) -> Self {
        let data = self.data.iter().map(|x| *x * scalar).collect();
        Self { data, shape: self.shape.clone() }
    }
}

impl<T> Tensor<T>
where
    T: std::ops::Div<Output = T> + Copy
{
    /// Divide a Tensor by a scalar
    /// 
    /// # Arguments
    /// 
    /// * `scalar` - Scalar to divide by
    pub fn scalar_div(&self, scalar: Scaler<T>) -> Self {
        let data = self.data.iter().map(|x| *x / scalar).collect();
        Self { data, shape: self.shape.clone() }
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
    fn reshape_normal() {
        let x = Tensor::new_from_num_vec([0.0, 1.0, 2.0, 3.0], [4,]);
        let x = x.reshape([2, 2]);
        assert_eq!(x.data(), &vec![0.0.into(), 1.0.into(), 2.0.into(), 3.0.into()]);
        assert_eq!(x.shape(), &vec![2, 2]);
    }

    #[test]
    fn reshape_zero_dim() {
        let x = Tensor::new_from_num_vec([1.0], []);
        let x = x.reshape([1, 1]);
        assert_eq!(x.data(), &vec![1.0.into()]);
        assert_eq!(x.shape(), &vec![1, 1]);
    }

    #[test]
    #[should_panic]
    fn reshape_error_mismatch_shape() {
        let x = Tensor::new_from_num_vec([0.0, 1.0, 2.0, 3.0], [4,]);
        let _ = x.reshape([2, 3]);
    }

    #[test]
    fn transpose_normal() {
        let x = Tensor::new_from_num_vec([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [3, 2]);
        let x = x.transpose();
        assert_eq!(x.data(), &vec![0.0.into(), 1.0.into(), 2.0.into(), 3.0.into(), 4.0.into(), 5.0.into()]);
        assert_eq!(x.shape(), &vec![2, 3]);
    }

    #[test]
    fn transpose_zero_dim() {
        let x = Tensor::new_from_num_vec([1.0], []);
        let x = x.transpose();
        assert_eq!(x.data(), &vec![1.0.into()]);
        assert_eq!(x.shape(), &vec![]);
    }

    #[test]
    fn transpose_1d() {
        let x = Tensor::new_from_num_vec([0.0, 1.0, 2.0], [3,]);
        let x = x.transpose();
        assert_eq!(x.data(), &vec![0.0.into(), 1.0.into(), 2.0.into()]);
        assert_eq!(x.shape(), &vec![3]);
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

    #[test]
    fn is_scalar_normal() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        assert_eq!(x.is_scalar(), false);
        let x = Tensor::new([1.0.into()], []);
        assert_eq!(x.is_scalar(), true);
    }

    #[test]
    fn at_normal() {
        let x = Tensor::<f32>::arrange([2, 3]);
        assert_eq!(x.at(&[0, 0]), &0.0.into());
        assert_eq!(x.at(&[0, 1]), &1.0.into());
        assert_eq!(x.at(&[0, 2]), &2.0.into());
        assert_eq!(x.at(&[1, 0]), &3.0.into());
        assert_eq!(x.at(&[1, 1]), &4.0.into());
        assert_eq!(x.at(&[1, 2]), &5.0.into());
    }

    #[test]
    fn at_scaler() {
        let x = Tensor::<f32>::new([0.0.into()], []);
        assert_eq!(x.at(&[]), &0.0.into());
    }

    #[test]
    #[should_panic]
    fn at_error_out_of_range() {
        let x = Tensor::<f32>::arrange([2, 3]);
        let _ = x.at(&[2, 0]);
    }

    #[test]
    #[should_panic]
    fn at_error_mismatch_ndim() {
        let x = Tensor::<f32>::arrange([2, 3]);
        let _ = x.at(&[0]);
    }

    #[test]
    fn at_mut_normal() {
        let mut x = Tensor::<f32>::arrange([2, 3]);
        *x.at_mut(&[0, 0]) = 10.0.into();
        *x.at_mut(&[0, 1]) = 11.0.into();
        *x.at_mut(&[0, 2]) = 12.0.into();
        *x.at_mut(&[1, 0]) = 13.0.into();
        *x.at_mut(&[1, 1]) = 14.0.into();
        *x.at_mut(&[1, 2]) = 15.0.into();
        assert_eq!(x.data(), &vec![10.0.into(), 11.0.into(), 12.0.into(), 13.0.into(), 14.0.into(), 15.0.into()]);
        assert_eq!(x.shape(), &vec![2, 3]);
    }

    #[test]
    fn at_mut_scaler() {
        let mut x = Tensor::<f32>::new([0.0.into()], []);
        assert_eq!(x.at_mut(&[]), &0.0.into());
    }

    #[test]
    #[should_panic]
    fn at_mut_error_out_of_range() {
        let mut x = Tensor::<f32>::arrange([2, 3]);
        let _ = x.at_mut(&[2, 0]);
    }

    #[test]
    #[should_panic]
    fn at_mut_error_mismatch_ndim() {
        let mut x = Tensor::<f32>::arrange([2, 3]);
        let _ = x.at_mut(&[0]);
    }

    #[test]
    fn arrange_normal() {
        let x = Tensor::<f32>::arrange([2, 3]);
        assert_eq!(x.data(), &vec![0.0.into(), 1.0.into(), 2.0.into(), 3.0.into(), 4.0.into(), 5.0.into()]);
        assert_eq!(x.shape(), &vec![2, 3]);
    }

    #[test]
    fn arrange_zero_dim() {
        let x = Tensor::<f32>::arrange([]);
        assert_eq!(x.data(), &vec![0.0.into()]);
        assert_eq!(x.shape(), &vec![]);
    }

    #[test]
    fn arrange_zero_shape() {
        let x = Tensor::<f32>::arrange([1, 0]);
        assert_eq!(x.data(), &vec![]);
        assert_eq!(x.shape(), &vec![1, 0]);
    }

    #[test]
    #[should_panic]
    fn arrange_error_over_shape() {
        let _ = Tensor::<f32>::arrange([usize::MAX, 1]);
    }

    #[test]
    fn add_normal() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into(), 5.0.into()], [3,]);
        let z = x + y;
        assert_eq!(z.data(), &vec![3.0.into(), 5.0.into(), 7.0.into()]);
        assert_eq!(z.shape(), &vec![3]);
    }

    #[test]
    #[should_panic]
    fn add_error_mismatch_shape() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into()], [2,]);
        let _ = x + y;
    }

    #[test]
    fn add_reference_normal() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into(), 5.0.into()], [3,]);
        let z = &x + &y;
        assert_eq!(z.data(), &vec![3.0.into(), 5.0.into(), 7.0.into()]);
        assert_eq!(z.shape(), &vec![3]);
    }

    #[test]
    #[should_panic]
    fn add_reference_error_mismatch_shape() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into()], [2,]);
        let _ = &x + &y;
    }

    #[test]
    fn sub_normal() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into(), 5.0.into()], [3,]);
        let z = x - y;
        assert_eq!(z.data(), &vec![(-3.0).into(), (-3.0).into(), (-3.0).into()]);
        assert_eq!(z.shape(), &vec![3]);
    }

    #[test]
    #[should_panic]
    fn sub_error_mismatch_shape() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into()], [2,]);
        let _ = x - y;
    }

    #[test]
    fn sub_reference_normal() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into(), 5.0.into()], [3,]);
        let z = &x - &y;
        assert_eq!(z.data(), &vec![(-3.0).into(), (-3.0).into(), (-3.0).into()]);
        assert_eq!(z.shape(), &vec![3]);
    }

    #[test]
    #[should_panic]
    fn sub_reference_error_mismatch_shape() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into()], [2,]);
        let _ = &x - &y;
    }

    #[test]
    fn mul_normal() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into(), 5.0.into()], [3,]);
        let z = x * y;
        assert_eq!(z.data(), &vec![0.0.into(), 4.0.into(), 10.0.into()]);
        assert_eq!(z.shape(), &vec![3]);
    }

    #[test]
    #[should_panic]
    fn mul_error_mismatch_shape() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into()], [2,]);
        let _ = x * y;
    }

    #[test]
    fn mul_reference_normal() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into(), 5.0.into()], [3,]);
        let z = &x * &y;
        assert_eq!(z.data(), &vec![0.0.into(), 4.0.into(), 10.0.into()]);
        assert_eq!(z.shape(), &vec![3]);
    }

    #[test]
    #[should_panic]
    fn mul_reference_error_mismatch_shape() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into()], [2,]);
        let _ = &x * &y;
    }

    #[test]
    fn div_normal() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into(), 5.0.into()], [3,]);
        let z = x / y;
        assert_eq!(z.data(), &vec![0.0.into(), 0.25.into(), 0.4.into()]);
        assert_eq!(z.shape(), &vec![3]);
    }

    #[test]
    #[should_panic]
    fn div_error_mismatch_shape() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into()], [2,]);
        let _ = x / y;
    }

    #[test]
    fn div_reference_normal() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into(), 5.0.into()], [3,]);
        let z = &x / &y;
        assert_eq!(z.data(), &vec![0.0.into(), 0.25.into(), 0.4.into()]);
        assert_eq!(z.shape(), &vec![3]);
    }

    #[test]
    #[should_panic]
    fn div_reference_error_mismatch_shape() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into()], [2,]);
        let _ = &x / &y;
    }

    #[test]
    fn neg_normal() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = -x;
        assert_eq!(y.data(), &vec![0.0.into(), (-1.0).into(), (-2.0).into()]);
        assert_eq!(y.shape(), &vec![3]);
    }

    #[test]
    fn neg_reference_normal() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = -&x;
        assert_eq!(y.data(), &vec![0.0.into(), (-1.0).into(), (-2.0).into()]);
        assert_eq!(y.shape(), &vec![3]);
    }

    #[test]
    fn add_assign_normal() {
        let mut x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into(), 5.0.into()], [3,]);
        x += &y;
        assert_eq!(x.data(), &vec![3.0.into(), 5.0.into(), 7.0.into()]);
        assert_eq!(x.shape(), &vec![3]);
    }

    #[test]
    #[should_panic]
    fn add_assign_error_mismatch_shape() {
        let mut x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into()], [2,]);
        x += &y;
    }

    #[test]
    fn sub_assign_normal() {
        let mut x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into(), 5.0.into()], [3,]);
        x -= &y;
        assert_eq!(x.data(), &vec![(-3.0).into(), (-3.0).into(), (-3.0).into()]);
        assert_eq!(x.shape(), &vec![3]);
    }

    #[test]
    #[should_panic]
    fn sub_assign_error_mismatch_shape() {
        let mut x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into()], [2,]);
        x -= &y;
    }

    #[test]
    fn mul_assign_normal() {
        let mut x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into(), 5.0.into()], [3,]);
        x *= &y;
        assert_eq!(x.data(), &vec![0.0.into(), 4.0.into(), 10.0.into()]);
        assert_eq!(x.shape(), &vec![3]);
    }

    #[test]
    #[should_panic]
    fn mul_assign_error_mismatch_shape() {
        let mut x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into()], [2,]);
        x *= &y;
    }

    #[test]
    fn div_assign_normal() {
        let mut x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into(), 5.0.into()], [3,]);
        x /= &y;
        assert_eq!(x.data(), &vec![0.0.into(), 0.25.into(), 0.4.into()]);
        assert_eq!(x.shape(), &vec![3]);
    }

    #[test]
    #[should_panic]
    fn div_assign_error_mismatch_shape() {
        let mut x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = Tensor::new([3.0.into(), 4.0.into()], [2,]);
        x /= &y;
    }

    #[test]
    fn scalar_add_normal() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = x.scalar_add(3.0.into());
        assert_eq!(y.data(), &vec![3.0.into(), 4.0.into(), 5.0.into()]);
        assert_eq!(y.shape(), &vec![3]);
    }

    #[test]
    fn scalar_sub_normal() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = x.scalar_sub(3.0.into());
        assert_eq!(y.data(), &vec![(-3.0).into(), (-2.0).into(), (-1.0).into()]);
        assert_eq!(y.shape(), &vec![3]);
    }

    #[test]
    fn scalar_mul_normal() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = x.scalar_mul(3.0.into());
        assert_eq!(y.data(), &vec![0.0.into(), 3.0.into(), 6.0.into()]);
        assert_eq!(y.shape(), &vec![3]);
    }

    #[test]
    fn scalar_div_normal() {
        let x = Tensor::new([0.0.into(), 1.0.into(), 2.0.into()], [3,]);
        let y = x.scalar_div(2.0.into());
        assert_eq!(y.data(), &vec![0.0.into(), 0.5.into(), 1.0.into()]);
        assert_eq!(y.shape(), &vec![3]);
    }
}
