use super::Tensor;

impl<T> Tensor<T>
{
    fn ops(self, rhs: Self, f: fn(T, T) -> T) -> Self {
        assert_eq!(self.shape, rhs.shape, "Shape mismatch: left: {:?}, right: {:?}", self.shape, rhs.shape);
        let data =
            self.data
            .into_iter()
            .zip(rhs.data.into_iter())
            .map(|(x, y)| f(x, y))
            .collect();
        Self { data, shape: self.shape }
    }
}

impl<T> Tensor<T>
where
    T: Clone
{
    fn ref_ops(&self, rhs: &Self, f: fn(T, T) -> T) -> Self {
        assert_eq!(self.shape, rhs.shape, "Shape mismatch: left: {:?}, right: {:?}", self.shape, rhs.shape);
        let data =
            self.data
            .iter()
            .zip(rhs.data.iter())
            .map(|(x, y)| f(x.clone(), y.clone()))
            .collect();
        Self { data, shape: self.shape.clone() }
    }
}

impl<T> Tensor<T>
where
    T: Clone
{
    fn assign_ops(&mut self, rhs: &Self, f: fn(&mut T, T) -> ()) {
        assert_eq!(self.shape, rhs.shape, "Shape mismatch: left: {:?}, right: {:?}", self.shape, rhs.shape);
        for (x, y) in self.data.iter_mut().zip(rhs.data.iter()) {
            f(x, y.clone());
        }
    }
}

impl<T> std::ops::Add for Tensor<T>
where
    T: std::ops::Add<Output = T>
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.ops(rhs, |x, y| x + y)
    }
}

impl<T> std::ops::Add for &Tensor<T>
where
    T: std::ops::Add<Output = T> + Clone
{
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.ref_ops(rhs, |x, y| x + y)
    }
}

impl<T> std::ops::Sub for Tensor<T>
where
    T: std::ops::Sub<Output = T>
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self.ops(rhs, |x, y| x - y)
    }
}

impl<T> std::ops::Sub for &Tensor<T>
where
    T: std::ops::Sub<Output = T> + Clone
{
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.ref_ops(rhs, |x, y| x - y)
    }
}

impl<T> std::ops::Mul for Tensor<T>
where
    T: std::ops::Mul<Output = T>
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        self.ops(rhs, |x, y| x * y)
    }
}

impl<T> std::ops::Mul for &Tensor<T>
where
    T: std::ops::Mul<Output = T> + Clone
{
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.ref_ops(rhs, |x, y| x * y)
    }
}

impl<T> std::ops::Div for Tensor<T>
where
    T: std::ops::Div<Output = T>
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        self.ops(rhs, |x, y| x / y)
    }
}

impl<T> std::ops::Div for &Tensor<T>
where
    T: std::ops::Div<Output = T> + Clone
{
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        self.ref_ops(rhs, |x, y| x / y)
    }
}

impl<T> std::ops::Rem for Tensor<T>
where
    T: std::ops::Rem<Output = T>
{
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        self.ops(rhs, |x, y| x % y)
    }
}

impl<T> std::ops::Rem for &Tensor<T>
where
    T: std::ops::Rem<Output = T> + Clone
{
    type Output = Tensor<T>;

    fn rem(self, rhs: Self) -> Self::Output {
        self.ref_ops(rhs, |x, y| x % y)
    }
}

impl<T> std::ops::Neg for Tensor<T>
where
    T: std::ops::Neg<Output = T>
{
    type Output = Self;

    fn neg(self) -> Self {
        let data = self.data.into_iter().map(|x| -x).collect();
        Self { data, shape: self.shape }
    }
}

impl<T> std::ops::Neg for &Tensor<T>
where
    T: std::ops::Neg<Output = T> + Clone
{
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        let data = self.data.iter().map(|x| -x.clone()).collect();
        Tensor { data, shape: self.shape.clone() }
    }
}

impl<T> std::ops::AddAssign<&Tensor<T>> for Tensor<T>
where
    T: std::ops::AddAssign + Clone
{
    fn add_assign(&mut self, other: &Self) {
        self.assign_ops(other, |x, y| { *x += y; })
    }
}

impl<T> std::ops::SubAssign<&Tensor<T>> for Tensor<T>
where
    T: std::ops::SubAssign + Clone
{
    fn sub_assign(&mut self, other: &Self) {
        self.assign_ops(other, |x, y| { *x -= y; })
    }
}

impl<T> std::ops::MulAssign<&Tensor<T>> for Tensor<T>
where
    T: std::ops::MulAssign + Clone
{
    fn mul_assign(&mut self, other: &Self) {
        self.assign_ops(other, |x, y| { *x *= y; })
    }
}

impl<T> std::ops::DivAssign<&Tensor<T>> for Tensor<T>
where
    T: std::ops::DivAssign + Clone
{
    fn div_assign(&mut self, other: &Self) {
        self.assign_ops(other, |x, y| { *x /= y; })
    }
}

impl<T> std::ops::RemAssign<&Tensor<T>> for Tensor<T>
where
    T: std::ops::RemAssign + Clone
{
    fn rem_assign(&mut self, other: &Self) {
        self.assign_ops(other, |x, y| { *x %= y; })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_normal() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([2, 3, 4, 5, 6, 7], [2, 3]).unwrap();
        let z = x + y;
        assert_eq!(z.get_data(), &vec![2, 4, 6, 8, 10, 12]);
        assert_eq!(z.get_shape(), &vec![2, 3]);
    }

    #[test]
    #[should_panic]
    fn add_error_mismatch_shape() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([2, 3, 4, 5, 6, 7], [3, 2]).unwrap();
        let _ = x + y;
    }

    #[test]
    fn add_reference_normal() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([2, 3, 4, 5, 6, 7], [2, 3]).unwrap();
        let z = &x + &y;
        assert_eq!(z.get_data(), &vec![2, 4, 6, 8, 10, 12]);
        assert_eq!(z.get_shape(), &vec![2, 3]);
    }

    #[test]
    #[should_panic]
    fn add_reference_error_mismatch_shape() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([2, 3, 4, 5, 6, 7], [3, 2]).unwrap();
        let _ = &x + &y;
    }

    #[test]
    fn sub_normal() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([3, 2, 1, 0, -1, -2], [2, 3]).unwrap();
        let z = x - y;
        assert_eq!(z.get_data(), &vec![-3, -1, 1, 3, 5, 7]);
        assert_eq!(z.get_shape(), &vec![2, 3]);
    }

    #[test]
    #[should_panic]
    fn sub_error_mismatch_shape() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([3, 2, 1, 0, -1, -2], [3, 2]).unwrap();
        let _ = x - y;
    }

    #[test]
    fn sub_reference_normal() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([3, 2, 1, 0, -1, -2], [2, 3]).unwrap();
        let z = &x - &y;
        assert_eq!(z.get_data(), &vec![-3, -1, 1, 3, 5, 7]);
        assert_eq!(z.get_shape(), &vec![2, 3]);
    }

    #[test]
    #[should_panic]
    fn sub_reference_error_mismatch_shape() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([3, 2, 1, 0, -1, -2], [3, 2]).unwrap();
        let _ = &x - &y;
    }

    #[test]
    fn mul_normal() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([3, 2, 1, 0, -1, -2], [2, 3]).unwrap();
        let z = x * y;
        assert_eq!(z.get_data(), &vec![0, 2, 2, 0, -4, -10]);
        assert_eq!(z.get_shape(), &vec![2, 3]);
    }

    #[test]
    #[should_panic]
    fn mul_error_mismatch_shape() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([3, 2, 1, 0, -1, -2], [3, 2]).unwrap();
        let _ = x * y;
    }

    #[test]
    fn mul_reference_normal() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([3, 2, 1, 0, -1, -2], [2, 3]).unwrap();
        let z = &x * &y;
        assert_eq!(z.get_data(), &vec![0, 2, 2, 0, -4, -10]);
        assert_eq!(z.get_shape(), &vec![2, 3]);
    }

    #[test]
    #[should_panic]
    fn mul_reference_error_mismatch_shape() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([3, 2, 1, 0, -1, -2], [3, 2]).unwrap();
        let _ = &x * &y;
    }

    #[test]
    fn div_normal() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([3, 2, 1, 1, 2, 3], [2, 3]).unwrap();
        let z = x / y;
        assert_eq!(z.get_data(), &vec![0, 0, 2, 3, 2, 1]);
        assert_eq!(z.get_shape(), &vec![2, 3]);
    }

    #[test]
    #[should_panic]
    fn div_error_mismatch_shape() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([3, 2, 1, 1, 2, 3], [3, 2]).unwrap();
        let _ = x / y;
    }

    #[test]
    fn div_reference_normal() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([3, 2, 1, 1, 2, 3], [2, 3]).unwrap();
        let z = &x / &y;
        assert_eq!(z.get_data(), &vec![0, 0, 2, 3, 2, 1]);
        assert_eq!(z.get_shape(), &vec![2, 3]);
    }

    #[test]
    #[should_panic]
    fn div_reference_error_mismatch_shape() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([3, 2, 1, 1, 2, 3], [3, 2]).unwrap();
        let _ = &x / &y;
    }

    #[test]
    fn rem_normal() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([3, 2, 1, 1, 2, 3], [2, 3]).unwrap();
        let z = x % y;
        assert_eq!(z.get_data(), &vec![0, 1, 0, 0, 0, 2]);
        assert_eq!(z.get_shape(), &vec![2, 3]);
    }

    #[test]
    #[should_panic]
    fn rem_error_mismatch_shape() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([3, 2, 1, 1, 2, 3], [3, 2]).unwrap();
        let _ = x % y;
    }

    #[test]
    fn rem_reference_normal() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([3, 2, 1, 1, 2, 3], [2, 3]).unwrap();
        let z = &x % &y;
        assert_eq!(z.get_data(), &vec![0, 1, 0, 0, 0, 2]);
        assert_eq!(z.get_shape(), &vec![2, 3]);
    }

    #[test]
    #[should_panic]
    fn rem_reference_error_mismatch_shape() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        let y = Tensor::new([3, 2, 1, 1, 2, 3], [3, 2]).unwrap();
        let _ = &x % &y;
    }

    #[test]
    fn neg_normal() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [3, 2]).unwrap();
        let y = -x;
        assert_eq!(y.get_data(), &vec![0, -1, -2, -3, -4, -5]);
        assert_eq!(y.get_shape(), &vec![3, 2]);
    }

    #[test]
    fn add_assign_normal() {
        let mut x = Tensor::new([0, 1, 2, 3, 4, 5], [3, 2]).unwrap();
        let y = Tensor::new([1, 2, 3, 4, 5, 6], [3, 2]).unwrap();
        x += &y;
        assert_eq!(x.get_data(), &vec![1, 3, 5, 7, 9, 11]);
        assert_eq!(x.get_shape(), &vec![3, 2]);
    }

    #[test]
    #[should_panic]
    fn add_assign_error_mismatch_shape() {
        let mut x = Tensor::new([0, 1, 2, 3, 4, 5], [3, 2]).unwrap();
        let y = Tensor::new([1, 2, 3, 4, 5, 6], [2, 3]).unwrap();
        x += &y;
    }

    #[test]
    fn sub_assign_normal() {
        let mut x = Tensor::new([0, 1, 2, 3, 4, 5], [3, 2]).unwrap();
        let y = Tensor::new([3, 2, 1, 1, 2, 3], [3, 2]).unwrap();
        x -= &y;
        assert_eq!(x.get_data(), &vec![-3, -1, 1, 2, 2, 2]);
        assert_eq!(x.get_shape(), &vec![3, 2]);
    }

    #[test]
    #[should_panic]
    fn sub_assign_error_mismatch_shape() {
        let mut x = Tensor::new([0, 1, 2, 3, 4, 5], [3, 2]).unwrap();
        let y = Tensor::new([3, 2, 1, 1, 2, 3], [2, 3]).unwrap();
        x -= &y;
    }

    #[test]
    fn mul_assign_normal() {
        let mut x = Tensor::new([0, 1, 2, 3, 4, 5], [3, 2]).unwrap();
        let y = Tensor::new([3, 2, 1, 1, 2, 3], [3, 2]).unwrap();
        x *= &y;
        assert_eq!(x.get_data(), &vec![0, 2, 2, 3, 8, 15]);
        assert_eq!(x.get_shape(), &vec![3, 2]);
    }

    #[test]
    #[should_panic]
    fn mul_assign_error_mismatch_shape() {
        let mut x = Tensor::new([0, 1, 2, 3, 4, 5], [3, 2]).unwrap();
        let y = Tensor::new([3, 2, 1, 1, 2, 3], [2, 3]).unwrap();
        x *= &y;
    }

    #[test]
    fn div_assign_normal() {
        let mut x = Tensor::new([0, 1, 2, 3, 4, 5], [3, 2]).unwrap();
        let y = Tensor::new([3, 2, 1, 1, 2, 3], [3, 2]).unwrap();
        x /= &y;
        assert_eq!(x.get_data(), &vec![0, 0, 2, 3, 2, 1]);
        assert_eq!(x.get_shape(), &vec![3, 2]);
    }

    #[test]
    #[should_panic]
    fn div_assign_error_mismatch_shape() {
        let mut x = Tensor::new([0, 1, 2, 3, 4, 5], [3, 2]).unwrap();
        let y = Tensor::new([3, 2, 1, 1, 2, 3], [2, 3]).unwrap();
        x /= &y;
    }

    #[test]
    fn rem_assign_normal() {
        let mut x = Tensor::new([0, 1, 2, 3, 4, 5], [3, 2]).unwrap();
        let y = Tensor::new([3, 2, 1, 1, 2, 3], [3, 2]).unwrap();
        x %= &y;
        assert_eq!(x.get_data(), &vec![0, 1, 0, 0, 0, 2]);
        assert_eq!(x.get_shape(), &vec![3, 2]);
    }

    #[test]
    #[should_panic]
    fn rem_assign_error_mismatch_shape() {
        let mut x = Tensor::new([0, 1, 2, 3, 4, 5], [3, 2]).unwrap();
        let y = Tensor::new([3, 2, 1, 1, 2, 3], [2, 3]).unwrap();
        x %= &y;
    }
}
