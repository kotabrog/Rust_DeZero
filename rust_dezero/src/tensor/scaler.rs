mod f32;
mod f64;

/// Scalar is a wrapper of a single value
/// 
/// # Fields
/// 
/// * `data` - The value of the scalar
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Scaler<T>
{
    data: T,
}

impl<T> Scaler<T>
{
    /// Create a new Scaler
    pub fn new(data: T) -> Self {
        Self { data }
    }

    /// Get the data
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Get the data type
    pub fn data_type(&self) -> &str {
        std::any::type_name::<T>()
    }
}

impl<T> From<T> for Scaler<T>
{
    fn from(data: T) -> Self {
        Self { data }
    }
}

impl<T> std::ops::Add for Scaler<T>
where
    T: std::ops::Add<Output = T>
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self { data: self.data + other.data }
    }
}

impl<T> std::ops::Add for &Scaler<T>
where
    T: std::ops::Add<Output = T> + Copy
{
    type Output = Scaler<T>;

    fn add(self, other: Self) -> Self::Output {
        Self::Output { data: self.data + other.data }
    }
}

impl<T> std::ops::Sub for Scaler<T>
where
    T: std::ops::Sub<Output = T>
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self { data: self.data - other.data }
    }
}

impl<T> std::ops::Sub for &Scaler<T>
where
    T: std::ops::Sub<Output = T> + Copy
{
    type Output = Scaler<T>;

    fn sub(self, other: Self) -> Self::Output {
        Self::Output { data: self.data - other.data }
    }
}

impl<T> std::ops::Mul for Scaler<T>
where
    T: std::ops::Mul<Output = T>
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self { data: self.data * other.data }
    }
}

impl<T> std::ops::Mul for &Scaler<T>
where
    T: std::ops::Mul<Output = T> + Copy
{
    type Output = Scaler<T>;

    fn mul(self, other: Self) -> Self::Output {
        Self::Output { data: self.data * other.data }
    }
}

impl<T> std::ops::Div for Scaler<T>
where
    T: std::ops::Div<Output = T>
{
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self { data: self.data / other.data }
    }
}

impl<T> std::ops::Div for &Scaler<T>
where
    T: std::ops::Div<Output = T> + Copy
{
    type Output = Scaler<T>;

    fn div(self, other: Self) -> Self::Output {
        Self::Output { data: self.data / other.data }
    }
}

impl<T> std::ops::Neg for Scaler<T>
where
    T: std::ops::Neg<Output = T>
{
    type Output = Self;

    fn neg(self) -> Self {
        Self { data: -self.data }
    }
}

impl<T> std::ops::Neg for &Scaler<T>
where
    T: std::ops::Neg<Output = T> + Copy
{
    type Output = Scaler<T>;

    fn neg(self) -> Self::Output {
        Self::Output { data: -self.data }
    }
}

impl<T> std::ops::AddAssign for Scaler<T>
where
    T: std::ops::AddAssign
{
    fn add_assign(&mut self, other: Self) {
        self.data += other.data;
    }
}

impl<T> std::ops::SubAssign for Scaler<T>
where
    T: std::ops::SubAssign
{
    fn sub_assign(&mut self, other: Self) {
        self.data -= other.data;
    }
}

impl<T> std::ops::MulAssign for Scaler<T>
where
    T: std::ops::MulAssign
{
    fn mul_assign(&mut self, other: Self) {
        self.data *= other.data;
    }
}

impl<T> std::ops::DivAssign for Scaler<T>
where
    T: std::ops::DivAssign
{
    fn div_assign(&mut self, other: Self) {
        self.data /= other.data;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_normal() {
        let x = Scaler::<f32>::new(1.0);
        assert_eq!(*x.data(), 1.0);
        assert_eq!(x.data_type(), "f32");
    }

    #[test]
    fn from_normal() {
        let x = Scaler::<f32>::new(1.0);
        assert_eq!(x, Scaler::<f32>::from(1.0));
        assert_eq!(x, 1.0.into());
        assert_eq!(x.data_type(), "f32");
    }

    #[test]
    fn add_normal() {
        let x = Scaler::<f32>::new(1.0);
        let y = Scaler::<f32>::new(2.0);
        assert_eq!(x + y, Scaler::<f32>::new(3.0));
    }

    #[test]
    fn add_reference_normal() {
        let x = Scaler::<f32>::new(1.0);
        let y = Scaler::<f32>::new(2.0);
        assert_eq!(&x + &y, Scaler::<f32>::new(3.0));
    }

    #[test]
    fn sub_normal() {
        let x = Scaler::<f32>::new(1.0);
        let y = Scaler::<f32>::new(2.0);
        assert_eq!(x - y, Scaler::<f32>::new(-1.0));
    }

    #[test]
    fn sub_reference_normal() {
        let x = Scaler::<f32>::new(1.0);
        let y = Scaler::<f32>::new(2.0);
        assert_eq!(&x - &y, Scaler::<f32>::new(-1.0));
    }

    #[test]
    fn mul_normal() {
        let x = Scaler::<f32>::new(1.0);
        let y = Scaler::<f32>::new(2.0);
        assert_eq!(x * y, Scaler::<f32>::new(2.0));
    }

    #[test]
    fn mul_reference_normal() {
        let x = Scaler::<f32>::new(1.0);
        let y = Scaler::<f32>::new(2.0);
        assert_eq!(&x * &y, Scaler::<f32>::new(2.0));
    }

    #[test]
    fn div_normal() {
        let x = Scaler::<f32>::new(1.0);
        let y = Scaler::<f32>::new(2.0);
        assert_eq!(x / y, Scaler::<f32>::new(0.5));
    }

    #[test]
    fn div_zero() {
        let x = Scaler::<f32>::new(1.0);
        let y = Scaler::<f32>::new(0.0);
        assert_eq!(x / y, Scaler::<f32>::new(std::f32::INFINITY));
    }

    #[test]
    fn div_reference_normal() {
        let x = Scaler::<f32>::new(1.0);
        let y = Scaler::<f32>::new(2.0);
        assert_eq!(&x / &y, Scaler::<f32>::new(0.5));
    }

    #[test]
    fn neg_normal() {
        let x = Scaler::<f32>::new(1.0);
        assert_eq!(-x, Scaler::<f32>::new(-1.0));
    }

    #[test]
    fn neg_reference_normal() {
        let x = Scaler::<f32>::new(1.0);
        assert_eq!(-&x, Scaler::<f32>::new(-1.0));
    }

    #[test]
    fn add_assign_normal() {
        let mut x = Scaler::<f32>::new(1.0);
        let y = Scaler::<f32>::new(2.0);
        x += y;
        assert_eq!(x, Scaler::<f32>::new(3.0));
    }

    #[test]
    fn sub_assign_normal() {
        let mut x = Scaler::<f32>::new(1.0);
        let y = Scaler::<f32>::new(2.0);
        x -= y;
        assert_eq!(x, Scaler::<f32>::new(-1.0));
    }

    #[test]
    fn mul_assign_normal() {
        let mut x = Scaler::<f32>::new(1.0);
        let y = Scaler::<f32>::new(2.0);
        x *= y;
        assert_eq!(x, Scaler::<f32>::new(2.0));
    }

    #[test]
    fn div_assign_normal() {
        let mut x = Scaler::<f32>::new(1.0);
        let y = Scaler::<f32>::new(2.0);
        x /= y;
        assert_eq!(x, Scaler::<f32>::new(0.5));
    }
}
