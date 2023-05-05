use super::Scaler;

impl Scaler<f64> {
    /// Returns a number to an integer power
    pub fn powi(&self, n: i32) -> Self {
        Self { data: self.data.powi(n) }
    }

    /// Returns a number to a floating point power
    pub fn powf(&self, n: f64) -> Self {
        Self { data: self.data.powf(n) }
    }

    /// Returns exponential of the number
    pub fn exp(&self) -> Self {
        Self { data: self.data.exp() }
    }

    /// Returns sin of the number
    pub fn sin(&self) -> Self {
        Self { data: self.data.sin() }
    }

    /// Returns cos of the number
    pub fn cos(&self) -> Self {
        Self { data: self.data.cos() }
    }

    /// Returns tanh of the number
    pub fn tanh(&self) -> Self {
        Self { data: self.data.tanh() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powi_normal() {
        let x = Scaler::<f64>::new(2.0);
        assert_eq!(x.powi(2), Scaler::<f64>::new(4.0));
    }

    #[test]
    fn powf_normal() {
        let x = Scaler::<f64>::new(2.0);
        assert_eq!(x.powf(2.0), Scaler::<f64>::new(4.0));
    }

    #[test]
    fn exp_normal() {
        let x = Scaler::<f64>::new(2.0);
        assert_eq!(x.exp(), Scaler::<f64>::new((2.0 as f64).exp()));
    }

    #[test]
    fn sin_normal() {
        let x = Scaler::<f64>::new(2.0);
        assert_eq!(x.sin(), Scaler::<f64>::new((2.0 as f64).sin()));
    }

    #[test]
    fn cos_normal() {
        let x = Scaler::<f64>::new(2.0);
        assert_eq!(x.cos(), Scaler::<f64>::new((2.0 as f64).cos()));
    }

    #[test]
    fn tanh_normal() {
        let x = Scaler::<f64>::new(2.0);
        assert_eq!(x.tanh(), Scaler::<f64>::new((2.0 as f64).tanh()));
    }
}
