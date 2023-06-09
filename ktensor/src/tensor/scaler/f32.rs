use super::Scaler;

impl Scaler<f32> {
    /// Returns a number to an integer power
    pub fn powi(&self, n: i32) -> Self {
        Self { data: self.data.powi(n) }
    }

    /// Returns a number to a floating point power
    pub fn powf(&self, n: f32) -> Self {
        Self { data: self.data.powf(n) }
    }

    /// Returns exponential of the number
    pub fn exp(&self) -> Self {
        Self { data: self.data.exp() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powi_normal() {
        let x = Scaler::<f32>::new(2.0);
        assert_eq!(x.powi(2), Scaler::<f32>::new(4.0));
    }

    #[test]
    fn powf_normal() {
        let x = Scaler::<f32>::new(2.0);
        assert_eq!(x.powf(2.0), Scaler::<f32>::new(4.0));
    }

    #[test]
    fn exp_normal() {
        let x = Scaler::<f32>::new(2.0);
        assert_eq!(x.exp(), Scaler::<f32>::new((2.0 as f32).exp()));
    }
}
