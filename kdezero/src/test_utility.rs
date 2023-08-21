use num_traits::Float;
use ktensor::Tensor;

pub fn assert_approx_eq<T>(a: T, b: T, eps: T)
where
    T: Float + std::fmt::Debug,
{
    let diff = (a - b).abs();
    if diff > eps {
        panic!(
            "assertion failed: `abs(left - right) <= eps` (left: `{:?}`, right: `{:?}`, eps: `{:?}`, diff: `{:?}`)",
            a, b, eps, diff
        );
    }
}

pub fn assert_approx_eq_tensor<T>(a: &Tensor<T>, b: &Tensor<T>, eps: T)
where
    T: Float + std::fmt::Debug,
{
    let diff = (a - b).abs();
    if diff.iter().any(|x| *x > eps) {
        panic!(
            "assertion failed: `abs(left - right) <= eps` (left: `{:?}`, right: `{:?}`, eps: `{:?}`, diff: `{:?}`)",
            a, b, eps, diff
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assert_approx_eq_normal() {
        assert_approx_eq(1.0, 0.99999, 1e-4);
    }

    #[test]
    #[should_panic]
    fn assert_approx_eq_panic() {
        assert_approx_eq(1.0, 2.0, 1e-4);
    }

    #[test]
    fn assert_approx_eq_tensor_normal() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])
            .unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 2.99999], vec![3])
            .unwrap();
        assert_approx_eq_tensor(&a, &b, 1e-4);
    }

    #[test]
    #[should_panic]
    fn assert_approx_eq_tensor_panic() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])
            .unwrap();
        let b = Tensor::new(vec![1.0, 1.0, 3.0], vec![3])
            .unwrap();
        assert_approx_eq_tensor(&a, &b, 1e-4);
    }
}
