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

pub fn numerical_diff<T>(f: &mut dyn FnMut(&Tensor<T>) -> Tensor<T>, x: &Tensor<T>, eps: T) -> Tensor<T>
where
    T: Float + Copy
{
    let x0 = x - eps;
    let x1 = x + eps;
    let y0 = f(&x0);
    let y1 = f(&x1);
    (y1 - y0) / (eps + eps)
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

    #[test]
    fn numerical_diff_normal() {
        let x = Tensor::new(vec![2.0], vec![])
            .unwrap();
        let mut f = |x: &Tensor<f64>| x.powi(2);
        let dy = numerical_diff(&mut f, &x, 1e-4);
        assert_approx_eq_tensor(
            &dy, &Tensor::new([4.0], []).unwrap(), 1e-6);
    }
}
