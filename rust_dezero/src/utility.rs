use crate::{Variable, Tensor};

/// Assert that two floats are approximately equal.
/// 
/// # Arguments
/// 
/// * `a` - Left hand side of the comparison
/// * `b` - Right hand side of the comparison
/// * `eps` - Maximum difference between `a` and `b`
/// 
/// # Panics
/// 
/// Panics if `a` and `b` are not approximately equal.
pub fn assert_approx_eq(a: f64, b: f64, eps: f64) {
    let diff = (a - b).abs();
    if diff > eps {
        panic!(
            "assertion failed: `abs(left - right) <= eps` (left: `{}`, right: `{}`, eps: `{}`, diff: `{}`)",
            a, b, eps, diff
        );
    }
}

/// Calculate the numerical gradient of a function.
/// 
/// # Arguments
/// 
/// * `f` - Function to calculate the gradient of
/// * `x` - Input variable
/// * `eps` - Small value to calculate the gradient
pub fn numerical_diff<T>(f: &dyn Fn(&Variable<T>) -> Variable<T>, x: &Variable<T>, eps: T) -> Tensor<T>
where
    T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> +
        std::ops::Mul<Output = T> + std::ops::Div<Output = T> +
        Copy + From<i8>,
{
    let x0 = Variable::<T>::new(x.data().scalar_sub(eps.into()));
    let x1 = Variable::<T>::new(x.data().scalar_add(eps.into()));
    let y0 = f(&x0);
    let y1 = f(&x1);
    (y1.data() - y0.data()).scalar_div((T::from(2) * eps).into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function::sample::Square;
    use crate::Function;

    #[test]
    fn assert_approx_eq_normal() {
        assert_approx_eq(1.0, 1.0, 1e-4);
    }

    #[test]
    #[should_panic]
    fn assert_approx_eq_panic() {
        assert_approx_eq(1.0, 2.0, 1e-4);
    }

    #[test]
    fn numerical_diff_normal() {
        let x = Variable::<f64>::new(Tensor::new_from_num_vec(vec![2.0], vec![]));
        let square = Square::new();
        let f = |x: &Variable<f64>| square.call(x);
        let dy = numerical_diff(&f, &x, 1e-4);
        assert_approx_eq(*dy.at(&[]).data(), 4.0, 1e-6);
    }
}
