use crate::Tensor;

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
/// * `x` - Input tensor
/// * `eps` - Small value to calculate the gradient
pub fn numerical_diff<T>(f: &mut dyn FnMut(&Tensor<T>) -> Tensor<T>, x: &Tensor<T>, eps: T) -> Tensor<T>
where
    T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> +
        std::ops::Mul<Output = T> + std::ops::Div<Output = T> +
        Copy + From<i8>,
{
    let x0 = x.scalar_sub(eps.into());
    let x1 = x.scalar_add(eps.into());
    let y0 = f(&x0);
    let y1 = f(&x1);
    (y1 - y0).scalar_div((T::from(2) * eps).into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

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
        let x = Tensor::new_from_num_vec(vec![2.0], vec![]);
        let mut f = |x: &Tensor<f64>| x.powi(2);
        let dy = numerical_diff(&mut f, &x, 1e-4);
        assert_approx_eq(*dy.at(&[]).data(), 4.0, 1e-6);
    }
}
