use super::Tensor;

/// Iterator for Tensor
pub struct TensorIter<'a, T> {
    iter: std::slice::Iter<'a, T>,
}

impl<T> Tensor<T> {
    pub fn iter(&self) -> TensorIter<T> {
        TensorIter {
            iter: self.data.iter(),
        }
    }
}

impl<'a, T> Iterator for TensorIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<'a, T: 'a> IntoIterator for &'a Tensor<T> {
    type Item = &'a T;
    type IntoIter = TensorIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iterator_normal() {
        let x = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let mut iter = x.into_iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), None);

        let x = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let mut iter = x.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), None);
    }
}
