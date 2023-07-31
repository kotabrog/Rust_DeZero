extern crate ktensor;

#[test]
fn sample() {
    use ktensor::Tensor;

    let x = Tensor::<f32>::arrange([2, 3]).unwrap();
    let y = Tensor::<f32>::arrange([3, 2]).unwrap();
    let z = x.matmul(&y).unwrap();

    let z = z + 1.0;

    let mut x = Tensor::<f32>::arrange([2, 2]).unwrap();
    x += &z;

    let sum = x.sum(Some([0, 1]), false);
    assert_eq!(sum.at([]).unwrap(), &101.0);
}
