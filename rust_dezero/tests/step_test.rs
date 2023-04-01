extern crate rust_dezero;

#[test]
fn step1() {
    use rust_dezero::{Variable, Tensor};

    let data = Tensor::new_from_num_vec(vec![1.0], vec![]);
    let mut x = Variable::new(data);
    println!("x: {:?}", x);
    x.set_data(Tensor::new_from_num_vec(vec![2.0], vec![]));
    println!("x: {:?}", x);
}

#[test]
fn step2() {
    use rust_dezero::{Variable, Tensor, Function, function::sample::Square};

    let data = Tensor::new_from_num_vec(vec![10.0], vec![]);
    let x = Variable::new(data);
    let f = Square::new();
    let y = f.call(x);
    println!("y: {:?}", y);
}

#[test]
fn step3() {
    use rust_dezero::{Variable, Tensor, Function, function::sample::{Square, Exp}};

    let a = Square::new();
    let b = Exp::new();
    let c = Square::new();

    let data = Tensor::new_from_num_vec(vec![0.5], vec![]);
    let x = Variable::new(data);
    let a = a.call(x);
    let b = b.call(a);
    let y = c.call(b);
    println!("y: {:?}", y);
}
