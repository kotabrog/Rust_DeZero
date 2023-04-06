extern crate rust_dezero;

#[test]
fn step1() {
    use rust_dezero::Tensor;
    use rust_dezero::variable::Variable;

    let data = Tensor::new_from_num_vec(vec![1.0], vec![]);
    let mut x = Variable::new(data);
    println!("x: {:?}", x);
    x.set_data(Tensor::new_from_num_vec(vec![2.0], vec![]));
    println!("x: {:?}", x);
}

// unrecoverable
// #[test]
// fn step2() {
//     use rust_dezero::{Variable, Tensor, Function, function::sample::Square};

//     let data = Tensor::new_from_num_vec(vec![10.0], vec![]);
//     let x = Variable::new(data);
//     let mut f = Square::new();
//     let y = f.call_mut(&x);
//     println!("y: {:?}", y);
// }

// unrecoverable
// #[test]
// fn step3() {
//     use rust_dezero::{Variable, Tensor, Function, function::sample::{Square, Exp}};

//     let mut a = Square::new();
//     let mut b = Exp::new();
//     let mut c = Square::new();

//     let data = Tensor::new_from_num_vec(vec![0.5], vec![]);
//     let x = Variable::new(data);
//     let a = a.call_mut(&x);
//     let b = b.call_mut(&a);
//     let y = c.call_mut(&b);
//     println!("y: {:?}", y);
// }

// unrecoverable
// #[test]
// fn step4() {
//     use rust_dezero::{Variable, Tensor, Function, function::sample::{Square, Exp}, utility::numerical_diff};

//     fn f(x: &Variable<f64>) -> Variable<f64> {
//         let mut a = Square::new();
//         let mut b = Exp::new();
//         let mut c = Square::new();

//         c.call_mut(&b.call_mut(&a.call_mut(&x)))
//     }
//     let data = Tensor::new_from_num_vec(vec![0.5], vec![]);
//     let x = Variable::new(data);
//     let y = numerical_diff(&mut f, &x, 1e-4);
//     println!("y: {:?}", y);
// }

// unrecoverable
// #[test]
// fn step6() {
//     use rust_dezero::{Variable, Tensor, Function, function::sample::{Square, Exp}};

//     let mut a_func = Square::new();
//     let mut b_func = Exp::new();
//     let mut c_func = Square::new();

//     let data = Tensor::new_from_num_vec(vec![0.5], vec![]);
//     let mut x = Variable::new(data);
//     let mut a = a_func.call_mut(&x);
//     let mut b = b_func.call_mut(&a);
//     let mut y = c_func.call_mut(&b);

//     y.set_grad(Tensor::new_from_num_vec(vec![1.0], vec![]));
//     b.set_grad(c_func.backward(y.grad().unwrap()));
//     a.set_grad(b_func.backward(b.grad().unwrap()));
//     x.set_grad(a_func.backward(a.grad().unwrap()));

//     println!("x grad: {:?}", x.grad());
// }

#[test]
fn step7() {
    use rust_dezero::{
        Tensor,
        variable::{VariableTable, Variable, VariableWrapper, VariableType},
        function::{FunctionTable, sample::{Square, Exp}},
    };

    let mut variables = VariableTable::new();
    let mut functions = FunctionTable::new();

    let square1 = Square::new();
    let exp1 = Exp::new();
    let square2 = Square::new();

    let square1_id = functions.add_function(Box::new(square1));
    let exp1_id = functions.add_function(Box::new(exp1));
    let square2_id = functions.add_function(Box::new(square2));

    let data = Tensor::new_from_num_vec(vec![0.5], vec![]);
    let x = VariableWrapper::from_variable_f64(Variable::new(data));
    let x_id = variables.add(Box::new(x));

    let f = functions.get_mut(square1_id).unwrap();
    let a_id = f.call_mut(x_id, &mut variables);

    let f = functions.get_mut(exp1_id).unwrap();
    let b_id = f.call_mut(a_id, &mut variables);

    let f = functions.get_mut(square2_id).unwrap();
    let y_id = f.call_mut(b_id, &mut variables);

    let y = variables.get(y_id).unwrap();
    println!("y: {:?}", y);

    variables.backward(y_id, &mut functions);

    let x = variables.get(x_id).unwrap().get_variable();
    let x_grad = match x {
        VariableType::F64(x) => x.grad(),
    };
    println!("x grad: {:?}", x_grad);
}
