extern crate rust_dezero;

#[test]
fn step1() {
    use rust_dezero::Tensor;
    use rust_dezero::variable::{VariableContents, VariableTable};

    let mut table = VariableTable::new();
    let data = Tensor::new_from_num_vec(vec![1.0], vec![]);
    let id = table.generate_variable_from_f64_tensor(data, "x");
    let x = table.get_mut(id).unwrap();
    assert_eq!(x.get_data().to_f64_tensor().unwrap(), &Tensor::new_from_num_vec(vec![1.0], vec![]));
    println!("x: {:?}", x);
    *x.get_mut_data() = VariableContents::F64(Box::new(Tensor::new_from_num_vec(vec![2.0], vec![])));
    assert_eq!(x.get_data().to_f64_tensor().unwrap(), &Tensor::new_from_num_vec(vec![2.0], vec![]));
    println!("x: {:?}", x);
}

#[test]
fn step2() {
    use rust_dezero::{
        Tensor,
        variable::VariableTable,
        function::{FunctionTable, operator::Square},
    };

    let mut variable_table = VariableTable::new();
    let mut function_table = FunctionTable::new();

    let data = vec![10.0];
    let square_id = function_table.generate_function_from_function_contents(Box::new(Square::new()));
    let x_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data.clone(), vec![]), "x");
    let y_id = function_table.forward(square_id, vec![x_id], &mut variable_table, false);

    let y = variable_table.get_variable_contents_f64(y_id[0]).unwrap();
    assert_eq!(y.data(), Tensor::new_from_num_vec(vec![100.0], vec![]).data());
    println!("y: {:?}", y);
}

#[test]
fn step3() {
    use rust_dezero::{
        Tensor,
        variable::VariableTable,
        function::{FunctionTable, operator::{Square, Exp}},
    };

    let mut variable_table = VariableTable::new();
    let mut function_table = FunctionTable::new();

    let data = vec![0.5];
    let square_id0 = function_table.generate_function_from_function_contents(Box::new(Square::new()));
    let exp_id = function_table.generate_function_from_function_contents(Box::new(Exp::new()));
    let square_id1 = function_table.generate_function_from_function_contents(Box::new(Square::new()));
    let x_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data.clone(), vec![]), "x");

    let y_id = function_table.forward(square_id0, vec![x_id], &mut variable_table, false);
    let y_id = function_table.forward(exp_id, y_id, &mut variable_table, false);
    let y_id = function_table.forward(square_id1, y_id, &mut variable_table, false);

    let y = variable_table.get_variable_contents_f64(y_id[0]).unwrap();
    assert_eq!(y, &Tensor::new_from_num_vec(vec![1.648721270700128], vec![]));
    println!("y: {:?}", y);
}

#[test]
fn step4() {
    use rust_dezero::{
        Tensor,
        variable::VariableTable,
        function::{FunctionTable, operator::{Square, Exp}},
        utility::{numerical_diff, assert_approx_eq},
    };

    let mut variable_table = VariableTable::new();
    let mut function_table = FunctionTable::new();

    let data = vec![0.5];
    let square_id0 = function_table.generate_function_from_function_contents(Box::new(Square::new()));
    let exp_id = function_table.generate_function_from_function_contents(Box::new(Exp::new()));
    let square_id1 = function_table.generate_function_from_function_contents(Box::new(Square::new()));
    let x_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data.clone(), vec![]), "x");

    let y_id = function_table.forward(square_id0, vec![x_id], &mut variable_table, false);
    let y_id = function_table.forward(exp_id, y_id, &mut variable_table, false);
    let y_id = function_table.forward(square_id1, y_id, &mut variable_table, false);

    let y = variable_table.get_variable_contents_f64(y_id[0]).unwrap();
    println!("y: {:?}", y);

    fn f(x: &Tensor<f64>) -> Tensor<f64> {
        x.powi(2).exp().powi(2)
    }
    let x = variable_table.get_variable_contents_f64(x_id).unwrap();
    let grad = numerical_diff(&mut f, x, 1e-6);
    println!("grad: {:?}", grad);
    assert_approx_eq(*grad.data()[0].data(), 3.297442, 1e-4)
}

#[test]
fn step7() {
    use rust_dezero::{
        Tensor,
        variable::VariableTable,
        function::{FunctionTable, operator::{Square, Exp}},
        utility::{numerical_diff, assert_approx_eq},
    };

    let mut variable_table = VariableTable::new();
    let mut function_table = FunctionTable::new();

    let data = vec![0.5];
    let square_id0 = function_table.generate_function_from_function_contents(Box::new(Square::new()));
    let exp_id = function_table.generate_function_from_function_contents(Box::new(Exp::new()));
    let square_id1 = function_table.generate_function_from_function_contents(Box::new(Square::new()));
    let x_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data.clone(), vec![]), "x");

    let y_id = function_table.forward(square_id0, vec![x_id], &mut variable_table, false);
    let y_id = function_table.forward(exp_id, y_id, &mut variable_table, false);
    let y_id = function_table.forward(square_id1, y_id, &mut variable_table, false);

    variable_table.backward(y_id, &mut function_table, false);

    let x_grad = variable_table.get_variable_grad_contents_f64(x_id).unwrap();
    println!("x_grad: {:?}", x_grad);

    fn f(x: &Tensor<f64>) -> Tensor<f64> {
        x.powi(2).exp().powi(2)
    }
    let x = variable_table.get_variable_contents_f64(x_id).unwrap();
    let grad = numerical_diff(&mut f, x, 1e-6);
    println!("grad: {:?}", grad);

    assert_approx_eq(*grad.data()[0].data(), *x_grad.data()[0].data(), 1e-4)
}

#[test]
fn step11() {
    use rust_dezero::{
        Tensor,
        variable::VariableTable,
        function::{FunctionTable, operator::Add},
    };

    let mut variable_table = VariableTable::new();
    let mut function_table = FunctionTable::new();

    let data0 = vec![2.0];
    let data1 = vec![3.0];
    let add_id = function_table.generate_function_from_function_contents(Box::new(Add::new()));
    let a_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data0.clone(), vec![]), "a");
    let b_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data1.clone(), vec![]), "b");

    let y_id = function_table.forward(add_id, vec![a_id, b_id], &mut variable_table, false);

    let y = variable_table.get_variable_contents_f64(y_id[0]).unwrap();
    println!("y: {:?}", y);
    assert_eq!(*y.data()[0].data(), 5.0);
}

#[test]
fn step13() {
    use rust_dezero::{
        Tensor,
        variable::VariableTable,
        function::{FunctionTable, operator::{Add, Square}},
    };

    let mut variable_table = VariableTable::new();
    let mut function_table = FunctionTable::new();

    let data0 = vec![2.0];
    let data1 = vec![3.0];
    let square_id0 = function_table.generate_function_from_function_contents(Box::new(Square::new()));
    let square_id1 = function_table.generate_function_from_function_contents(Box::new(Square::new()));
    let add_id = function_table.generate_function_from_function_contents(Box::new(Add::new()));
    let a_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data0.clone(), vec![]), "a");
    let b_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data1.clone(), vec![]), "b");

    let x_id = function_table.forward(square_id0, vec![a_id], &mut variable_table, false);
    let y_id = function_table.forward(square_id1, vec![b_id], &mut variable_table, false);
    let z_id = function_table.forward(add_id, vec![x_id[0], y_id[0]], &mut variable_table, false);

    let z = variable_table.get_variable_contents_f64(z_id[0]).unwrap();
    println!("z: {:?}", z);
    assert_eq!(*z.data()[0].data(), 13.0);

    variable_table.backward(z_id, &mut function_table, false);

    let a_grad = variable_table.get_variable_grad_contents_f64(a_id).unwrap();
    let b_grad = variable_table.get_variable_grad_contents_f64(b_id).unwrap();
    println!("a_grad: {:?}", a_grad);
    println!("b_grad: {:?}", b_grad);
    assert_eq!(a_grad.data()[0].data(), &4.0);
    assert_eq!(b_grad.data()[0].data(), &6.0);
}

#[test]
fn step14() {
    use rust_dezero::{
        Tensor,
        variable::VariableTable,
        function::{FunctionTable, operator::Add},
    };

    let mut variable_table = VariableTable::new();
    let mut function_table = FunctionTable::new();

    let data = vec![3.0];
    let add_id0 = function_table.generate_function_from_function_contents(Box::new(Add::new()));
    let add_id1 = function_table.generate_function_from_function_contents(Box::new(Add::new()));
    let add_id2 = function_table.generate_function_from_function_contents(Box::new(Add::new()));
    let x_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data.clone(), vec![]), "x");

    let y_id = function_table.forward(add_id0, vec![x_id, x_id], &mut variable_table, false);
    variable_table.backward(y_id, &mut function_table, false);

    let x_grad = variable_table.get_variable_grad_contents_f64(x_id).unwrap();
    println!("x_grad: {:?}", x_grad);
    assert_eq!(x_grad.data()[0].data(), &2.0);

    variable_table.clear_grad(x_id);

    let y_id = function_table.forward(add_id1, vec![x_id, x_id], &mut variable_table, false);
    let z_id = function_table.forward(add_id2, vec![y_id[0], x_id], &mut variable_table, false);

    variable_table.backward(z_id, &mut function_table, false);


    let x_grad = variable_table.get_variable_grad_contents_f64(x_id).unwrap();
    println!("x_grad: {:?}", x_grad);
    assert_eq!(x_grad.data()[0].data(), &3.0);
}

#[test]
fn step16() {
    use rust_dezero::{
        Tensor,
        variable::VariableTable,
        function::{FunctionTable, operator::{Add, Square}},
    };

    let mut variable_table = VariableTable::new();
    let mut function_table = FunctionTable::new();

    let data = vec![2.0];
    let square_id0 = function_table.generate_function_from_function_contents(Box::new(Square::new()));
    let square_id1 = function_table.generate_function_from_function_contents(Box::new(Square::new()));
    let square_id2 = function_table.generate_function_from_function_contents(Box::new(Square::new()));
    let add_id = function_table.generate_function_from_function_contents(Box::new(Add::new()));
    let x_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data.clone(), vec![]), "a");

    let a_id = function_table.forward(square_id0, vec![x_id], &mut variable_table, false)[0];
    let b_id = function_table.forward(square_id1, vec![a_id], &mut variable_table, false)[0];
    let c_id = function_table.forward(square_id2, vec![a_id], &mut variable_table, false)[0];
    let z_id = function_table.forward(add_id, vec![b_id, c_id], &mut variable_table, false);

    let z = variable_table.get_variable_contents_f64(z_id[0]).unwrap();
    println!("z: {:?}", z);
    assert_eq!(*z.data()[0].data(), 32.0);

    variable_table.backward(z_id, &mut function_table, false);

    let x_grad = variable_table.get_variable_grad_contents_f64(x_id).unwrap();
    println!("x_grad: {:?}", x_grad);
    assert_eq!(x_grad.data()[0].data(), &64.0);
}

#[test]
fn step18_1() {
    use rust_dezero::{
        Tensor,
        variable::VariableTable,
        function::{FunctionTable, operator::Add},
    };

    let mut variable_table = VariableTable::new();
    let mut function_table = FunctionTable::new();

    let data = vec![2.0];
    let add_id0 = function_table.generate_function_from_function_contents(Box::new(Add::new()));
    let add_id1 = function_table.generate_function_from_function_contents(Box::new(Add::new()));
    let x0_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data.clone(), vec![]), "x0");
    let x1_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data.clone(), vec![]), "x1");

    let t_id = function_table.forward(add_id0, vec![x0_id, x1_id], &mut variable_table, false);
    let y_id = function_table.forward(add_id1, vec![t_id[0], x0_id], &mut variable_table, false);

    variable_table.backward(y_id.clone(), &mut function_table, false);

    let y_grad = variable_table.get_variable_grad_contents_f64(y_id[0]);
    println!("y_grad: {:?}", y_grad);
    assert_eq!(y_grad, None);

    let t_grad = variable_table.get_variable_grad_contents_f64(t_id[0]);
    println!("t_grad: {:?}", t_grad);
    assert_eq!(t_grad, None);

    let x0_grad = variable_table.get_variable_grad_contents_f64(x0_id).unwrap();
    println!("x0_grad: {:?}", x0_grad);
    assert_eq!(x0_grad.data()[0].data(), &2.0);

    let x1_grad = variable_table.get_variable_grad_contents_f64(x1_id).unwrap();
    println!("x1_grad: {:?}", x1_grad);
    assert_eq!(x1_grad.data()[0].data(), &1.0);
}

#[test]
fn step18_2() {
    use rust_dezero::{
        Tensor,
        variable::VariableTable,
        function::{FunctionTable, operator::Square},
    };

    let mut variable_table = VariableTable::new();
    let mut function_table = FunctionTable::new();

    let data = vec![2.0];
    let square_id = function_table.generate_function_from_function_contents(Box::new(Square::new()));
    let x_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data.clone(), vec![]), "x");

    let y_id = function_table.forward(square_id, vec![x_id], &mut variable_table, true);

    let y = variable_table.get_variable_contents_f64(y_id[0]).unwrap();
    println!("y: {:?}", y);
    assert_eq!(y.data()[0].data(), &4.0);
}

#[test]
fn step19() {
    use rust_dezero::{
        Tensor,
        variable::VariableTable,
    };

    let mut variable_table = VariableTable::new();

    let data = vec![2.0];
    let x_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data.clone(), vec![]), "x");

    let x = variable_table.get(x_id).unwrap();
    println!("x.name: {:?}", x.get_name());
    assert_eq!(x.get_name(), "x");
}

#[test]
fn step20() {
    use rust_dezero::{
        Tensor,
        variable::VariableTable,
        function::{FunctionTable, operator::{Add, Mul}},
    };

    let mut variable_table = VariableTable::new();
    let mut function_table = FunctionTable::new();

    let data0 = vec![3.0];
    let data1 = vec![2.0];
    let data2 = vec![1.0];
    let mul_id = function_table.generate_function_from_function_contents(Box::new(Mul::new()));
    let add_id = function_table.generate_function_from_function_contents(Box::new(Add::new()));
    let a_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data0.clone(), vec![]), "a");
    let b_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data1.clone(), vec![]), "b");
    let c_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data2.clone(), vec![]), "c");

    let y_id = function_table.forward(mul_id, vec![a_id, b_id], &mut variable_table, false)[0];
    let y_id = function_table.forward(add_id, vec![y_id, c_id], &mut variable_table, false);

    let y = variable_table.get_variable_contents_f64(y_id[0]).unwrap();
    println!("y: {:?}", y);
    assert_eq!(*y.data()[0].data(), 7.0);

    variable_table.backward(y_id, &mut function_table, false);

    let a_grad = variable_table.get_variable_grad_contents_f64(a_id).unwrap();
    println!("a_grad: {:?}", a_grad);
    assert_eq!(a_grad.data()[0].data(), &2.0);

    let b_grad = variable_table.get_variable_grad_contents_f64(b_id).unwrap();
    println!("b_grad: {:?}", b_grad);
    assert_eq!(b_grad.data()[0].data(), &3.0);
}

#[test]
fn step26() {
    use std::fs::create_dir;
    use rust_dezero::{
        Tensor,
        variable::VariableTable,
        function::{FunctionTable, operator::{Add, Mul}},
    };

    let mut variable_table = VariableTable::new();
    let mut function_table = FunctionTable::new();

    let data0 = vec![3.0];
    let data1 = vec![2.0];
    let data2 = vec![1.0];
    let mul_id = function_table.generate_function_from_function_contents(Box::new(Mul::new()));
    let add_id = function_table.generate_function_from_function_contents(Box::new(Add::new()));
    let a_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data0.clone(), vec![]), "a");
    let b_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data1.clone(), vec![]), "b");
    let c_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data2.clone(), vec![]), "c");

    let y_id = function_table.forward(mul_id, vec![a_id, b_id], &mut variable_table, false)[0];
    let y_id = function_table.forward(add_id, vec![y_id, c_id], &mut variable_table, false);

    match create_dir("output") {
        Ok(_) => println!("create output directory"),
        Err(_) => {},
    }
    variable_table.plot_dot_graph(y_id, &function_table, "output/sample", true);
}

// #[test]
// fn step27() {
//     use rust_dezero::{
//         Tensor,
//         variable::{VariableTable, Variable, VariableWrapper, VariableType},
//         function::{FunctionTable, sample::Sin},
//     };

//     let mut variables = VariableTable::new();
//     let mut functions = FunctionTable::new();

//     let sin = Sin::new();

//     let sin_id = functions.add_function(Box::new(sin));

//     let data = Tensor::new_from_num_vec(vec![std::f64::consts::FRAC_PI_4], vec![]);
//     let x = VariableWrapper::from_variable_f64(Variable::new(data), Some("x"));
//     let x_id = variables.add(Box::new(x));

//     let f = functions.get_mut(sin_id).unwrap();
//     let y = f.call_mut(vec![x_id], &mut variables, false);

//     variables.backward(y.clone(), &mut functions, false);

//     let y = variables.get(y[0]).unwrap().get_variable();
//     let y = match y {
//         VariableType::F64(y) => y.data(),
//     };
//     assert_eq!(y, &Tensor::new_from_num_vec(vec![0.7071067811865475], vec![]));
//     println!("y: {:?}", y);

//     let x = variables.get_mut(x_id).unwrap().get_variable();
//     let x_grad = match x {
//         VariableType::F64(x) => x.grad().unwrap(),
//     };
//     assert_eq!(x_grad, &Tensor::new_from_num_vec(vec![0.7071067811865476], vec![]));
//     println!("x grad: {:?}", x_grad);
// }
