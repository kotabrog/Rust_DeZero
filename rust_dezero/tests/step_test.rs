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

// #[test]
// fn step7() {
//     use rust_dezero::{
//         Tensor,
//         variable::{VariableTable, Variable, VariableWrapper, VariableType},
//         function::{FunctionTable, sample::{Square, Exp}},
//     };

//     let mut variables = VariableTable::new();
//     let mut functions = FunctionTable::new();

//     let square1 = Square::new();
//     let exp1 = Exp::new();
//     let square2 = Square::new();

//     let square1_id = functions.add_function(Box::new(square1));
//     let exp1_id = functions.add_function(Box::new(exp1));
//     let square2_id = functions.add_function(Box::new(square2));

//     let data = Tensor::new_from_num_vec(vec![0.5], vec![]);
//     let x = VariableWrapper::from_variable_f64(Variable::new(data), None);
//     let x_id = variables.add(Box::new(x));

//     let f = functions.get_mut(square1_id).unwrap();
//     let a_id = f.call_mut(vec![x_id], &mut variables, false);

//     let f = functions.get_mut(exp1_id).unwrap();
//     let b_id = f.call_mut(a_id, &mut variables, false);

//     let f = functions.get_mut(square2_id).unwrap();
//     let y_id = f.call_mut(b_id, &mut variables, false);

//     let y = variables.get(y_id[0]).unwrap();
//     println!("y: {:?}", y);

//     variables.backward(y_id, &mut functions, true);

//     let x = variables.get(x_id).unwrap().get_variable();
//     let x_grad = match x {
//         VariableType::F64(x) => x.grad(),
//     };
//     println!("x grad: {:?}", x_grad);
// }

// #[test]
// fn step11() {
//     use rust_dezero::{
//         Tensor,
//         variable::{VariableTable, Variable, VariableWrapper},
//         function::{FunctionTable, sample::Add},
//     };

//     let mut variables = VariableTable::new();
//     let mut functions = FunctionTable::new();

//     let function = Add::new();

//     let function_id = functions.add_function(Box::new(function));

//     let data = Tensor::new_from_num_vec(vec![2.0], vec![]);
//     let a = VariableWrapper::from_variable_f64(Variable::new(data), None);
//     let a_id = variables.add(Box::new(a));

//     let data = Tensor::new_from_num_vec(vec![3.0], vec![]);
//     let b = VariableWrapper::from_variable_f64(Variable::new(data), None);
//     let b_id = variables.add(Box::new(b));

//     let f = functions.get_mut(function_id).unwrap();
//     let y_id = f.call_mut(vec![a_id, b_id], &mut variables, false);

//     let y = variables.get(y_id[0]).unwrap();
//     println!("y: {:?}", y);
// }

// #[test]
// fn step12() {
//     use rust_dezero::{
//         Tensor,
//         variable::{VariableTable, Variable, VariableWrapper, VariableType},
//         function::{FunctionTable, sample::{Add, Square}},
//     };

//     let mut variables = VariableTable::new();
//     let mut functions = FunctionTable::new();

//     let square1 = Square::new();
//     let square2 = Square::new();
//     let add = Add::new();

//     let square1_id = functions.add_function(Box::new(square1));
//     let square2_id = functions.add_function(Box::new(square2));
//     let add_id = functions.add_function(Box::new(add));

//     let data = Tensor::new_from_num_vec(vec![2.0], vec![]);
//     let a = VariableWrapper::from_variable_f64(Variable::new(data), None);
//     let a_id = variables.add(Box::new(a));

//     let data = Tensor::new_from_num_vec(vec![3.0], vec![]);
//     let b = VariableWrapper::from_variable_f64(Variable::new(data), None);
//     let b_id = variables.add(Box::new(b));

//     let f = functions.get_mut(square1_id).unwrap();
//     let c_id = f.call_mut(vec![a_id], &mut variables, false)[0];

//     let f = functions.get_mut(square2_id).unwrap();
//     let d_id = f.call_mut(vec![b_id], &mut variables, false)[0];

//     let f = functions.get_mut(add_id).unwrap();
//     let y_id = f.call_mut(vec![c_id, d_id], &mut variables, false);

//     variables.backward(y_id.clone(), &mut functions, true);

//     let a = variables.get(a_id).unwrap().get_variable();
//     let b = variables.get(b_id).unwrap().get_variable();
//     let y = variables.get(y_id[0]).unwrap().get_variable();

//     let a_grad = match a {
//         VariableType::F64(a) => a.grad().unwrap(),
//     };
//     let b_grad = match b {
//         VariableType::F64(b) => b.grad().unwrap(),
//     };
//     let y = match y {
//         VariableType::F64(y) => y.data(),
//     };

//     println!("y: {:?}", y);
//     println!("a grad: {:?}", a_grad);
//     println!("b grad: {:?}", b_grad);
// }

// #[test]
// fn step14() {
//     use rust_dezero::{
//         Tensor,
//         variable::{VariableTable, Variable, VariableWrapper, VariableType},
//         function::{FunctionTable, sample::Add},
//     };

//     let mut variables = VariableTable::new();
//     let mut functions = FunctionTable::new();

//     let add1 = Add::new();
//     let add2 = Add::new();
//     let add3 = Add::new();

//     let add1_id = functions.add_function(Box::new(add1));
//     let add2_id = functions.add_function(Box::new(add2));
//     let add3_id = functions.add_function(Box::new(add3));

//     let data = Tensor::new_from_num_vec(vec![3.0], vec![]);
//     let x = VariableWrapper::from_variable_f64(Variable::new(data), None);
//     let x_id = variables.add(Box::new(x));

//     let f = functions.get_mut(add1_id).unwrap();
//     let y = f.call_mut(vec![x_id, x_id], &mut variables, false);

//     variables.backward(y, &mut functions, true);

//     let x = variables.get_mut(x_id).unwrap();
//     let x_type = x.get_variable();
//     let x_grad = match x_type {
//         VariableType::F64(var) => var.grad().unwrap(),
//     };
//     println!("x grad: {:?}", x_grad);

//     x.clear_grad();

//     let f = functions.get_mut(add2_id).unwrap();
//     let y_id = f.call_mut(vec![x_id, x_id], &mut variables, false)[0];

//     let f = functions.get_mut(add3_id).unwrap();
//     let y = f.call_mut(vec![y_id, x_id], &mut variables, false);

//     variables.backward(y, &mut functions, true);

//     let x = variables.get_mut(x_id).unwrap().get_variable();
//     let x_grad = match x {
//         VariableType::F64(x) => x.grad().unwrap(),
//     };
//     println!("x grad: {:?}", x_grad);
// }

// #[test]
// fn step16() {
//     use rust_dezero::{
//         Tensor,
//         variable::{VariableTable, Variable, VariableWrapper, VariableType},
//         function::{FunctionTable, sample::{Add, Square}},
//     };

//     let mut variables = VariableTable::new();
//     let mut functions = FunctionTable::new();

//     let square1 = Square::new();
//     let square2 = Square::new();
//     let square3 = Square::new();
//     let add = Add::new();

//     let square1_id = functions.add_function(Box::new(square1));
//     let square2_id = functions.add_function(Box::new(square2));
//     let square3_id = functions.add_function(Box::new(square3));
//     let add_id = functions.add_function(Box::new(add));

//     let data = Tensor::new_from_num_vec(vec![2.0], vec![]);
//     let x = VariableWrapper::from_variable_f64(Variable::new(data), None);
//     let x_id = variables.add(Box::new(x));

//     let f = functions.get_mut(square1_id).unwrap();
//     let a = f.call_mut(vec![x_id], &mut variables, false);

//     let f = functions.get_mut(square2_id).unwrap();
//     let b = f.call_mut(a.clone(), &mut variables, false);

//     let f = functions.get_mut(square3_id).unwrap();
//     let c = f.call_mut(a.clone(), &mut variables, false);

//     let f = functions.get_mut(add_id).unwrap();
//     let y = f.call_mut(vec![b[0], c[0]], &mut variables, false);

//     variables.backward(y.clone(), &mut functions, true);

//     let y = variables.get(y[0]).unwrap().get_variable();
//     let y = match y {
//         VariableType::F64(y) => y.data(),
//     };
//     assert_eq!(y, &Tensor::new_from_num_vec(vec![32.0], vec![]));
//     println!("y: {:?}", y);

//     let x = variables.get_mut(x_id).unwrap().get_variable();
//     let x_grad = match x {
//         VariableType::F64(x) => x.grad().unwrap(),
//     };
//     assert_eq!(x_grad, &Tensor::new_from_num_vec(vec![64.0], vec![]));
//     println!("x grad: {:?}", x_grad);
// }

// #[test]
// fn step18_1() {
//     use rust_dezero::{
//         Tensor,
//         variable::{VariableTable, Variable, VariableWrapper, VariableType},
//         function::{FunctionTable, sample::Add},
//     };

//     let mut variables = VariableTable::new();
//     let mut functions = FunctionTable::new();

//     let add1 = Add::new();
//     let add2 = Add::new();

//     let add1_id = functions.add_function(Box::new(add1));
//     let add2_id = functions.add_function(Box::new(add2));

//     let data = Tensor::new_from_num_vec(vec![1.0], vec![]);
//     let x0 = VariableWrapper::from_variable_f64(Variable::new(data.clone()), None);
//     let x0_id = variables.add(Box::new(x0));

//     let x1 = VariableWrapper::from_variable_f64(Variable::new(data), None);
//     let x1_id = variables.add(Box::new(x1));

//     let f = functions.get_mut(add1_id).unwrap();
//     let t = f.call_mut(vec![x0_id, x1_id], &mut variables, false);

//     let f = functions.get_mut(add2_id).unwrap();
//     let y = f.call_mut(vec![x0_id, t[0]], &mut variables, false);

//     variables.backward(y.clone(), &mut functions, false);

//     let y = variables.get(y[0]).unwrap().get_variable();
//     let y_grad = match y {
//         VariableType::F64(y) => y.grad(),
//     };
//     assert_eq!(y_grad, None);
//     println!("y grad: {:?}", y_grad);

//     let t = variables.get(t[0]).unwrap().get_variable();
//     let t_grad = match t {
//         VariableType::F64(t) => t.grad(),
//     };
//     assert_eq!(t_grad, None);
//     println!("t grad: {:?}", t_grad);

//     let x0 = variables.get_mut(x0_id).unwrap().get_variable();
//     let x0_grad = match x0 {
//         VariableType::F64(x0) => x0.grad().unwrap(),
//     };
//     assert_eq!(x0_grad, &Tensor::new_from_num_vec(vec![2.0], vec![]));
//     println!("x0 grad: {:?}", x0_grad);

//     let x1 = variables.get_mut(x1_id).unwrap().get_variable();
//     let x1_grad = match x1 {
//         VariableType::F64(x1) => x1.grad().unwrap(),
//     };
//     assert_eq!(x1_grad, &Tensor::new_from_num_vec(vec![1.0], vec![]));
//     println!("x1 grad: {:?}", x1_grad);
// }

// #[test]
// fn step18_2() {
//     use rust_dezero::{
//         Tensor,
//         variable::{VariableTable, Variable, VariableWrapper, VariableType},
//         function::{FunctionTable, sample::Square},
//     };

//     let mut variables = VariableTable::new();
//     let mut functions = FunctionTable::new();

//     let square = Square::new();

//     functions.add_function(Box::new(square));

//     let data = Tensor::new_from_num_vec(vec![2.0], vec![]);
//     let x = VariableWrapper::from_variable_f64(Variable::new(data.clone()), None);
//     let x_id = variables.add(Box::new(x));

//     let f = functions.get_mut(x_id).unwrap();
//     let y = f.call_mut(vec![x_id], &mut variables, true);

//     let y = variables.get(y[0]).unwrap().get_variable();
//     let y = match y {
//         VariableType::F64(y) => y.data(),
//     };
//     assert_eq!(y, &Tensor::new_from_num_vec(vec![4.0], vec![]));
//     println!("y: {:?}", y);
// }

// #[test]
// fn step19() {
//     use rust_dezero::{
//         Tensor,
//         variable::{Variable, VariableWrapper},
//     };

//     let x = VariableWrapper::from_variable_f64(Variable::new(Tensor::new_from_num_vec(vec![2.0], vec![])), None);
//     let x_name = x.get_name();
//     assert_eq!(x_name, "");
//     println!("x name: {:?}", x_name);

//     let x = VariableWrapper::from_variable_f64(Variable::new(Tensor::new_from_num_vec(vec![2.0], vec![])), Some("x"));
//     let x_name = x.get_name();
//     assert_eq!(x_name, "x");
//     println!("x name: {:?}", x_name);
// }

// #[test]
// fn step20() {
//     use rust_dezero::{
//         Tensor,
//         variable::{VariableTable, Variable, VariableWrapper, VariableType},
//         function::{FunctionTable, sample::{Add, Mul}},
//     };

//     let mut variables = VariableTable::new();
//     let mut functions = FunctionTable::new();

//     let add = Add::new();
//     let mul = Mul::new();

//     let add_id = functions.add_function(Box::new(add));
//     let mul_id = functions.add_function(Box::new(mul));

//     let data = Tensor::new_from_num_vec(vec![3.0], vec![]);
//     let a = VariableWrapper::from_variable_f64(Variable::new(data), None);
//     let a_id = variables.add(Box::new(a));

//     let data = Tensor::new_from_num_vec(vec![2.0], vec![]);
//     let b = VariableWrapper::from_variable_f64(Variable::new(data), None);
//     let b_id = variables.add(Box::new(b));

//     let data = Tensor::new_from_num_vec(vec![1.0], vec![]);
//     let c = VariableWrapper::from_variable_f64(Variable::new(data), None);
//     let c_id = variables.add(Box::new(c));

//     let f = functions.get_mut(mul_id).unwrap();
//     let y = f.call_mut(vec![a_id, b_id], &mut variables, false);

//     let f = functions.get_mut(add_id).unwrap();
//     let y = f.call_mut(vec![y[0], c_id], &mut variables, false);

//     variables.backward(y.clone(), &mut functions, false);

//     let y = variables.get(y[0]).unwrap().get_variable();
//     let y = match y {
//         VariableType::F64(y) => y.data(),
//     };
//     assert_eq!(y, &Tensor::new_from_num_vec(vec![7.0], vec![]));
//     println!("y: {:?}", y);

//     let a = variables.get_mut(a_id).unwrap().get_variable();
//     let a_grad = match a {
//         VariableType::F64(a) => a.grad().unwrap(),
//     };
//     assert_eq!(a_grad, &Tensor::new_from_num_vec(vec![2.0], vec![]));
//     println!("a grad: {:?}", a_grad);

//     let b = variables.get_mut(b_id).unwrap().get_variable();
//     let b_grad = match b {
//         VariableType::F64(b) => b.grad().unwrap(),
//     };
//     assert_eq!(b_grad, &Tensor::new_from_num_vec(vec![3.0], vec![]));
//     println!("b grad: {:?}", b_grad);
// }

// #[test]
// fn step26() {
//     use rust_dezero::{
//         Tensor,
//         variable::{VariableTable, Variable, VariableWrapper},
//         function::{FunctionTable, sample::{Add, Mul}},
//     };

//     let mut variables = VariableTable::new();
//     let mut functions = FunctionTable::new();

//     let add = Add::new();
//     let mul = Mul::new();

//     let add_id = functions.add_function(Box::new(add));
//     let mul_id = functions.add_function(Box::new(mul));

//     let data = Tensor::new_from_num_vec(vec![3.0], vec![]);
//     let a = VariableWrapper::from_variable_f64(Variable::new(data), Some("a"));
//     let a_id = variables.add(Box::new(a));

//     let data = Tensor::new_from_num_vec(vec![2.0], vec![]);
//     let b = VariableWrapper::from_variable_f64(Variable::new(data), Some("b"));
//     let b_id = variables.add(Box::new(b));

//     let data = Tensor::new_from_num_vec(vec![1.0], vec![]);
//     let c = VariableWrapper::from_variable_f64(Variable::new(data), Some("c"));
//     let c_id = variables.add(Box::new(c));

//     let f = functions.get_mut(mul_id).unwrap();
//     let y = f.call_mut(vec![a_id, b_id], &mut variables, false);

//     let f = functions.get_mut(add_id).unwrap();
//     let y = f.call_mut(vec![y[0], c_id], &mut variables, false);

//     variables.plot_dot_graph(y, &functions, "output/sample", true);
// }

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
