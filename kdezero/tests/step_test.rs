extern crate kdezero;

#[test]
fn step1() {
    use ktensor::Tensor;
    use kdezero::variable::{VariableContents, VariableTable};

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
    use ktensor::Tensor;
    use kdezero::{
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
    use ktensor::Tensor;
    use kdezero::{
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
    use ktensor::{Tensor, utility::{numerical_diff, assert_approx_eq}};
    use kdezero::{
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
    use ktensor::{Tensor, utility::{numerical_diff, assert_approx_eq}};
    use kdezero::{
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
    use ktensor::Tensor;
    use kdezero::{
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
    use ktensor::Tensor;
    use kdezero::{
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
    use ktensor::Tensor;
    use kdezero::{
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
    use ktensor::Tensor;
    use kdezero::{
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
    use ktensor::Tensor;
    use kdezero::{
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
    use ktensor::Tensor;
    use kdezero::{
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
    use ktensor::Tensor;
    use kdezero::{
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
    use ktensor::Tensor;
    use kdezero::{
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
    use ktensor::Tensor;
    use kdezero::{
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
    variable_table.plot_dot_graph(y_id, &function_table, "output/step26", true);
}

#[test]
fn step27() {
    use ktensor::Tensor;
    use kdezero::{
        variable::VariableTable,
        function::{FunctionTable, operator::Sin},
    };

    let mut variable_table = VariableTable::new();
    let mut function_table = FunctionTable::new();

    let data = vec![std::f64::consts::FRAC_PI_4];
    let sin_id = function_table.generate_function_from_function_contents(Box::new(Sin::new()));
    let x_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data.clone(), vec![]), "x");

    let y_id = function_table.forward(sin_id, vec![x_id], &mut variable_table, false)[0];

    let y = variable_table.get_variable_contents_f64(y_id).unwrap();
    println!("y: {:?}", y);
    assert_eq!(*y.data()[0].data(), data[0].sin());

    variable_table.backward(vec![y_id], &mut function_table, false);

    let x_grad = variable_table.get_variable_grad_contents_f64(x_id).unwrap();
    println!("x_grad: {:?}", x_grad);
    assert_eq!(x_grad.data()[0].data(), &data[0].cos());
}

#[test]
fn step33() {
    use ktensor::Tensor;
    use kdezero::{
        variable::VariableTable,
        function::{FunctionTable, operator::{Mul, Sub, Pow}},
    };

    fn f(tensor: Tensor<f64>) -> (VariableTable, FunctionTable, usize, usize) {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let x_id = variable_table.generate_variable_from_f64_tensor(tensor, "x");
        let const_id = variable_table.generate_variable_from_f64_tensor(
            Tensor::new_from_num_vec(vec![2.0], vec![]), "const");

        let f_id = function_table.generate_function_from_function_contents(Box::new(Pow::new(4.0)));
        let temp0_id = function_table.forward(f_id, vec![x_id], &mut variable_table, false)[0];

        let f_id = function_table.generate_function_from_function_contents(Box::new(Pow::new(2.0)));
        let temp1_id = function_table.forward(f_id, vec![x_id], &mut variable_table, false)[0];

        let f_id = function_table.generate_function_from_function_contents(Box::new(Mul::new()));
        let temp1_id = function_table.forward(f_id, vec![const_id, temp1_id], &mut variable_table, false)[0];

        let f_id = function_table.generate_function_from_function_contents(Box::new(Sub::new()));
        let y_id = function_table.forward(f_id, vec![temp0_id, temp1_id], &mut variable_table, false)[0];

        (variable_table, function_table, x_id, y_id)
    }

    let mut x_data = Tensor::new_from_num_vec(vec![2.0], vec![]);
    let iters = 10;

    for i in 0..iters {
        let (mut variable_table, mut function_table, x_id, y_id) = f(x_data.clone());
        let y_data = variable_table.get_variable_contents_f64(y_id).unwrap().clone();

        variable_table.backward(vec![y_id], &mut function_table, false);

        let gx = variable_table.get_variable_grad_contents_f64(x_id).unwrap().clone();

        let gx_id = variable_table.get_variable_grad_id(x_id).unwrap();
        variable_table.clear_grad(x_id);
        variable_table.backward(vec![gx_id], &mut function_table, false);

        let gx2 = variable_table.get_variable_grad_contents_f64(x_id).unwrap().clone();

        x_data -= &(&gx / &gx2);
        println!("iter: {}, x: {:?}, y: {:?}", i, x_data, y_data);
    }
}

#[test]
fn step35() {
    use std::fs::create_dir;
    use ktensor::Tensor;
    use kdezero::{
        variable::VariableTable,
        function::{FunctionTable, operator::Tanh},
    };
    let iters = 0;

    let mut variable_table = VariableTable::new();
    let mut function_table = FunctionTable::new();

    let data = vec![1.0];
    let tanh_id = function_table.generate_function_from_function_contents(Box::new(Tanh::new()));
    let x_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data.clone(), vec![]), "x");

    let y_id = function_table.forward(tanh_id, vec![x_id], &mut variable_table, false)[0];
    variable_table.set_variable_name(y_id, "y");

    variable_table.backward(vec![y_id], &mut function_table, false);

    for _ in 0..iters {
        let gx_id = variable_table.get_variable_grad_id(x_id).unwrap();
        variable_table.clear_grad(x_id);
        variable_table.backward(vec![gx_id], &mut function_table, false);
    }

    let gx_id = variable_table.get_variable_grad_id(x_id).unwrap();
    variable_table.set_variable_name(gx_id, format!("gx{}", iters + 1).as_str());

    match create_dir("output") {
        Ok(_) => println!("create output directory"),
        Err(_) => {},
    }
    variable_table.plot_dot_graph(vec![gx_id], &function_table, "output/step35", true);
}

#[test]
fn step36() {
    use ktensor::Tensor;
    use kdezero::{
        variable::VariableTable,
        function::{FunctionTable, operator::{Pow, Add}},
    };
    let mut variable_table = VariableTable::new();
    let mut function_table = FunctionTable::new();

    let data = vec![2.0];
    let x_id = variable_table.generate_variable_from_f64_tensor(
        Tensor::new_from_num_vec(data.clone(), vec![]), "x");

    let pow_id = function_table.generate_function_from_function_contents(Box::new(Pow::new(2.0)));
    let y_id = function_table.forward(pow_id, vec![x_id], &mut variable_table, false)[0];

    variable_table.backward(vec![y_id], &mut function_table, false);

    let gx_id = variable_table.get_variable_grad_id(x_id).unwrap();
    variable_table.clear_grad(x_id);

    let pow_id = function_table.generate_function_from_function_contents(Box::new(Pow::new(3.0)));
    let z_id = function_table.forward(pow_id, vec![gx_id], &mut variable_table, false)[0];
    let add_id = function_table.generate_function_from_function_contents(Box::new(Add::new()));
    let z_id = function_table.forward(add_id, vec![z_id, y_id], &mut variable_table, false)[0];

    variable_table.backward(vec![z_id], &mut function_table, false);

    let gx = variable_table.get_variable_grad_contents_f64(x_id).unwrap();
    println!("gx: {:?}", gx);
    assert_eq!(gx.data()[0].data(), &100.0);
}

#[test]
fn step42() {
    use std::fs::create_dir;
    use plotters::prelude::*;
    use ktensor::{Tensor, tensor::random::TensorRng};
    use kdezero::{
        variable::VariableTable,
        function::{FunctionTable, operator::{
            MeanSquaredError, Add, BroadcastTo, MatMul,
        }},
    };

    fn predict(x: &Tensor<f64>, w: &Tensor<f64>, b: &Tensor<f64>)
            -> (VariableTable, FunctionTable, usize, usize, usize) {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let x_id = variable_table.generate_variable_from_f64_tensor(x.clone(), "x");
        let w_id = variable_table.generate_variable_from_f64_tensor(w.clone(), "");
        let b_id = variable_table.generate_variable_from_f64_tensor(b.clone(), "");
        let x_shape = variable_table
            .get(x_id).expect("Invalid variable id")
            .shape().clone();
        let matmul_id = function_table.generate_function_from_function_contents(Box::new(MatMul::new()));
        let temp_id0 = function_table.forward(matmul_id, vec![x_id, w_id], &mut variable_table, false)[0];
        let broadcast_to_id = function_table.generate_function_from_function_contents(Box::new(BroadcastTo::new(vec![x_shape[0], 1])));
        let temp_id1 = function_table.forward(broadcast_to_id, vec![b_id], &mut variable_table, false)[0];
        let add_id = function_table.generate_function_from_function_contents(Box::new(Add::new()));
        let ret_id = function_table.forward(add_id, vec![temp_id0, temp_id1], &mut variable_table, false)[0];
        (variable_table, function_table, ret_id, w_id, b_id)
    }
    let mut rng = TensorRng::new();

    let x = rng.gen::<f64, _>(&[100, 1]);
    let y = (
            x.scalar_mul(2.0.into())
        ).scalar_add(5.0.into())
        + rng.gen::<f64, _>(&[100, 1]);

    let mut w = Tensor::new_from_num_vec(vec![0.0], vec![1, 1]);
    let mut b = Tensor::new_from_num_vec(vec![0.0], vec![]);

    let lr = 0.1;
    let iters = 100;

    for i in 0..iters {
        let (mut variable_table, mut function_table, pred_id, w_id, b_id)
            = predict(&x, &w, &b);

        let y_id = variable_table.generate_variable_from_f64_tensor(y.clone(), "y");

        let mse_id = function_table.generate_function_from_function_contents(Box::new(MeanSquaredError::new()));
        let loss_id = function_table.forward(mse_id, vec![y_id, pred_id], &mut variable_table, false)[0];

        variable_table.clear_grads(&vec![w_id, b_id]);
        variable_table.backward(vec![loss_id], &mut function_table, false);

        let w_grad = variable_table
            .get_variable_grad_contents_f64(w_id).expect("Invalid variable id");
        let b_grad = variable_table
            .get_variable_grad_contents_f64(b_id).expect("Invalid variable id");

        w = w - w_grad.scalar_mul(lr.into());
        b = b - b_grad.scalar_mul(lr.into());

        let loss = variable_table
            .get_variable_contents_f64(loss_id).expect("Invalid variable id");

        if i % 10 == 0 {
            println!("iter {} w: {:?}, b: {:?}, loss: {:?}", i, w, b, loss)
        }
    }

    match create_dir("output") {
        Ok(_) => println!("create output directory"),
        Err(_) => {},
    }

    fn plot(data: &[(f64, f64)], file_name: &str, w: f64, b: f64)
            -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(file_name, (640, 480)).into_drawing_area();
        root.fill(&WHITE)?;

        let x_max = data.iter().map(|&(x, _)| x).fold(f64::NEG_INFINITY, f64::max);
        let x_min = data.iter().map(|&(x, _)| x).fold(f64::INFINITY, f64::min);
        let y_max = data.iter().map(|&(_, y)| y).fold(f64::NEG_INFINITY, f64::max);
        let y_min = data.iter().map(|&(_, y)| y).fold(f64::INFINITY, f64::min);

        let x_margin = (x_max - x_min) * 0.05;
        let y_margin = (y_max - y_min) * 0.05;
    
        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                (x_min - x_margin)..(x_max + x_margin),
                (y_min - y_margin)..(y_max + y_margin)
            )?;

        chart.configure_mesh().draw()?;
    
        let shape_style = ShapeStyle::from(&BLUE).filled();
        chart.draw_series(data.iter().map(|&(x, y)| Circle::new((x, y), 5, shape_style)))?;

        let line_points: Vec<(f64, f64)> = (0..=100)
            .map(|x| x as f64 / 100.0 * (x_max - x_min) + x_min)
            .map(|x| (x, w * x + b))
            .collect();
        chart.draw_series(LineSeries::new(line_points, &RED))?;

        root.present()?;

        Ok(())
    }

    let x_data = x.data().iter().map(|x| *x.data()).collect::<Vec<f64>>();
    let y_data = y.data().iter().map(|y| *y.data()).collect::<Vec<f64>>();
    let data = x_data.iter().zip(y_data.iter()).map(|(x, y)| (*x, *y)).collect::<Vec<(f64, f64)>>();
    let w_data = *w.at(&[0, 0]).data();
    let b_data = *b.at(&[]).data();
    plot(&data, "output/linear_regression.png", w_data, b_data).expect("Failed to plot");
}

#[test]
fn step43() {
    use std::fs::create_dir;
    use plotters::prelude::*;
    use ktensor::{Tensor, tensor::random::TensorRng};
    use kdezero::{
        variable::VariableTable,
        function::{FunctionTable, operator::{
            MeanSquaredError,
        }, function::{linear, sigmoid}},
    };

    fn predict(x: &Tensor<f64>, w1: &Tensor<f64>, b1: &Tensor<f64>, w2: &Tensor<f64>, b2: &Tensor<f64>)
            -> (VariableTable, FunctionTable, usize, usize, usize, usize, usize) {
        let mut variable_table = VariableTable::new();
        let mut function_table = FunctionTable::new();

        let x_id = variable_table.generate_variable_from_f64_tensor(x.clone(), "x");
        let w1_id = variable_table.generate_variable_from_f64_tensor(w1.clone(), "");
        let b1_id = variable_table.generate_variable_from_f64_tensor(b1.clone(), "");
        let w2_id = variable_table.generate_variable_from_f64_tensor(w2.clone(), "");
        let b2_id = variable_table.generate_variable_from_f64_tensor(b2.clone(), "");
        let y_id =
            linear(x_id, w1_id, Some(b1_id), &mut variable_table, &mut function_table);
        let y_id = sigmoid(y_id, &mut variable_table, &mut function_table);
        let y_id =
            linear(y_id, w2_id, Some(b2_id), &mut variable_table, &mut function_table);
        (variable_table, function_table, y_id, w1_id, b1_id, w2_id, b2_id)
    }

    let mut rng = TensorRng::new();

    let x = rng.gen::<f64, _>(&[100, 1]);
    let y = x.scalar_mul(2.0.into())
        .scalar_mul(std::f64::consts::PI.into())
        .sin()
        + rng.gen::<f64, _>(&[100, 1]);

    let (input_num, h_num, output_num) = (1, 10, 1);
    let mut w1 = rng.gen::<f64, _>(&[input_num, h_num]);
    let mut b1 = Tensor::full(0.0, vec![h_num]);
    let mut w2 = rng.gen::<f64, _>(&[h_num, output_num]);
    let mut b2 = Tensor::full(0.0, vec![output_num]);

    let lr = 0.2;
    let iters = 100;
    // let iters = 10000;

    for i in 0..iters {
        let (mut variable_table, mut function_table, pred_id, w1_id, b1_id, w2_id, b2_id)
            = predict(&x, &w1, &b1, &w2, &b2);

        let y_id = variable_table.generate_variable_from_f64_tensor(y.clone(), "y");

        let mse_id = function_table.generate_function_from_function_contents(Box::new(MeanSquaredError::new()));
        let loss_id = function_table.forward(mse_id, vec![y_id, pred_id], &mut variable_table, false)[0];

        variable_table.backward(vec![loss_id], &mut function_table, false);

        let w1_grad = variable_table
            .get_variable_grad_contents_f64(w1_id).expect("Invalid variable id");
        let b1_grad = variable_table
            .get_variable_grad_contents_f64(b1_id).expect("Invalid variable id");
        let w2_grad = variable_table
            .get_variable_grad_contents_f64(w2_id).expect("Invalid variable id");
        let b2_grad = variable_table
            .get_variable_grad_contents_f64(b2_id).expect("Invalid variable id");

        w1 = w1 - w1_grad.scalar_mul(lr.into());
        b1 = b1 - b1_grad.scalar_mul(lr.into());
        w2 = w2 - w2_grad.scalar_mul(lr.into());
        b2 = b2 - b2_grad.scalar_mul(lr.into());

        let loss = variable_table
            .get_variable_contents_f64(loss_id).expect("Invalid variable id");

        if i % (iters / 10) == 0 {
            println!("iter {}\nw1: {:?}\nb1: {:?}\nw2: {:?}\nb2: {:?}\nloss: {:?}", i, w1, b1, w2, b2, loss)
        }
    }

    match create_dir("output") {
        Ok(_) => println!("create output directory"),
        Err(_) => {},
    }

    fn plot(data: &[(f64, f64)], file_name: &str, line_points: Vec<(f64, f64)>)
            -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(file_name, (640, 480)).into_drawing_area();
        root.fill(&WHITE)?;

        let x_max = data.iter().map(|&(x, _)| x).fold(f64::NEG_INFINITY, f64::max);
        let x_min = data.iter().map(|&(x, _)| x).fold(f64::INFINITY, f64::min);
        let y_max = data.iter().map(|&(_, y)| y).fold(f64::NEG_INFINITY, f64::max);
        let y_min = data.iter().map(|&(_, y)| y).fold(f64::INFINITY, f64::min);

        let x_margin = (x_max - x_min) * 0.05;
        let y_margin = (y_max - y_min) * 0.05;
    
        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                (x_min - x_margin)..(x_max + x_margin),
                (y_min - y_margin)..(y_max + y_margin)
            )?;

        chart.configure_mesh().draw()?;
    
        let shape_style = ShapeStyle::from(&BLUE).filled();
        chart.draw_series(data.iter().map(|&(x, y)| Circle::new((x, y), 5, shape_style)))?;

        chart.draw_series(LineSeries::new(line_points, &RED))?;

        root.present()?;

        Ok(())
    }

    let x_data = x.data().iter().map(|x| *x.data()).collect::<Vec<f64>>();
    let y_data = y.data().iter().map(|y| *y.data()).collect::<Vec<f64>>();
    let data = x_data.iter().zip(y_data.iter()).map(|(x, y)| (*x, *y)).collect::<Vec<(f64, f64)>>();
    let line_points_x = Tensor::new_from_num_vec(
        (0..=100).map(|x| x as f64 / 100.0), vec![101, 1]);
    let (variable_table, _, y_id, _, _, _, _) = predict(&line_points_x, &w1, &b1, &w2, &b2);
    let line_points_y = variable_table.get_variable_contents_f64(y_id).expect("Invalid variable id");
    let line_points_x = line_points_x
        .data().iter().map(|x| *x.data()).collect::<Vec<f64>>();
    let line_points_y = line_points_y
        .data().iter().map(|x| *x.data()).collect::<Vec<f64>>();
    let line_points = line_points_x.into_iter()
        .zip(line_points_y.into_iter()).collect();
    plot(&data, "output/neural_network.png", line_points).expect("Failed to plot");
}
