extern crate kdezero;
extern crate ktensor;

#[test]
fn step1() {
    use ktensor::Tensor;
    use kdezero::variable::Variable;

    let tensor = Tensor::new(vec![1.0], vec![])
        .unwrap();
    let mut x = Variable::new(
        0, None, tensor.into()
    );
    assert_eq!(x.get_id(), 0);
    assert_eq!(x.get_node(), None);
    assert_eq!(x.get_data().to_string(), "F64");
    assert_eq!(x.get_data(), &Tensor::new(vec![1.0], vec![]).unwrap().into());
    println!("x: {:?}", x);
    x.set_data(
        Tensor::new(vec![2.0], vec![]).unwrap().into()
    );
    assert_eq!(x.get_data().to_string(), "F64");
    assert_eq!(x.get_data(), &Tensor::new(vec![2.0], vec![]).unwrap().into());
    println!("x: {:?}", x);
    x.set_data(
        Tensor::<f32>::new(vec![3.0], vec![]).unwrap().into()
    );
    assert_eq!(x.get_data().to_string(), "F32");
    assert_eq!(x.get_data(), &Tensor::<f32>::new(vec![3.0], vec![]).unwrap().into());
    println!("x: {:?}", x);
}

#[test]
fn step2() {
    use ktensor::Tensor;
    use kdezero::{
        node::NodeData,
        variable::VariableData,
        operator::operator_contents::Square,
        model::Model,
    };

    let tensor = Tensor::new(vec![10.0], vec![])
        .unwrap();
    let mut model = Model::new();
    model.add_new_node(
        0, "in".to_string(), NodeData::Variable(0), vec![], vec![1]
    ).unwrap();
    model.add_new_node(
        1, "op".to_string(), NodeData::Operator(0), vec![0], vec![2]
    ).unwrap();
    model.add_new_node(
        2, "out".to_string(), NodeData::Variable(1), vec![1], vec![]
    ).unwrap();
    model.add_new_variable(
        0, Some(0), tensor.into()
    ).unwrap();
    model.add_new_variable(
        1, Some(2), VariableData::None
    ).unwrap();
    model.add_new_operator(
        0, Some(1), vec![], Box::new(Square {})
    ).unwrap();
    model.forward().unwrap();
    let input_variable = model.get_variables().get_variable(0).unwrap();
    let output_variable = model.get_variables().get_variable(1).unwrap();
    assert_eq!(output_variable.get_data().to_string(), "F64");
    assert_eq!(output_variable.get_data(), &Tensor::new(vec![100.0], vec![]).unwrap().into());
    println!("input variable: {:?}", input_variable);
    println!("output variable: {:?}", output_variable);
}

#[test]
fn step2_2() {
    use ktensor::Tensor;
    use kdezero::{
        variable::VariableData,
        model::{Model, ModelVariable, ModelOperator},
    };

    let tensor = Tensor::new(vec![10.0], vec![])
        .unwrap();
    let mut model = Model::make_model(
        vec![ModelVariable::new(
                "in", tensor.into()
        )],
        vec![ModelVariable::new(
                "out", VariableData::None
        )],
        vec![ModelOperator::new(
                "op", Box::new(kdezero::operator::operator_contents::Square {}),
                vec!["in"], vec!["out"], vec![]
        )],
        vec![]
    ).unwrap();
    model.forward().unwrap();
    let input_variable = model.get_variable_from_name("in").unwrap();
    let output_variable = model.get_variable_from_name("out").unwrap();
    assert_eq!(output_variable.get_data().to_string(), "F64");
    assert_eq!(output_variable.get_data(), &Tensor::new(vec![100.0], vec![]).unwrap().into());
    println!("input variable: {:?}", input_variable);
    println!("output variable: {:?}", output_variable);
}

#[test]
fn step2_3() {
    use ktensor::Tensor;
    use kdezero::{
        variable::VariableData,
        model::{Model, ModelVariable, ModelOperator},
    };

    let tensor = Tensor::new(vec![10.0], vec![])
        .unwrap();
    let mut model = Model::make_model(
        vec![ModelVariable::new(
                "in", tensor.into()
        )],
        vec![ModelVariable::new(
                "out", VariableData::None
        )],
        vec![ModelOperator::new(
                "op", Box::new(kdezero::operator::operator_contents::Square {}),
                vec!["in"], vec!["out"], vec![]
        )],
        vec![]
    ).unwrap();
    model.forward().unwrap();
    let input_variable = model.get_variable_from_name("in").unwrap();
    let output_variable = model.get_variable_from_name("out").unwrap();
    assert_eq!(output_variable.get_data().to_string(), "F64");
    assert_eq!(output_variable.get_data(), &Tensor::new(vec![100.0], vec![]).unwrap().into());
    println!("input variable: {:?}", input_variable);
    println!("output variable: {:?}", output_variable);
    let output_id = model.get_node_id_from_name("out").unwrap();
    model.backward(output_id).unwrap();
    let input_grad_variable = model.get_grad_from_variable_name("in").unwrap();
    let output_grad_variable = model.get_grad_from_variable_name("out").unwrap();
    assert_eq!(input_grad_variable.get_data().to_string(), "F64");
    assert_eq!(input_grad_variable.get_data(), &Tensor::new(vec![20.0], vec![]).unwrap().into());
    assert_eq!(output_grad_variable.get_data().to_string(), "F64");
    assert_eq!(output_grad_variable.get_data(), &Tensor::new(vec![1.0], vec![]).unwrap().into());
    println!("input grad variable: {:?}", input_grad_variable);
    println!("output grad variable: {:?}", output_grad_variable);
}

#[test]
fn step3() {
    use ktensor::Tensor;
    use kdezero::{
        operator::operator_contents::{Square, Exp},
        variable::VariableData,
        model::{Model, ModelVariable, ModelOperator},
    };

    let tensor = Tensor::new(vec![0.5], vec![])
        .unwrap();
    let mut model = Model::make_model(
        vec![ModelVariable::new(
                "in", tensor.into()
        )],
        vec![ModelVariable::new(
                "out", VariableData::None
        )],
        vec![ModelOperator::new(
                "op1", Box::new(Square {}),
                vec!["in"], vec!["out1"], vec![]
        ), ModelOperator::new(
                "op2", Box::new(Exp {}),
                vec!["out1"], vec!["out2"], vec![]
        ), ModelOperator::new(
                "op3", Box::new(Square {}),
                vec!["out2"], vec!["out"], vec![]
        )],
        vec![]
    ).unwrap();
    model.forward().unwrap();
    let input_variable = model.get_variable_from_name("in").unwrap();
    let output_variable = model.get_variable_from_name("out").unwrap();
    assert_eq!(output_variable.get_data().to_string(), "F64");
    assert_eq!(output_variable.get_data(), &Tensor::new(vec![1.648721270700128], vec![]).unwrap().into());
    println!("input variable: {:?}", input_variable);
    println!("output variable: {:?}", output_variable);
}

#[test]
fn step4() {
    use ktensor::Tensor;
    use kdezero::test_utility::{
        assert_approx_eq_tensor,
        numerical_diff,
    };

    let tensor = Tensor::new(vec![0.5], vec![])
        .unwrap();
    let mut f =
        |x: &Tensor<f64>| x.powi(2).exp().powi(2);
    let dy = numerical_diff(&mut f, &tensor, 1e-4);
    assert_approx_eq_tensor(
        &dy, &Tensor::new(vec![3.29744], vec![]).unwrap(), 1e-4);
}

#[test]
fn step7() {
    use ktensor::Tensor;
    use kdezero::{
        operator::operator_contents::{Square, Exp},
        variable::VariableData,
        model::{Model, ModelVariable, ModelOperator},
    };

    let tensor = Tensor::new(vec![0.5], vec![])
        .unwrap();
    let mut model = Model::make_model(
        vec![ModelVariable::new(
                "in", tensor.into()
        )],
        vec![ModelVariable::new(
                "out", VariableData::None
        )],
        vec![ModelOperator::new(
                "op1", Box::new(Square {}),
                vec!["in"], vec!["out1"], vec![]
        ), ModelOperator::new(
                "op2", Box::new(Exp {}),
                vec!["out1"], vec!["out2"], vec![]
        ), ModelOperator::new(
                "op3", Box::new(Square {}),
                vec!["out2"], vec!["out"], vec![]
        )],
        vec![]
    ).unwrap();
    model.forward().unwrap();
    let input_variable = model.get_variable_from_name("in").unwrap();
    let output_variable = model.get_variable_from_name("out").unwrap();
    println!("input variable: {:?}", input_variable);
    println!("output variable: {:?}", output_variable);
    let output_id = model.get_node_id_from_name("out").unwrap();
    model.backward(output_id).unwrap();
    let input_grad_variable = model.get_grad_from_variable_name("in").unwrap();
    assert_eq!(input_grad_variable.get_data().to_string(), "F64");
    assert_eq!(input_grad_variable.get_data(), &Tensor::new(vec![3.297442541400256], vec![]).unwrap().into());
    println!("input grad variable: {:?}", input_grad_variable);
}

#[test]
fn step11() {
    use ktensor::Tensor;
    use kdezero::{
        operator::operator_contents::Add,
        variable::VariableData,
        model::{Model, ModelVariable, ModelOperator},
    };

    let tensor0 = Tensor::new(vec![2.0], vec![])
        .unwrap();
    let tensor1 = Tensor::new(vec![3.0], vec![])
        .unwrap();
    let mut model = Model::make_model(
        vec![
            ModelVariable::new("in0", tensor0.into()),
            ModelVariable::new("in1", tensor1.into())
        ],
        vec![ModelVariable::new(
                "out", VariableData::None
        )],
        vec![ModelOperator::new(
                "op1", Box::new(Add {}),
                vec!["in0", "in1"], vec!["out"], vec![]
        )], vec![]
    ).unwrap();
    model.forward().unwrap();
    let input_variable0 = model.get_variable_from_name("in0").unwrap();
    let input_variable1 = model.get_variable_from_name("in1").unwrap();
    let output_variable = model.get_variable_from_name("out").unwrap();
    assert_eq!(output_variable.get_data().to_string(), "F64");
    assert_eq!(output_variable.get_data(), &Tensor::new(vec![5.0], vec![]).unwrap().into());
    println!("input variable0: {:?}", input_variable0);
    println!("input variable1: {:?}", input_variable1);
    println!("output variable: {:?}", output_variable);
}

#[test]
fn step13() {
    use ktensor::Tensor;
    use kdezero::{
        operator::operator_contents::{Add, Square},
        variable::VariableData,
        model::{Model, ModelVariable, ModelOperator},
    };

    let tensor0 = Tensor::new(vec![2.0], vec![])
        .unwrap();
    let tensor1 = Tensor::new(vec![3.0], vec![])
        .unwrap();
    let mut model = Model::make_model(
        vec![
            ModelVariable::new("in0", tensor0.into()),
            ModelVariable::new("in1", tensor1.into()),
        ],
        vec![ModelVariable::new(
                "out", VariableData::None
        )],
        vec![
            ModelOperator::new(
                "op0", Box::new(Square {}),
                vec!["in0"], vec!["square0"], vec![]
            ), ModelOperator::new(
                "op1", Box::new(Square {}),
                vec!["in1"], vec!["square1"], vec![]
            ), ModelOperator::new(
                "op2", Box::new(Add {}),
                vec!["square0", "square1"], vec!["out"], vec![]
            )
        ], vec![]
    ).unwrap();
    model.forward().unwrap();
    let input_variable0 = model.get_variable_from_name("in0").unwrap();
    let input_variable1 = model.get_variable_from_name("in1").unwrap();
    let output_variable = model.get_variable_from_name("out").unwrap();
    assert_eq!(output_variable.get_data().to_string(), "F64");
    assert_eq!(output_variable.get_data(), &Tensor::new(vec![13.0], vec![]).unwrap().into());
    println!("input variable0: {:?}", input_variable0);
    println!("input variable1: {:?}", input_variable1);
    println!("output variable: {:?}", output_variable);
    let output_id = model.get_node_id_from_name("out").unwrap();
    model.backward(output_id).unwrap();
    let input_grad_variable = model.get_grad_from_variable_name("in0").unwrap();
    assert_eq!(input_grad_variable.get_data().to_string(), "F64");
    assert_eq!(input_grad_variable.get_data(), &Tensor::new(vec![4.0], vec![]).unwrap().into());
    println!("input grad variable0: {:?}", input_grad_variable);
    let input_grad_variable = model.get_grad_from_variable_name("in1").unwrap();
    assert_eq!(input_grad_variable.get_data().to_string(), "F64");
    assert_eq!(input_grad_variable.get_data(), &Tensor::new(vec![6.0], vec![]).unwrap().into());
    println!("input grad variable1: {:?}", input_grad_variable);
}

#[test]
fn step14() {
    use ktensor::Tensor;
    use kdezero::{
        operator::operator_contents::Add,
        variable::VariableData,
        model::{Model, ModelVariable, ModelOperator},
    };

    let tensor = Tensor::new(vec![3.0], vec![])
        .unwrap();
    let mut model = Model::make_model(
        vec![ModelVariable::new(
                "in", tensor.into()
        )],
        vec![ModelVariable::new(
                "out", VariableData::None
        )],
        vec![ModelOperator::new(
                "op1", Box::new(Add {}),
                vec!["in", "in"], vec!["out"], vec![]
        )], vec![]
    ).unwrap();
    model.forward().unwrap();
    let input_variable = model.get_variable_from_name("in").unwrap();
    let output_variable = model.get_variable_from_name("out").unwrap();
    assert_eq!(output_variable.get_data().to_string(), "F64");
    assert_eq!(output_variable.get_data(), &Tensor::new(vec![6.0], vec![]).unwrap().into());
    println!("input variable: {:?}", input_variable);
    println!("output variable: {:?}", output_variable);
    let output_id = model.get_node_id_from_name("out").unwrap();
    model.backward(output_id).unwrap();
    let input_grad_variable = model.get_grad_from_variable_name("in").unwrap();
    assert_eq!(input_grad_variable.get_data().to_string(), "F64");
    assert_eq!(input_grad_variable.get_data(), &Tensor::new(vec![2.0], vec![]).unwrap().into());
    println!("input grad variable: {:?}", input_grad_variable);
}

#[test]
fn step16() {
    use ktensor::Tensor;
    use kdezero::{
        operator::operator_contents::{Add, Square},
        variable::VariableData,
        model::{Model, ModelVariable, ModelOperator},
    };

    let tensor = Tensor::new(vec![2.0], vec![])
        .unwrap();
    let mut model = Model::make_model(
        vec![ModelVariable::new(
                "in", tensor.into()
        )],
        vec![ModelVariable::new(
                "out", VariableData::None
        )],
        vec![
            ModelOperator::new(
                "op0", Box::new(Square {}),
                vec!["in"], vec!["square0"], vec![]
            ), ModelOperator::new(
                "op1", Box::new(Square {}),
                vec!["square0"], vec!["square1"], vec![]
            ), ModelOperator::new(
                "op2", Box::new(Square {}),
                vec!["square0"], vec!["square2"], vec![]
            ), ModelOperator::new(
                "op3", Box::new(Add {}),
                vec!["square1", "square2"], vec!["out"], vec![]
            )
        ], vec![]
    ).unwrap();
    model.forward().unwrap();
    let input_variable = model.get_variable_from_name("in").unwrap();
    let output_variable = model.get_variable_from_name("out").unwrap();
    assert_eq!(output_variable.get_data().to_string(), "F64");
    assert_eq!(output_variable.get_data(), &Tensor::new(vec![32.0], vec![]).unwrap().into());
    println!("input variable: {:?}", input_variable);
    println!("output variable: {:?}", output_variable);
    let output_id = model.get_node_id_from_name("out").unwrap();
    model.backward(output_id).unwrap();
    let input_grad_variable = model.get_grad_from_variable_name("in").unwrap();
    assert_eq!(input_grad_variable.get_data().to_string(), "F64");
    assert_eq!(input_grad_variable.get_data(), &Tensor::new(vec![64.0], vec![]).unwrap().into());
    println!("input grad variable: {:?}", input_grad_variable);
}

#[test]
fn step20() {
    use ktensor::Tensor;
    use kdezero::{
        operator::operator_contents::{Add, Mul},
        variable::VariableData,
        model::{Model, ModelVariable, ModelOperator},
    };

    let tensor0 = Tensor::new(vec![3.0], vec![])
        .unwrap();
    let tensor1 = Tensor::new(vec![2.0], vec![])
        .unwrap();
    let tensor2 = Tensor::new(vec![1.0], vec![])
        .unwrap();
    let mut model = Model::make_model(
        vec![
            ModelVariable::new("in0", tensor0.into()),
            ModelVariable::new("in1", tensor1.into()),
            ModelVariable::new("in2", tensor2.into()),
        ],
        vec![ModelVariable::new(
                "out", VariableData::None
        )],
        vec![
            ModelOperator::new(
                "op0", Box::new(Mul {}),
                vec!["in0", "in1"], vec!["add0"], vec![]
            ), ModelOperator::new(
                "op1", Box::new(Add {}),
                vec!["add0", "in2"], vec!["out"], vec![]
            )
        ], vec![]
    ).unwrap();
    model.forward().unwrap();
    let output_id = model.get_node_id_from_name("out").unwrap();
    model.backward(output_id).unwrap();

    let output_variable = model.get_variable_from_name("out").unwrap();
    assert_eq!(output_variable.get_type(), "F64");
    assert_eq!(output_variable.get_data(), &Tensor::new(vec![7.0], vec![]).unwrap().into());
    println!("output variable: {:?}", output_variable);

    let input_grad_variable = model.get_grad_from_variable_name("in0").unwrap();
    assert_eq!(input_grad_variable.get_type(), "F64");
    assert_eq!(input_grad_variable.get_data(), &Tensor::new(vec![2.0], vec![]).unwrap().into());
    println!("input grad variable: {:?}", input_grad_variable);

    let input_grad_variable = model.get_grad_from_variable_name("in1").unwrap();
    assert_eq!(input_grad_variable.get_type(), "F64");
    assert_eq!(input_grad_variable.get_data(), &Tensor::new(vec![3.0], vec![]).unwrap().into());
    println!("input grad variable: {:?}", input_grad_variable);
}

#[test]
fn step26() {
    use std::fs::create_dir;
    use ktensor::Tensor;
    use kdezero::{
        operator::operator_contents::{Add, Mul},
        variable::VariableData,
        model::{Model, ModelVariable, ModelOperator},
    };

    let tensor0 = Tensor::new(vec![3.0], vec![])
        .unwrap();
    let tensor1 = Tensor::new(vec![2.0], vec![])
        .unwrap();
    let tensor2 = Tensor::new(vec![1.0], vec![])
        .unwrap();
    let model = Model::make_model(
        vec![
            ModelVariable::new("in0", tensor0.into()),
            ModelVariable::new("in1", tensor1.into()),
            ModelVariable::new("in2", tensor2.into()),
        ],
        vec![ModelVariable::new(
                "out", VariableData::None
        )],
        vec![
            ModelOperator::new(
                "op0", Box::new(Mul {}),
                vec!["in0", "in1"], vec!["add0"], vec![]
            ), ModelOperator::new(
                "op1", Box::new(Add {}),
                vec!["add0", "in2"], vec!["out"], vec![]
            )
        ], vec![]
    ).unwrap();

    match create_dir("output") {
        Ok(_) => println!("create output directory"),
        Err(_) => {},
    }
    model.plot_dot_graph("output/step26", true).unwrap();
}

#[test]
fn step27() {
    use ktensor::Tensor;
    use kdezero::{
        operator::operator_contents::Sin,
        variable::VariableData,
        model::{Model, ModelVariable, ModelOperator},
    };

    let tensor0 = Tensor::new(vec![
        std::f64::consts::FRAC_PI_4,
    ], vec![]).unwrap();

    let mut model = Model::make_model(
        vec![ModelVariable::new(
                "in", tensor0.into()
        )],
        vec![ModelVariable::new(
                "out", VariableData::None
        )],
        vec![
            ModelOperator::new(
                "op0", Box::new(Sin {}),
                vec!["in"], vec!["out"], vec![]
            )
        ], vec![]
    ).unwrap();

    model.forward().unwrap();
    let output_id = model.get_node_id_from_name("out").unwrap();
    model.backward(output_id).unwrap();

    let output_variable = model.get_variable_from_name("out").unwrap();
    assert_eq!(output_variable.get_type(), "F64");
    assert_eq!(output_variable.get_data(), &Tensor::new(vec![
        std::f64::consts::FRAC_PI_4.sin(),
    ], vec![]).unwrap().into());
    println!("output variable: {:?}", output_variable);

    let input_grad_variable = model.get_grad_from_variable_name("in").unwrap();
    assert_eq!(input_grad_variable.get_type(), "F64");
    assert_eq!(input_grad_variable.get_data(), &Tensor::new(vec![
        std::f64::consts::FRAC_PI_4.cos(),
    ], vec![]).unwrap().into());
    println!("input grad variable: {:?}", input_grad_variable);
}

#[test]
fn step33() {
    use ktensor::Tensor;
    use kdezero::{
        operator::operator_contents::{Sub, Pow, ScalarMul},
        variable::VariableData,
        model::{Model, ModelVariable, ModelOperator},
        test_utility::assert_approx_eq_tensor,
    };

    fn f(x: Tensor<f64>) -> Model {
        Model::make_model(
            vec![ModelVariable::new(
                    "x", x.into()
            )],
            vec![ModelVariable::new(
                    "y", VariableData::None
            )],
            vec![
                ModelOperator::new(
                    "op0", Box::new(Pow {}),
                    vec!["x"], vec!["x4"],
                    vec![Tensor::scalar(4usize).into()]
                ), ModelOperator::new(
                    "op1", Box::new(Pow {}),
                    vec!["x"], vec!["x2"],
                    vec![Tensor::scalar(2usize).into()]
                ), ModelOperator::new(
                    "op2", Box::new(ScalarMul {}),
                    vec!["x2"], vec!["x2_2"],
                    vec![Tensor::scalar(2.0).into()]
                ), ModelOperator::new(
                    "op3", Box::new(Sub {}),
                    vec!["x4", "x2_2"], vec!["y"], vec![]
                )
            ], vec![]
        ).unwrap()
    }

    let mut x = Tensor::scalar(2.0);
    let iters = 10;
    for i in 0..iters {
        println!("iter: {}, x: {:?}", i, x);

        let mut model = f(x.clone());
        model.forward().unwrap();
        let output_id = model.get_node_id_from_name("y").unwrap();
        model.backward(output_id).unwrap();

        let input_id = model.get_node_id_from_name("x").unwrap();
        let grad_id = model.get_grad_id_from_node_id(input_id).unwrap();
        let grad_model = model.get_grad_model_mut();
        grad_model.backward(grad_id).unwrap();

        let grad_model = model.get_grad_model().unwrap();
        let input_grad = model.get_grad_from_variable_name("x").unwrap()
            .get_data().to_f64_tensor().unwrap();
        let input_grad_grad = grad_model.get_grad_from_variable_name("x").unwrap()
            .get_data().to_f64_tensor().unwrap();
        x = x - input_grad / input_grad_grad;
    }

    assert_approx_eq_tensor(
        &x, &Tensor::new(vec![1.0], vec![]).unwrap(), 1e-4);
}
