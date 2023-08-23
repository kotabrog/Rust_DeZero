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
