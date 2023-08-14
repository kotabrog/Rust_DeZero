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
                "in".to_string(), tensor.into()
        )],
        vec![ModelVariable::new(
                "out".to_string(), VariableData::None
        )],
        vec![ModelOperator::new(
                "op".to_string(), Box::new(kdezero::operator::operator_contents::Square {}),
                vec!["in".to_string()], vec!["out".to_string()], vec![]
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
                "in".to_string(), tensor.into()
        )],
        vec![ModelVariable::new(
                "out".to_string(), VariableData::None
        )],
        vec![ModelOperator::new(
                "op".to_string(), Box::new(kdezero::operator::operator_contents::Square {}),
                vec!["in".to_string()], vec!["out".to_string()], vec![]
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
    model.backward().unwrap();
    let input_grad_variable = model.get_grad_from_variable_name("in").unwrap();
    let output_grad_variable = model.get_grad_from_variable_name("out").unwrap();
    assert_eq!(input_grad_variable.get_data().to_string(), "F64");
    assert_eq!(input_grad_variable.get_data(), &Tensor::new(vec![20.0], vec![]).unwrap().into());
    assert_eq!(output_grad_variable.get_data().to_string(), "F64");
    assert_eq!(output_grad_variable.get_data(), &Tensor::new(vec![1.0], vec![]).unwrap().into());
    println!("input grad variable: {:?}", input_grad_variable);
    println!("output grad variable: {:?}", output_grad_variable);
}
