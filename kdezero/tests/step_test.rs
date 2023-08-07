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
        node::{NodeData, Graph},
        variable::{VariableData, Variables},
        operator::{Operator, operator_contents::square::Square},
    };

    let tensor = Tensor::new(vec![10.0], vec![])
        .unwrap();
    let mut graph = Graph::new();
    graph.add_new_node(
        0, "in".to_string(), NodeData::Variable(0), vec![], vec![1]
    ).unwrap();
    graph.add_new_node(
        1, "op".to_string(), NodeData::Operator(0), vec![0], vec![2]
    ).unwrap();
    graph.add_new_node(
        2, "out".to_string(), NodeData::Variable(1), vec![1], vec![]
    ).unwrap();
    let mut variables = Variables::new();
    variables.add_new_variable(
        0, Some(0), tensor.into()
    ).unwrap();
    variables.add_new_variable(
        1, Some(2), VariableData::None
    ).unwrap();
    let operator = Operator::new(
        0, Some(1), vec![], Box::new(Square {})
    );
    let result = operator.get_operator().forward(
        operator.get_node().unwrap(),
        &graph, &mut variables
    );
    let result = result.unwrap();
    assert_eq!(result, vec![2]);
    let input_variable = variables.get_variable(0).unwrap();
    let output_variable = variables.get_variable(1).unwrap();
    assert_eq!(output_variable.get_data().to_string(), "F64");
    assert_eq!(output_variable.get_data(), &Tensor::new(vec![100.0], vec![]).unwrap().into());
    println!("input variable: {:?}", input_variable);
    println!("output variable: {:?}", output_variable);
}
