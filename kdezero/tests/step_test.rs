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
        node::{Node, NodeData},
        variable::{Variable, VariableData},
        operator::{Operator, operators::square::Square},
    };
    use std::collections::HashMap;

    let tensor = Tensor::new(vec![10.0], vec![])
        .unwrap();
    let mut nodes = HashMap::new();
    let in_node = Node::new(
        0, "in".to_string(), NodeData::Variable(0), vec![], vec![1]
    );
    let op_node = Node::new(
        1, "op".to_string(), NodeData::Operator(0), vec![0], vec![2],
    );

    let out_node = Node::new(
        2, "out".to_string(), NodeData::Variable(1), vec![1], vec![]
    );
    nodes.insert(0, in_node);
    nodes.insert(1, op_node);
    nodes.insert(2, out_node);
    let mut variables = HashMap::new();
    let in_variable = Variable::new(
        0, Some(0), tensor.into()
    );
    let out_variable = Variable::new(
        1, Some(2), VariableData::None
    );
    variables.insert(0, in_variable);
    variables.insert(1, out_variable);
    let operator = Operator::new(
        0, Some(1), vec![], Box::new(Square {})
    );
    let result = operator.get_operator().forward(
        operator.get_node().unwrap(),
        &nodes, &mut variables
    );
    let result = result.unwrap();
    assert_eq!(result, vec![2]);
    assert_eq!(variables[&1].get_data().to_string(), "F64");
    assert_eq!(variables[&1].get_data(), &Tensor::new(vec![100.0], vec![]).unwrap().into());
    println!("input variable: {:?}", variables[&0]);
    println!("output variable: {:?}", variables[&1]);
}
