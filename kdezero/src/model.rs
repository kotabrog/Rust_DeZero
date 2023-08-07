use std::collections::HashMap;
use crate::node::graph::Graph;
use crate::variable::variables::Variables;
use crate::operator::Operator;

pub struct Model {
    nodes: Graph,
    variables: Variables,
    operators: HashMap<usize, Operator>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            nodes: Graph::new(),
            variables: Variables::new(),
            operators: HashMap::new(),
        }
    }

    pub fn get_nodes(&self) -> &Graph {
        &self.nodes
    }

    pub fn get_variables(&self) -> &Variables {
        &self.variables
    }

    pub fn get_operators(&self) -> &HashMap<usize, Operator> {
        &self.operators
    }

    // fn add_node(&mut self, node: Node) {
    //     self.nodes.insert(node.get_id(), node);
    // }

    // fn get_node(&self, id: usize) -> Option<&Node> {
    //     self.nodes.get(&id)
    // }
}
