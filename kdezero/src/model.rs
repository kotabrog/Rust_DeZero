use std::collections::HashMap;
use crate::node::Node;
use crate::variable::Variable;
use crate::operator::Operator;

pub struct Model {
    nodes: HashMap<usize, Node>,
    variables: HashMap<usize, Variable>,
    operators: HashMap<usize, Operator>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            variables: HashMap::new(),
            operators: HashMap::new(),
        }
    }

    pub fn get_nodes(&self) -> &HashMap<usize, Node> {
        &self.nodes
    }

    pub fn get_variables(&self) -> &HashMap<usize, Variable> {
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
