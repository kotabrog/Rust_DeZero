use std::collections::HashMap;
use super::Node;

pub struct Graph {
    pub nodes: HashMap<usize, Node>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    pub fn get_nodes(&self) -> &HashMap<usize, Node> {
        &self.nodes
    }

    pub fn add_node(&mut self, node: Node) -> Option<Node>{
        self.nodes.insert(node.get_id(), node)
    }

    pub fn get_node(&self, id: usize) -> Option<&Node> {
        self.nodes.get(&id)
    }
}
