mod getter;
mod setter;
mod check_state;
mod topological_sort;
mod utility;

use std::collections::HashMap;
use super::{Node, NodeData};

#[derive(Debug, Clone)]
pub struct Graph {
    nodes: HashMap<usize, Node>,
    next_id: usize,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
        }
    }
}
