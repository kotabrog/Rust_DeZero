use anyhow::Result;
use std::collections::{HashMap, VecDeque, HashSet};
use super::{Node, NodeData};
use crate::error::KdezeroError;

pub struct Graph {
    nodes: HashMap<usize, Node>,
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

    pub fn get_node(&self, id: usize) -> Result<&Node> {
        match self.nodes.get(&id) {
            Some(node) => Ok(node),
            None => Err(
                KdezeroError::NotFoundError(
                    id.to_string(),
                    "Graph".to_string()
                ).into()
            ),
        }
    }

    pub fn add_new_node(
        &mut self, id: usize, name: String,
        data: NodeData, inputs: Vec<usize>, outputs: Vec<usize>
    ) -> Result<()> {
        if self.nodes.contains_key(&id) {
            return Err(
                KdezeroError::ExistError(
                    id.to_string(),
                    "Graph".to_string()
                ).into()
            );
        }
        let node = Node::new(id, name, data, inputs, outputs);
        self.nodes.insert(id, node);
        Ok(())
    }

    pub fn topological_sort(&self) -> Result<Vec<usize>> {
        fn get_empty_inputs_node(graph: &Graph) -> VecDeque<usize> {
            let mut empty_inputs_node = VecDeque::new();
            for (id, node) in graph.nodes.iter() {
                if node.get_inputs().is_empty() {
                    empty_inputs_node.push_back(*id);
                }
            }
            empty_inputs_node
        }

        fn check_all_inputs_visited(
            graph: &Graph, node_id: usize, visited: &HashSet<usize>
        ) -> bool {
            let node = graph.get_node(node_id).unwrap();
            for input in node.get_inputs() {
                if !visited.contains(input) {
                    return false;
                }
            }
            true
        }

        let mut sorted_list = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = get_empty_inputs_node(self);
        while !queue.is_empty() {
            let id = queue.pop_front().unwrap();
            if visited.contains(&id) {
                continue;
            }
            if check_all_inputs_visited(self, id, &visited) {
                sorted_list.push(id);
                visited.insert(id);
                queue.extend(self.nodes[&id].get_outputs())
            } else {
                queue.push_back(id);
            }
        }
        if sorted_list.len() != self.nodes.len() {
            return Err(
                KdezeroError::NotCollectGraphError(
                    "the graph must be a directed acyclic graph".to_string()
                ).into()
            );
        }
        Ok(sorted_list)
    }
}
