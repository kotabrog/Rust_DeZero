use anyhow::Result;
use std::collections::{HashMap, VecDeque, HashSet};
use super::{Node, NodeData};
use crate::error::KdezeroError;

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

    pub fn get_nodes(&self) -> &HashMap<usize, Node> {
        &self.nodes
    }

    pub fn get_next_id(&self) -> usize {
        self.next_id
    }

    pub fn add_node(&mut self, node: Node) -> Option<Node>{
        self.nodes.insert(node.get_id(), node)
    }

    fn check_id_in_nodes(&self, id: usize) -> Result<()> {
        if self.nodes.contains_key(&id) {
            return Err(
                KdezeroError::ExistError(
                    id.to_string(),
                    "Graph".to_string()
                ).into()
            );
        }
        Ok(())
    }

    fn check_id_not_in_nodes(&self, id: usize) -> Result<()> {
        if !self.nodes.contains_key(&id) {
            return Err(
                KdezeroError::NotFoundError(
                    id.to_string(),
                    "Graph".to_string()
                ).into()
            );
        }
        Ok(())
    }

    pub fn get_node(&self, id: usize) -> Result<&Node> {
        self.check_id_not_in_nodes(id)?;
        Ok(self.nodes.get(&id).unwrap())
    }

    pub fn get_node_from_name(&self, name: &str) -> Result<&Node> {
        for node in self.nodes.values() {
            if node.get_name() == name {
                return Ok(node);
            }
        }
        Err(
            KdezeroError::NotFoundError(
                name.to_string(),
                "Graph".to_string()
            ).into()
        )
    }

    pub fn add_new_node(
        &mut self, id: usize, name: String,
        data: NodeData, inputs: Vec<usize>, outputs: Vec<usize>
    ) -> Result<()> {
        self.check_id_in_nodes(id)?;
        let node = Node::new(id, name, data, inputs, outputs);
        self.nodes.insert(id, node);
        self.next_id = self.next_id.max(id) + 1;
        Ok(())
    }

    pub(crate) fn set_node_inputs(&mut self, node_id: usize, inputs: Vec<usize>) -> Result<()> {
        self.check_id_not_in_nodes(node_id)?;
        let node = self.nodes.get_mut(&node_id).unwrap();
        node.set_inputs(inputs);
        Ok(())
    }

    pub(crate) fn add_node_input(&mut self, node_id: usize, input: usize) -> Result<()> {
        self.check_id_not_in_nodes(node_id)?;
        let node = self.nodes.get_mut(&node_id).unwrap();
        node.add_input(input);
        Ok(())
    }

    pub(crate) fn add_node_output(&mut self, node_id: usize, output: usize) -> Result<()> {
        self.check_id_not_in_nodes(node_id)?;
        let node = self.nodes.get_mut(&node_id).unwrap();
        node.add_output(output);
        Ok(())
    }

    pub fn topological_sort(&self, reverse: bool) -> Result<Vec<usize>> {
        fn get_empty_inputs_node(graph: &Graph, reverse: bool) -> VecDeque<usize> {
            let mut empty_inputs_node = VecDeque::new();
            for (id, node) in graph.nodes.iter() {
                let nodes = if reverse {
                    node.get_outputs()
                } else {
                    node.get_inputs()
                };
                if nodes.is_empty() {
                    empty_inputs_node.push_back(*id);
                }
            }
            empty_inputs_node
        }

        fn check_all_inputs_visited(
            graph: &Graph, node_id: usize, visited: &HashSet<usize>, reverse: bool
        ) -> bool {
            let node = graph.get_node(node_id).unwrap();
            let nodes = if reverse {
                node.get_outputs()
            } else {
                node.get_inputs()
            };
            for input in nodes {
                if !visited.contains(input) {
                    return false;
                }
            }
            true
        }

        let mut sorted_list = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = get_empty_inputs_node(self, reverse);
        while !queue.is_empty() {
            let id = queue.pop_front().unwrap();
            if visited.contains(&id) {
                continue;
            }
            if check_all_inputs_visited(self, id, &visited, reverse) {
                sorted_list.push(id);
                visited.insert(id);
                let nodes = if reverse {
                    self.nodes[&id].get_inputs()
                } else {
                    self.nodes[&id].get_outputs()
                };
                queue.extend(nodes)
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
