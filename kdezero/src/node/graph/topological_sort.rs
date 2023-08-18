use anyhow::Result;
use std::collections::{VecDeque, HashSet};
use super::Graph;
use crate::error::KdezeroError;

impl Graph {
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
        let max_count = self.nodes.len().pow(3) + 100;
        for _ in 0..max_count {
            if queue.is_empty() {
                break;
            }
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
