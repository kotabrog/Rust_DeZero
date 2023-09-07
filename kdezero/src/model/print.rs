use std::fs::File;
use std::io::prelude::*;
use std::process::Command;
use anyhow::Result;
use super::Model;
use crate::node::NodeData;

impl Model {
    pub fn print_model(&self) {
        self.graph.print_graph();
        self.variables.print_variables();
        // println!("operators: {:?}", self.operators);
        println!("inputs: {:?}", self.inputs);
        println!("outputs: {:?}", self.outputs);
        println!("sorted_forward_nodes: {:?}", self.sorted_forward_nodes);
        println!("sorted_backward_nodes: {:?}", self.sorted_backward_nodes);
        if let Some(grad_model) = &self.grad_model {
            println!("grad_model: ");
            grad_model.print_model();
        }
    }

    fn variable_to_dot_string(&self, node_id: usize, name: &str) -> String {
        format!("var_{} [label=\"{}\", color=orange, style=filled]\n", node_id, name)
    }

    fn function_to_dot_string(&self, node_id: usize, name: &str) -> String {
        format!("func_{} [label=\"{}\", color=lightblue, style=filled, shape=box]\n", node_id, name)
    }

    fn connect_to_dot_string(&self, from_id: usize, to_id: usize, from_is_func: bool) -> String {
        if from_is_func {
            format!("func_{} -> var_{};\n", from_id, to_id)
        } else {
            format!("var_{} -> func_{};\n", from_id, to_id)
        }
    }

    fn node_to_dot_string(&self, node_id: usize) -> Result<String> {
        let node = self.graph.get_node(node_id)?;
        let mut dot_string = String::new();
        match node.get_data() {
            NodeData::Variable(_) => {
                let name = node.get_name();
                dot_string.push_str(&self.variable_to_dot_string(node_id, name));
                for output in node.get_outputs().iter() {
                    dot_string.push_str(&self.connect_to_dot_string(node_id, *output, false));
                }
            },
            NodeData::Operator(_) => {
                let name = node.get_name();
                dot_string.push_str(&self.function_to_dot_string(node_id, name));
                for output in node.get_outputs().iter() {
                    dot_string.push_str(&self.connect_to_dot_string(node_id, *output, true));
                }
            },
            _ => {},
        }
        Ok(dot_string)
    }

    pub fn get_dot_graph(&self) -> Result<String> {
        let mut text = String::new();
        for node_id in self.graph.get_nodes().keys() {
            text.push_str(&self.node_to_dot_string(*node_id)?);
        }
        Ok(format!("digraph g {{\n{}}}", text))
    }

    fn write_dot_graph_to_file(&self, out_path: &str) -> Result<()> {
        let text = self.get_dot_graph()?;
        let mut file = File::create(out_path)?;
        file.write_all(text.as_bytes())?;
        Ok(())
    }

    pub fn plot_dot_graph(&self, out_path_without_extension: &str, to_png: bool) -> Result<()> {
        let dot_path = format!("{}.dot", out_path_without_extension);
        self.write_dot_graph_to_file(&dot_path)?;
        if to_png {
            let png_path = format!("{}.png", out_path_without_extension);
            Command::new("dot")
                .args([dot_path.as_str(), "-T", "png", "-o", png_path.as_str()])
                .status()?;
        }
        Ok(())
    }
}
