pub mod model_variable;
pub mod model_operator;
mod make;
mod run;
mod getter;
mod check_state;

pub use model_variable::ModelVariable;
pub use model_operator::ModelOperator;

use anyhow::Result;
use crate::node::{NodeData, Graph};
use crate::variable::{Variables, VariableData};
use crate::operator::{Operators, OperatorContents};

pub struct Model {
    graph: Graph,
    variables: Variables,
    operators: Operators,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    sorted_forward_nodes: Vec<usize>,
    sorted_backward_nodes: Vec<usize>,
    grad_model: Option<Box<Model>>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            variables: Variables::new(),
            operators: Operators::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            sorted_forward_nodes: Vec::new(),
            sorted_backward_nodes: Vec::new(),
            grad_model: None,
        }
    }

    pub fn add_new_node(
        &mut self, id: usize, name: String,
        data: NodeData, inputs: Vec<usize>, outputs: Vec<usize>)
    -> Result<()> {
        self.graph.add_new_node(id, name, data, inputs, outputs)?;
        if !self.sorted_forward_nodes.is_empty() {
            self.sorted_forward_nodes = Vec::new();
        }
        Ok(())
    }

    pub fn add_new_variable(
        &mut self, id: usize, node: Option<usize>, data: VariableData)
    -> Result<()> {
        self.variables.add_new_variable(id, node, data)
    }

    pub fn add_new_operator(
        &mut self, id: usize, node: Option<usize>, params: Vec<usize>, operator: Box<dyn OperatorContents>
    ) -> Result<()> {
        self.operators.add_new_operator(id, node, params, operator)
    }

    fn set_node_inputs(&mut self, node_id: usize, inputs: Vec<usize>) -> Result<()> {
        self.graph.set_node_inputs(node_id, inputs)
    }

    fn add_node_input(&mut self, node_id: usize, input: usize) -> Result<()> {
        self.graph.add_node_input(node_id, input)
    }

    fn add_node_output(&mut self, node_id: usize, output: usize) -> Result<()> {
        self.graph.add_node_output(node_id, output)
    }

    pub fn set_grad(&mut self, id: usize, grad: Option<usize>) -> Result<()> {
        self.variables.set_grad(id, grad)
    }

    pub fn set_inputs(&mut self, inputs: Vec<usize>) {
        self.inputs = inputs;
    }

    pub fn set_outputs(&mut self, outputs: Vec<usize>) {
        self.outputs = outputs;
    }

    pub(crate) fn set_ones_grad(&mut self, nodes: &Vec<usize>) -> Result<()> {
        for &node_id in nodes {
            let node = self.graph.get_node(node_id)?;
            let variable_id = node.get_data().get_variable_id()?;
            let variable_data =
                self.variables.get_variable(variable_id)?.get_data();
            let grad_data = VariableData::ones_like(variable_data)?;
            let grad = self.variables.get_grad(variable_id)?;
            match grad {
                Some(_) => (),
                None => {
                    self.init_grad_model();
                    let grad_model = self.grad_model.as_mut().unwrap();
                    let grad_node_id = grad_model.graph.get_next_id();
                    let grad_variable_id = grad_model.variables.get_next_id();
                    grad_model.add_new_node(
                        grad_node_id, "".to_string(),
                        NodeData::Variable(grad_variable_id),
                        Vec::new(), Vec::new()
                    )?;
                    grad_model.add_new_variable(
                        grad_variable_id, Some(grad_node_id), grad_data
                    )?;
                    self.set_grad(variable_id, Some(grad_node_id))?;
                },
            }
        }
        Ok(())
    }

    pub(crate) fn init_grad_model(&mut self) {
        match self.grad_model {
            Some(_) => (),
            None => {
                self.grad_model = Some(Box::new(Model::new()));
            }
        }
    }
}
