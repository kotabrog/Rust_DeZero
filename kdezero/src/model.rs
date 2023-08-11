pub mod model_variable;
pub mod model_operator;
mod make;

pub use model_variable::ModelVariable;
pub use model_operator::ModelOperator;

use anyhow::Result;
use crate::node::{NodeData, Graph};
use crate::variable::{Variable, Variables, VariableData};
use crate::operator::{Operators, OperatorContents};

pub struct Model {
    graph: Graph,
    variables: Variables,
    operators: Operators,
    sorted_forward_nodes: Vec<usize>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            variables: Variables::new(),
            operators: Operators::new(),
            sorted_forward_nodes: Vec::new(),
        }
    }

    pub fn get_graph(&self) -> &Graph {
        &self.graph
    }

    pub fn get_variables(&self) -> &Variables {
        &self.variables
    }

    pub fn get_operators(&self) -> &Operators {
        &self.operators
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

    pub fn get_variable_from_name(&self, name: &str) -> Result<&Variable> {
        let node = self.graph.get_node_from_name(name)?;
        let variable_id = node.get_data().get_variable_id()?;
        self.variables.get_variable(variable_id)
    }

    pub fn forward(&mut self) -> Result<()> {
        if self.sorted_forward_nodes.is_empty() {
            self.sorted_forward_nodes = self.graph.topological_sort()?;
        }
        for id in self.sorted_forward_nodes.iter() {
            let node = self.graph.get_node(*id)?;
            match node.get_data() {
                NodeData::Operator(operator_id) => {
                    let operator = self.operators.get_operator(*operator_id)?;
                    operator.forward(&self.graph, &mut self.variables)?;
                },
                _ => (),
            }
        }
        Ok(())
    }
}
