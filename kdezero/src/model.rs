use anyhow::Result;
use crate::node::{NodeData, Graph};
use crate::variable::{Variables, VariableData};
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
