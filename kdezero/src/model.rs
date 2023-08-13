pub mod model_variable;
pub mod model_operator;
mod make;

pub use model_variable::ModelVariable;
pub use model_operator::ModelOperator;

use anyhow::Result;
use ktensor::Tensor;
use crate::node::{NodeData, Graph};
use crate::variable::{Variable, Variables, VariableData};
use crate::operator::{Operators, OperatorContents};
use crate::error::KdezeroError;

pub struct Model {
    graph: Graph,
    variables: Variables,
    operators: Operators,
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
            sorted_forward_nodes: Vec::new(),
            sorted_backward_nodes: Vec::new(),
            grad_model: None,
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

    pub fn get_grad_model(&self) -> &Option<Box<Model>> {
        &self.grad_model
    }

    pub fn get_grad_model_mut(&mut self) -> Option<&mut Model> {
        self.grad_model.as_mut().map(|model| &mut **model)
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

    pub fn get_variable_from_name(&self, name: &str) -> Result<&Variable> {
        let node = self.graph.get_node_from_name(name)?;
        let variable_id = node.get_data().get_variable_id()?;
        self.variables.get_variable(variable_id)
    }

    pub fn get_grad_from_variable_name(&self, name: &str) -> Result<&Variable> {
        let node = self.graph.get_node_from_name(name)?;
        let variable_id = node.get_data().get_variable_id()?;
        let grad_id = self.variables.get_grad(variable_id)?
            .ok_or_else(|| KdezeroError::NotFoundError(
                "Variable.grad".to_string(),
                "Variable".to_string()
            ))?;
        self.grad_model.as_ref()
            .ok_or_else(
                || KdezeroError::NotFoundError(
                    "grad_model".to_string(),
                    "Model".to_string()
                )
            )?.variables.get_variable(grad_id)
    }

    pub fn forward(&mut self) -> Result<()> {
        if self.sorted_forward_nodes.is_empty() {
            self.sorted_forward_nodes = self.graph.topological_sort(false)?;
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

    pub fn backward(&mut self, name: &str) -> Result<()> {
        if self.sorted_backward_nodes.is_empty() {
            self.sorted_backward_nodes = self.graph.topological_sort(true)?;
        }
        let grad_model = match self.grad_model {
            Some(ref mut grad_model) => grad_model,
            None => {
                self.grad_model = Some(Box::new(Model::new()));
                self.grad_model.as_mut().unwrap()
            }
        };
        let node = self.graph.get_node_from_name(name)?;
        let variable_id = node.get_data().get_variable_id()?;
        let variable_data =
            self.variables.get_variable(variable_id)?.get_data();
        let grad_data = match variable_data {
            VariableData::F32(tensor) =>
                VariableData::F32(Box::new(Tensor::ones_like(tensor))),
            VariableData::F64(tensor) =>
                VariableData::F64(Box::new(Tensor::ones_like(tensor))),
            VariableData::USIZE(tensor) =>
                VariableData::USIZE(Box::new(Tensor::ones_like(tensor))),
            VariableData::I32(tensor) =>
                VariableData::I32(Box::new(Tensor::ones_like(tensor))),
            VariableData::I64(tensor) =>
                VariableData::I64(Box::new(Tensor::ones_like(tensor))),
            _ => return Err(KdezeroError::NotImplementedTypeError(
                variable_data.to_string(),
                "Model".to_string()
            ).into()),
        };
        let grad = self.variables.get_grad(variable_id)?;
        match grad {
            Some(_) => (),
            None => {
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
        let grad_model = match self.grad_model {
            Some(ref mut grad_model) => grad_model,
            None => {
                self.grad_model = Some(Box::new(Model::new()));
                self.grad_model.as_mut().unwrap()
            }
        };
        for id in self.sorted_backward_nodes.iter() {
            let node = self.graph.get_node(*id)?;
            match node.get_data() {
                NodeData::Operator(operator_id) => {
                    let operator = self.operators.get_operator(*operator_id)?;
                    operator.backward(
                        &self.graph, &mut self.variables, grad_model
                    )?;
                },
                _ => (),
            }
        }
        Ok(())
    }
}