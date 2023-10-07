use super::Model;

use anyhow::Result;
use crate::node::NodeData;
use crate::variable::VariableData;
use crate::operator::operator_contents::Add;

impl Model {
    pub fn forward(&mut self) -> Result<()> {
        if self.sorted_forward_nodes.is_empty() {
            self.sorted_forward_nodes = self.graph.topological_sort(false)?;
        }
        for id in self.sorted_forward_nodes.clone().iter() {
            let node = self.graph.get_node(*id)?;
            let operator_contents = match node.get_data() {
                NodeData::Operator(operator_id) => {
                    let operator = self.operators.get_operator_mut(*operator_id)?;
                    Some((*operator_id, operator.take_operator_result()?))
                },
                _ => None,
            };
            if let Some(temp) = operator_contents {
                let (operator_id, mut operator_contents) = temp;
                operator_contents.forward(*id, self)?;
                let operator = self.operators.get_operator_mut(operator_id)?;
                operator.set_operator(operator_contents);
            }
        }
        Ok(())
    }

    pub fn backward(&mut self, node_id: usize) -> Result<()> {
        if self.sorted_backward_nodes.is_empty() {
            self.sorted_backward_nodes = self.graph.topological_sort(true)?;
        }
        let remain_outputs: Vec<usize> = self.get_outputs().clone()
            .into_iter().filter(|&output_id| output_id != node_id)
            .collect();
        self.init_grad_model();
        self.set_ones_grad(&vec![node_id])?;
        self.set_zeros_grad(&remain_outputs)?;
        for id in self.sorted_backward_nodes.clone().iter() {
            let node = self.graph.get_node(*id)?;
            let operator_contents = match node.get_data() {
                NodeData::Operator(operator_id) => {
                    let operator = self.operators.get_operator_mut(*operator_id)?;
                    Some((*operator_id, operator.take_operator_result()?))
                },
                _ => None,
            };
            if let Some(temp) = operator_contents {
                let (operator_id, mut operator_contents) = temp;
                operator_contents.backward(*id, self)?;
                let operator = self.operators.get_operator_mut(operator_id)?;
                operator.set_operator(operator_contents);
            }
        }

        let mut grad_output = vec![];
        for output in self.inputs.clone() {
            grad_output.push(self.get_grad_id_from_node_id(output)?);
        }
        self.get_grad_model_mut().set_outputs(grad_output);

        let mut grad_input = vec![self.get_grad_id_from_node_id(node_id)?];
        let grad_model = self.get_grad_model_result()?;
        let self_id = self.graph.get_next_id();
        for id in grad_model.graph.get_nodes().keys() {
            if *id < self_id {
                if grad_model.get_node_inputs_from_node_id(*id)?.is_empty() {
                    grad_input.push(*id);
                }
            }
        }
        self.get_grad_model_mut().set_inputs(grad_input);
        self.get_grad_model_mut().forward()?;
        Ok(())
    }

    pub(crate) fn set_or_add_grad(&mut self, id: usize, grad: usize) -> Result<()> {
        let target_grad = self.get_grad_from_node_id(id)?;
        let grad_id =
            if let Some(target_grad_id) = target_grad {
                let grad_model = self.get_grad_model_mut();
                let add_output_node_id = grad_model.graph.get_next_id();
                let add_node_id = grad_model.graph.get_next_id() + 1;
                let add_output_variable_id = grad_model.variables.get_next_id();
                let add_operator_id = grad_model.operators.get_next_id();
                grad_model.add_new_node(
                    add_output_node_id, "".to_string(),
                    NodeData::Variable(add_output_variable_id),
                    vec![add_node_id], vec![]
                )?;
                grad_model.add_new_node(
                    add_node_id, "".to_string(),
                    NodeData::Operator(add_operator_id),
                    vec![target_grad_id, grad], vec![add_output_node_id]
                )?;
                grad_model.add_new_variable(
                    add_output_variable_id, Some(add_output_node_id),
                    VariableData::None
                )?;
                grad_model.add_new_operator(
                    add_operator_id, Some(add_node_id),
                    vec![], Box::new(Add {})
                )?;
                grad_model.add_node_output(target_grad_id, add_node_id)?;
                grad_model.add_node_output(grad, add_node_id)?;
                add_output_node_id
            } else {
                grad
        };
        self.set_grad_from_node_id(id, Some(grad_id))?;
        Ok(())
    }

    pub(crate) fn clone_node_to_grad_model(&mut self, node_id: usize) -> Result<()> {
        let node = self.graph.get_node(node_id)?;
        let node_data = node.get_data().clone();
        match node_data {
            NodeData::Variable(variable_id) => {
                let variable = self.variables.get_variable(variable_id)?.clone();
                self.grad_model.as_mut().unwrap().variables
                    .add_variable_no_check(variable);
            },
            NodeData::Operator(operator_id) => {
                let operator = self.operators.get_operator(operator_id)?.clone();
                self.grad_model.as_mut().unwrap().operators
                    .add_operator_no_check(operator);
            },
            _ => (),
        }
        let mut node = node.clone();
        node.set_inputs(vec![]);
        node.set_outputs(vec![]);
        self.get_grad_model_mut().graph
            .add_node_no_check(node);
        Ok(())
    }

    pub(crate) fn clone_node_to_grad_model_if_needed(&mut self, node_id: usize) -> Result<()> {
        if !self.get_grad_model_result()?.is_in_node_id(node_id) {
            self.clone_node_to_grad_model(node_id)?;
        }
        Ok(())
    }
}
