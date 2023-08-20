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
            match node.get_data() {
                NodeData::Operator(operator_id) => {
                    let operator = self.operators.get_operator(*operator_id)?;
                    let (node_id, operator) = operator.get_backward_set()?;
                    operator.forward(node_id, self)?;
                },
                _ => (),
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
        self.set_ones_grad(&vec![node_id])?;
        self.set_zeros_grad(&remain_outputs)?;
        for id in self.sorted_backward_nodes.clone().iter() {
            let node = self.graph.get_node(*id)?;
            match node.get_data() {
                NodeData::Operator(operator_id) => {
                    let operator = self.operators.get_operator(*operator_id)?;
                    let (node_id, operator) = operator.get_backward_set()?;
                    operator.backward(node_id, self)?;
                },
                _ => (),
            }
        }
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
}
