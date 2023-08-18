use super::Model;

use anyhow::Result;
use crate::node::NodeData;

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

    pub fn backward(&mut self) -> Result<()> {
        if self.sorted_backward_nodes.is_empty() {
            self.sorted_backward_nodes = self.graph.topological_sort(true)?;
        }
        self.set_ones_grad(&self.outputs.clone())?;
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
}
