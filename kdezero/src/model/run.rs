use super::Model;

use anyhow::Result;
use crate::node::NodeData;

impl Model {
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

    pub fn backward(&mut self) -> Result<()> {
        if self.sorted_backward_nodes.is_empty() {
            self.sorted_backward_nodes = self.graph.topological_sort(true)?;
        }
        self.set_ones_grad(&self.outputs.clone())?;
        self.init_grad_model();
        let grad_model = self.grad_model.as_mut().unwrap();
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
