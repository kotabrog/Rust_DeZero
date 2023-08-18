use anyhow::Result;
use super::Model;

impl Model {
    pub(crate) fn check_inputs_len(&self, node_id: usize, len: usize) -> Result<()> {
        self.graph
            .get_node(node_id)?
            .check_inputs_len(len)
    }

    pub(crate) fn check_outputs_len(&self, node_id: usize, len: usize) -> Result<()> {
        self.graph
            .get_node(node_id)?
            .check_outputs_len(len)
    }

    pub(crate) fn check_inputs_len_at_least(&self, node_id: usize, len: usize) -> Result<()> {
        self.graph
            .get_node(node_id)?
            .check_inputs_len_at_least(len)
    }

    pub(crate) fn check_outputs_len_at_least(&self, node_id: usize, len: usize) -> Result<()> {
        self.graph
            .get_node(node_id)?
            .check_outputs_len_at_least(len)
    }

    pub(crate) fn check_inputs_outputs_len(
        &self, node_id: usize, inputs_len: usize, outputs_len: usize
    ) -> Result<(&Vec<usize>, &Vec<usize>)> {
        let node = self.graph.get_node(node_id)?;
        node.check_inputs_len(inputs_len)?;
        node.check_outputs_len(outputs_len)?;
        Ok((node.get_inputs(), node.get_outputs()))
    }
}
