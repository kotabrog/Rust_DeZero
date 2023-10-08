use anyhow::Result;
use super::Model;

impl Model {
    pub(crate) fn check_inputs_len(&self, node_id: usize, len: usize) -> Result<()> {
        self.graph
            .check_inputs_len(node_id, len)
    }

    pub(crate) fn check_outputs_len(&self, node_id: usize, len: usize) -> Result<()> {
        self.graph
            .check_outputs_len(node_id, len)
    }

    pub(crate) fn check_inputs_len_at_least(&self, node_id: usize, len: usize) -> Result<()> {
        self.graph
            .check_inputs_len_at_least(node_id, len)
    }

    pub(crate) fn check_outputs_len_at_least(&self, node_id: usize, len: usize) -> Result<()> {
        self.graph
            .check_outputs_len_at_least(node_id, len)
    }

    pub(crate) fn check_inputs_outputs_len(
        &self, node_id: usize, inputs_len: usize, outputs_len: usize
    ) -> Result<(&Vec<usize>, &Vec<usize>)> {
        let node = self.graph.get_node(node_id)?;
        node.check_inputs_len(inputs_len)?;
        node.check_outputs_len(outputs_len)?;
        Ok((node.get_inputs(), node.get_outputs()))
    }

    // pub(crate) fn check_params_len(
    //     &self, node_id: usize, params_len: usize
    // ) -> Result<&Vec<usize>> {
    //     let operator_id = self.get_operator_id_from_node_id(node_id)?;
    //     self.operators.check_params_len(operator_id, params_len)
    // }

    pub(crate) fn is_in_node_id(&self, node_id: usize) -> bool {
        self.graph.get_nodes().contains_key(&node_id)
    }

    // pub(crate) fn is_all_output_grad_present(&self, node_id: usize) -> Result<bool> {
    //     let output = self.get_node_outputs_from_node_id(node_id)?;
    //     for &output_id in output.iter() {
    //         let output_grad_id = self.get_grad_data_from_node_id(output_id)?;
    //         if output_grad_id.is_none() {
    //             return Ok(false);
    //         }
    //     }
    //     Ok(true)
    // }
}
