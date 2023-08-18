use anyhow::Result;
use super::{Node, NodeData, Graph};

impl Graph {
    pub(crate) fn add_node(&mut self, node: Node) -> Result<()> {
        let id = node.get_id();
        self.check_id_in_nodes(id)?;
        self.nodes.insert(id, node);
        self.update_next_id(id);
        Ok(())
    }

    pub(crate) fn add_new_node(
        &mut self, id: usize, name: String,
        data: NodeData, inputs: Vec<usize>, outputs: Vec<usize>
    ) -> Result<()> {
        let node = Node::new(id, name, data, inputs, outputs);
        self.add_node(node)
    }

    pub(crate) fn add_node_input(&mut self, node_id: usize, input: usize) -> Result<()> {
        let node = self.get_node_mut(node_id)?;
        node.add_input(input);
        Ok(())
    }

    pub(crate) fn add_node_output(&mut self, node_id: usize, output: usize) -> Result<()> {
        let node = self.get_node_mut(node_id)?;
        node.add_output(output);
        Ok(())
    }

    pub(crate) fn set_node_inputs(&mut self, node_id: usize, inputs: Vec<usize>) -> Result<()> {
        let node = self.get_node_mut(node_id)?;
        node.set_inputs(inputs);
        Ok(())
    }

    pub(crate) fn set_node_data(&mut self, node_id: usize, data: NodeData) -> Result<()> {
        let node = self.get_node_mut(node_id)?;
        node.set_data(data);
        Ok(())
    }

    pub(crate) fn change_node_id(&mut self, old_id: usize, new_id: usize) -> Result<()> {
        self.check_id_not_in_nodes(old_id)?;
        self.check_id_in_nodes(new_id)?;
        let mut node = self.nodes.remove(&old_id).unwrap();
        node.set_id(new_id);
        self.nodes.insert(new_id, node);
        for node in self.nodes.values_mut() {
            node.change_input_and_output_node_id(old_id, new_id);
        }
        Ok(())
    }
}
