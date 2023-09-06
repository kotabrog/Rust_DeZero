use anyhow::Result;
use super::super::Model;
use crate::node::NodeData;
use crate::error::KdezeroError;

impl Model {
    pub(crate) fn change_node_id(&mut self, node_id: usize, new_node_id: usize) -> Result<()> {
        let input_pos = self.inputs.iter()
            .position(|&x| x == node_id);
        if let Some(pos) = input_pos {
            self.inputs[pos] = new_node_id;
        }
        let output_pos = self.outputs.iter()
            .position(|&x| x == node_id);
        if let Some(pos) = output_pos {
            self.outputs[pos] = new_node_id;
        }
        self.graph.change_node_id(node_id, new_node_id)?;
        match self.get_node_data_from_node_id(new_node_id)? {
            NodeData::Variable(variable_id) =>
                self.variables.set_node_id(*variable_id, Some(new_node_id))?,
            NodeData::Operator(operator_id) =>
                self.operators.set_node(*operator_id, Some(new_node_id))?,
            _ => {}
        }
        Ok(())
    }

    pub(crate) fn change_variable_id(&mut self, variable_id: usize, new_variable_id: usize) -> Result<()> {
        self.variables.change_variable_id(variable_id, new_variable_id)?;
        let node_id_option = self.variables.get_node_id(new_variable_id)?;
        if let Some(node_id) = node_id_option {
            self.graph.set_node_data(node_id, NodeData::Variable(new_variable_id))?;
        }
        self.operators.change_variable_id(variable_id, new_variable_id)?;
        Ok(())
    }

    pub(crate) fn change_operator_id(&mut self, operator_id: usize, new_operator_id: usize) -> Result<()> {
        self.operators.change_operator_id(operator_id, new_operator_id)?;
        let node_id = self.operators.get_node_id(new_operator_id)?;
        self.graph.set_node_data(node_id, NodeData::Operator(new_operator_id))?;
        Ok(())
    }

    pub(crate) fn insert_structure_model(&mut self, mut model: Model, inputs: &[usize])
        -> Result<Vec<usize>>{
        if inputs.len() != model.inputs.len() {
            return Err(KdezeroError::SizeError(
                "inputs size".to_string(),
                model.inputs.len(),
                inputs.len(),
            ).into());
        }
        self.sorted_forward_nodes = vec![];
        self.sorted_backward_nodes = vec![];

        let model_node_ids = model.graph
            .get_nodes().keys().map(|&x| x).collect::<Vec<usize>>();
        let mut new_node_id = self.graph.get_next_id()
            .max(model.graph.get_next_id());
        for node_id in model_node_ids {
            model.change_node_id(node_id, new_node_id)?;
            new_node_id += 1;
        }
        self.graph.set_next_id(new_node_id);

        let variable_node_ids = model.variables
            .get_variables().keys().map(|&x| x).collect::<Vec<usize>>();
        let mut new_variable_id = self.variables.get_next_id()
            .max(model.variables.get_next_id());
        for variable_id in variable_node_ids {
            model.change_variable_id(variable_id, new_variable_id)?;
            new_variable_id += 1;
        }
        self.variables.set_next_id(new_variable_id);

        let operator_node_ids = model.operators
            .get_operators().keys().map(|&x| x).collect::<Vec<usize>>();
        let mut new_operator_id = self.operators.get_next_id()
            .max(model.operators.get_next_id());
        for operator_id in operator_node_ids {
            model.change_operator_id(operator_id, new_operator_id)?;
            new_operator_id += 1;
        }
        self.operators.set_next_id(new_operator_id);

        let model_inputs = model.inputs.clone();
        for (node_id, new_node_id) in model_inputs.iter().zip(inputs) {
            let variable_id = model.get_variable_id_from_node_id(*node_id)?;
            model.change_node_id(*node_id, *new_node_id)?;
            model.variables.delete_variable(variable_id)?;
        }

        let outputs = model.outputs.clone();

        for (node_id, node) in model.graph.move_all_node() {
            if model.inputs.contains(&node_id) {
                let pos = model.inputs.iter()
                    .position(|&x| x == node_id).unwrap();
                let output_ids = node.move_output();
                for output_id in output_ids {
                    self.add_node_output(inputs[pos], output_id)?;
                }
            } else {
                self.graph.get_nodes_mut().insert(node_id, node);
            }
        }
        for (variable_id, variable) in model.variables.move_all_variable() {
            self.variables.get_variables_mut().insert(variable_id, variable);
        }
        for (operator_id, operator) in model.operators.move_all_operator() {
            self.operators.get_operators_mut().insert(operator_id, operator);
        }

        Ok(outputs)
    }
}
