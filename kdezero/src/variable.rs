pub mod variables;
pub mod variable_data;

pub use variables::Variables;
pub use variable_data::VariableData;

#[derive(Debug, Clone, PartialEq)]
pub struct Variable {
    id: usize,
    node: Option<usize>,
    data: VariableData,
}

impl Variable {
    pub fn new(id: usize, node: Option<usize>, data: VariableData) -> Self {
        Self {
            id,
            node,
            data,
        }
    }

    pub fn get_id(&self) -> usize {
        self.id
    }

    pub fn get_node(&self) -> Option<usize> {
        self.node
    }

    pub fn get_data(&self) -> &VariableData {
        &self.data
    }

    pub fn set_data(&mut self, data: VariableData) {
        self.data = data;
    }
}
