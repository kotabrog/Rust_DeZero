use crate::variable::VariableData;

pub struct ModelVariable {
    pub name: String,
    pub data: VariableData,
}

impl ModelVariable {
    pub fn new(name: String, data: VariableData) -> Self {
        Self { name, data }
    }
}
