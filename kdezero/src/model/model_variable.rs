use crate::variable::VariableData;

pub struct ModelVariable {
    pub name: String,
    pub data: VariableData,
}

impl ModelVariable {
    pub fn new(name: &str, data: VariableData) -> Self {
        Self { name: name.to_string(), data }
    }
}
