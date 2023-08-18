use crate::operator::OperatorContents;
use crate::variable::VariableData;

pub struct ModelOperator {
    pub name: String,
    pub data: Box<dyn OperatorContents>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub params: Vec<VariableData>,
}

impl ModelOperator {
    pub fn new(
        name: &str, data: Box<dyn OperatorContents>,
        inputs: Vec<&str>, outputs: Vec<&str>, params: Vec<VariableData>
    ) -> Self {
        Self {
            name: name.to_string(),
            data,
            inputs: inputs.into_iter().map(|s| s.to_string()).collect(),
            outputs: outputs.into_iter().map(|s| s.to_string()).collect(),
            params
        }
    }
}
