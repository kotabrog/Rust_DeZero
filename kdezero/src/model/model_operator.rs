use crate::operator::OperatorContents;

pub struct ModelOperator {
    pub name: String,
    pub data: Box<dyn OperatorContents>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl ModelOperator {
    pub fn new(
        name: &str, data: Box<dyn OperatorContents>,
        inputs: Vec<&str>, outputs: Vec<&str>
    ) -> Self {
        Self {
            name: name.to_string(),
            data,
            inputs: inputs.into_iter().map(|s| s.to_string()).collect(),
            outputs: outputs.into_iter().map(|s| s.to_string()).collect(),
        }
    }
}
