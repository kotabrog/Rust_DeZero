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
        name: String, data: Box<dyn OperatorContents>,
        inputs: Vec<String>, outputs: Vec<String>, params: Vec<VariableData>
    ) -> Self {
        Self { name, data, inputs, outputs, params }
    }
}
