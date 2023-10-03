use anyhow::Result;
use crate::error::KdezeroError;

#[derive(Debug, Clone)]
pub enum NodeData {
    None,
    Variable(usize),
    Operator(usize),
    Layer(usize),
}

impl NodeData {
    pub fn to_string(&self) -> String {
        match self {
            NodeData::None => "None".to_string(),
            NodeData::Variable(_) => "Variable".to_string(),
            NodeData::Operator(_) => "Operator".to_string(),
            NodeData::Layer(_) => "Layer".to_string(),
        }
    }

    pub fn get_variable_id(&self) -> Result<usize> {
        match self {
            NodeData::Variable(id) => Ok(*id),
            _ => Err(
                KdezeroError::NotCollectTypeError(
                    self.to_string(),
                    "Variable".to_string()
                ).into()
            ),
        }
    }

    pub fn get_operator_id(&self) -> Result<usize> {
        match self {
            NodeData::Operator(id) => Ok(*id),
            _ => Err(
                KdezeroError::NotCollectTypeError(
                    self.to_string(),
                    "Operator".to_string()
                ).into()
            ),
        }
    }
}
