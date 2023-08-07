extern crate ktensor;

use ktensor::Tensor;

#[derive(Debug, Clone, PartialEq)]
pub enum VariableData {
    None,
    F32(Box<Tensor<f32>>),
    F64(Box<Tensor<f64>>),
    USIZE(Box<Tensor<usize>>),
    I32(Box<Tensor<i32>>),
    I64(Box<Tensor<i64>>),
}

impl VariableData {
    pub fn to_string(&self) -> String {
        match self {
            VariableData::None => "None".to_string(),
            VariableData::F32(_) => "F32".to_string(),
            VariableData::F64(_) => "F64".to_string(),
            VariableData::USIZE(_) => "USIZE".to_string(),
            VariableData::I32(_) => "I32".to_string(),
            VariableData::I64(_) => "I64".to_string(),
        }
    }
}

impl From<Tensor<f32>> for VariableData {
    fn from(tensor: Tensor<f32>) -> Self {
        Self::F32(Box::new(tensor))
    }
}

impl From<Tensor<f64>> for VariableData {
    fn from(tensor: Tensor<f64>) -> Self {
        Self::F64(Box::new(tensor))
    }
}

impl From<Tensor<usize>> for VariableData {
    fn from(tensor: Tensor<usize>) -> Self {
        Self::USIZE(Box::new(tensor))
    }
}

impl From<Tensor<i32>> for VariableData {
    fn from(tensor: Tensor<i32>) -> Self {
        Self::I32(Box::new(tensor))
    }
}

impl From<Tensor<i64>> for VariableData {
    fn from(tensor: Tensor<i64>) -> Self {
        Self::I64(Box::new(tensor))
    }
}
