use anyhow::Result;
use ktensor::Tensor;
use super::VariableData;
use crate::error::KdezeroError;

impl VariableData {
    pub fn ones_like(variable_data: &Self) -> Result<Self> {
        match variable_data {
            VariableData::F32(tensor) =>
                Ok(VariableData::F32(Box::new(Tensor::ones_like(tensor)))),
            VariableData::F64(tensor) =>
                Ok(VariableData::F64(Box::new(Tensor::ones_like(tensor)))),
            VariableData::USIZE(tensor) =>
                Ok(VariableData::USIZE(Box::new(Tensor::ones_like(tensor)))),
            VariableData::I32(tensor) =>
                Ok(VariableData::I32(Box::new(Tensor::ones_like(tensor)))),
            VariableData::I64(tensor) =>
                Ok(VariableData::I64(Box::new(Tensor::ones_like(tensor)))),
            _ => Err(KdezeroError::NotImplementedTypeError(
                "ones_like".to_string(),
                variable_data.to_string()
            ).into()),
        }
    }

    pub fn zeros_like(variable_data: &Self) -> Result<Self> {
        match variable_data {
            VariableData::F32(tensor) =>
                Ok(VariableData::F32(Box::new(Tensor::zeros_like(tensor)))),
            VariableData::F64(tensor) =>
                Ok(VariableData::F64(Box::new(Tensor::zeros_like(tensor)))),
            VariableData::USIZE(tensor) =>
                Ok(VariableData::USIZE(Box::new(Tensor::zeros_like(tensor)))),
            VariableData::I32(tensor) =>
                Ok(VariableData::I32(Box::new(Tensor::zeros_like(tensor)))),
            VariableData::I64(tensor) =>
                Ok(VariableData::I64(Box::new(Tensor::zeros_like(tensor)))),
            _ => Err(KdezeroError::NotImplementedTypeError(
                "zeros_like".to_string(),
                variable_data.to_string()
            ).into()),
        }
    }

    pub fn scalar(variable_data: &Self, scalar: f64) -> Result<Self> {
        match variable_data {
            VariableData::F32(_) =>
                Ok(VariableData::F32(Box::new(Tensor::new(vec![scalar as f32], vec![])?))),
            VariableData::F64(_) =>
                Ok(VariableData::F64(Box::new(Tensor::new(vec![scalar], vec![])?))),
            VariableData::USIZE(_) =>
                Ok(VariableData::USIZE(Box::new(Tensor::new(vec![scalar as usize], vec![])?))),
            VariableData::I32(_) =>
                Ok(VariableData::I32(Box::new(Tensor::new(vec![scalar as i32], vec![])?))),
            VariableData::I64(_) =>
                Ok(VariableData::I64(Box::new(Tensor::new(vec![scalar as i64], vec![])?))),
            _ => Err(KdezeroError::NotImplementedTypeError(
                "scalar".to_string(),
                variable_data.to_string()
            ).into()),
        }
    }
}
