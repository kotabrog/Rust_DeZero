use anyhow::Result;
use num_traits::NumCast;
use ktensor::Tensor;
use super::VariableData;
use crate::error::KdezeroError;

impl VariableData {
    pub fn check_type(&self, other: &Self) -> Result<()> {
        match (self, other) {
            (VariableData::F32(_), VariableData::F32(_)) => Ok(()),
            (VariableData::F64(_), VariableData::F64(_)) => Ok(()),
            (VariableData::USIZE(_), VariableData::USIZE(_)) => Ok(()),
            (VariableData::I32(_), VariableData::I32(_)) => Ok(()),
            (VariableData::I64(_), VariableData::I64(_)) => Ok(()),
            (VariableData::Bool(_), VariableData::Bool(_)) => Ok(()),
            _ => Err(KdezeroError::NotCollectTypeError(
                self.to_string(),
                other.to_string(),
            ).into()),
        }
    }

    pub fn as_type_from_other<T: NumCast>(tensor: Tensor<T>, other: &Self) -> Result<Self> {
        match other {
            VariableData::F32(_) => Ok(tensor.as_type::<f32>()?.into()),
            VariableData::F64(_) => Ok(tensor.as_type::<f64>()?.into()),
            VariableData::USIZE(_) => Ok(tensor.as_type::<usize>()?.into()),
            VariableData::I32(_) => Ok(tensor.as_type::<i32>()?.into()),
            VariableData::I64(_) => Ok(tensor.as_type::<i64>()?.into()),
            _ => Err(KdezeroError::NotImplementedTypeError(
                other.to_string(), "as_type_from_other".to_string()
            ).into()),
        }
    }

    pub fn to_f32_tensor(&self) -> Result<&Tensor<f32>> {
        match self {
            VariableData::F32(tensor) => Ok(tensor),
            _ => Err(KdezeroError::NotCollectTypeError(
                self.to_string(),
                "F32".to_string(),
            ).into()),
        }
    }

    pub fn to_f64_tensor(&self) -> Result<&Tensor<f64>> {
        match self {
            VariableData::F64(tensor) => Ok(tensor),
            _ => Err(KdezeroError::NotCollectTypeError(
                self.to_string(),
                "F64".to_string(),
            ).into()),
        }
    }

    pub fn to_usize_tensor(&self) -> Result<&Tensor<usize>> {
        match self {
            VariableData::USIZE(tensor) => Ok(tensor),
            _ => Err(KdezeroError::NotCollectTypeError(
                self.to_string(),
                "USIZE".to_string(),
            ).into()),
        }
    }

    pub fn to_i32_tensor(&self) -> Result<&Tensor<i32>> {
        match self {
            VariableData::I32(tensor) => Ok(tensor),
            _ => Err(KdezeroError::NotCollectTypeError(
                self.to_string(),
                "I32".to_string(),
            ).into()),
        }
    }

    pub fn to_i64_tensor(&self) -> Result<&Tensor<i64>> {
        match self {
            VariableData::I64(tensor) => Ok(tensor),
            _ => Err(KdezeroError::NotCollectTypeError(
                self.to_string(),
                "I64".to_string(),
            ).into()),
        }
    }

    pub fn to_bool_tensor(&self) -> Result<&Tensor<bool>> {
        match self {
            VariableData::Bool(tensor) => Ok(tensor),
            _ => Err(KdezeroError::NotCollectTypeError(
                self.to_string(),
                "Bool".to_string(),
            ).into()),
        }
    }
}
