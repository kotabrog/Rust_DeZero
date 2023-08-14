use anyhow::Result;
use super::VariableData;
use crate::error::KdezeroError;

impl VariableData {
    pub fn pow(&self, power: u32) -> Result<Self> {
        match self {
            VariableData::F32(tensor) =>
                Ok(VariableData::F32(
                    Box::new(tensor.clone().powi(
                        power.try_into()
                            .map_err(|_| KdezeroError::OverflowError(
                                "u32 to i32".to_string(),
                            ))?)))),
            VariableData::F64(tensor) =>
                Ok(VariableData::F64(
                    Box::new(tensor.clone().powi(
                        power.try_into()
                            .map_err(|_| KdezeroError::OverflowError(
                                "u32 to i32".to_string(),
                            ))?)))),
            VariableData::USIZE(tensor) =>
                Ok(VariableData::USIZE(Box::new(tensor.clone().pow(power)))),
            VariableData::I32(tensor) =>
                Ok(VariableData::I32(Box::new(tensor.clone().pow(power)))),
            VariableData::I64(tensor) =>
                Ok(VariableData::I64(Box::new(tensor.clone().pow(power)))),
            _ => Err(KdezeroError::NotImplementedTypeError(
                self.to_string(),
                "Square".to_string()
            ).into()),
        }
    }
}
