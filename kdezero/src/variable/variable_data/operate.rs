use anyhow::Result;
use super::VariableData;
use crate::error::KdezeroError;

impl VariableData {
    pub fn neg(&self) -> Result<Self> {
        match self {
            VariableData::F32(tensor) =>
                Ok(VariableData::F32(Box::new(-&**tensor))),
            VariableData::F64(tensor) =>
                Ok(VariableData::F64(Box::new(-&**tensor))),
            VariableData::I32(tensor) =>
                Ok(VariableData::I32(Box::new(-&**tensor))),
            VariableData::I64(tensor) =>
                Ok(VariableData::I64(Box::new(-&**tensor))),
            _ => Err(KdezeroError::NotImplementedTypeError(
                "neg".to_string(),
                self.to_string()
            ).into()),
        }
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (VariableData::F32(tensor1), VariableData::F32(tensor2)) =>
                Ok(VariableData::F32(Box::new(&**tensor1 + &**tensor2))),
            (VariableData::F64(tensor1), VariableData::F64(tensor2)) =>
                Ok(VariableData::F64(Box::new(&**tensor1 + &**tensor2))),
            (VariableData::USIZE(tensor1), VariableData::USIZE(tensor2)) =>
                Ok(VariableData::USIZE(Box::new(&**tensor1 + &**tensor2))),
            (VariableData::I32(tensor1), VariableData::I32(tensor2)) =>
                Ok(VariableData::I32(Box::new(&**tensor1 + &**tensor2))),
            (VariableData::I64(tensor1), VariableData::I64(tensor2)) =>
                Ok(VariableData::I64(Box::new(&**tensor1 + &**tensor2))),
            _ => Err(KdezeroError::NotImplementedTypeError(
                "add".to_string(),
                format!("{} and {}", self.to_string(), other.to_string())
            ).into()),
        }
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (VariableData::F32(tensor1), VariableData::F32(tensor2)) =>
                Ok(VariableData::F32(Box::new(&**tensor1 - &**tensor2))),
            (VariableData::F64(tensor1), VariableData::F64(tensor2)) =>
                Ok(VariableData::F64(Box::new(&**tensor1 - &**tensor2))),
            (VariableData::USIZE(tensor1), VariableData::USIZE(tensor2)) =>
                Ok(VariableData::USIZE(Box::new(&**tensor1 - &**tensor2))),
            (VariableData::I32(tensor1), VariableData::I32(tensor2)) =>
                Ok(VariableData::I32(Box::new(&**tensor1 - &**tensor2))),
            (VariableData::I64(tensor1), VariableData::I64(tensor2)) =>
                Ok(VariableData::I64(Box::new(&**tensor1 - &**tensor2))),
            _ => Err(KdezeroError::NotImplementedTypeError(
                "sub".to_string(),
                format!("{} and {}", self.to_string(), other.to_string())
            ).into()),
        }
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (VariableData::F32(tensor1), VariableData::F32(tensor2)) =>
                Ok(VariableData::F32(Box::new(&**tensor1 * &**tensor2))),
            (VariableData::F64(tensor1), VariableData::F64(tensor2)) =>
                Ok(VariableData::F64(Box::new(&**tensor1 * &**tensor2))),
            (VariableData::USIZE(tensor1), VariableData::USIZE(tensor2)) =>
                Ok(VariableData::USIZE(Box::new(&**tensor1 * &**tensor2))),
            (VariableData::I32(tensor1), VariableData::I32(tensor2)) =>
                Ok(VariableData::I32(Box::new(&**tensor1 * &**tensor2))),
            (VariableData::I64(tensor1), VariableData::I64(tensor2)) =>
                Ok(VariableData::I64(Box::new(&**tensor1 * &**tensor2))),
            _ => Err(KdezeroError::NotImplementedTypeError(
                "mul".to_string(),
                format!("{} and {}", self.to_string(), other.to_string())
            ).into()),
        }
    }

    pub fn div(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (VariableData::F32(tensor1), VariableData::F32(tensor2)) =>
                Ok(VariableData::F32(Box::new(&**tensor1 / &**tensor2))),
            (VariableData::F64(tensor1), VariableData::F64(tensor2)) =>
                Ok(VariableData::F64(Box::new(&**tensor1 / &**tensor2))),
            (VariableData::USIZE(tensor1), VariableData::USIZE(tensor2)) =>
                Ok(VariableData::USIZE(Box::new(&**tensor1 / &**tensor2))),
            (VariableData::I32(tensor1), VariableData::I32(tensor2)) =>
                Ok(VariableData::I32(Box::new(&**tensor1 / &**tensor2))),
            (VariableData::I64(tensor1), VariableData::I64(tensor2)) =>
                Ok(VariableData::I64(Box::new(&**tensor1 / &**tensor2))),
            _ => Err(KdezeroError::NotImplementedTypeError(
                "div".to_string(),
                format!("{} and {}", self.to_string(), other.to_string())
            ).into()),
        }
    }

    pub fn scalar_add(&self, value: f64) -> Result<Self> {
        match self {
            VariableData::F32(tensor) =>
                Ok(VariableData::F32(Box::new(&**tensor + (value as f32)))),
            VariableData::F64(tensor) =>
                Ok(VariableData::F64(Box::new(&**tensor + value))),
            VariableData::USIZE(tensor) =>
                Ok(VariableData::USIZE(Box::new(&**tensor + (value as usize)))),
            VariableData::I32(tensor) =>
                Ok(VariableData::I32(Box::new(&**tensor + (value as i32)))),
            VariableData::I64(tensor) =>
                Ok(VariableData::I64(Box::new(&**tensor + (value as i64)))),
            _ => Err(KdezeroError::NotImplementedTypeError(
                "scalar_add".to_string(),
                self.to_string()
            ).into()),
        }
    }

    pub fn scalar_mul(&self, value: f64) -> Result<Self> {
        match self {
            VariableData::F32(tensor) =>
                Ok(VariableData::F32(Box::new(&**tensor * (value as f32)))),
            VariableData::F64(tensor) =>
                Ok(VariableData::F64(Box::new(&**tensor * value))),
            VariableData::USIZE(tensor) =>
                Ok(VariableData::USIZE(Box::new(&**tensor * (value as usize)))),
            VariableData::I32(tensor) =>
                Ok(VariableData::I32(Box::new(&**tensor * (value as i32)))),
            VariableData::I64(tensor) =>
                Ok(VariableData::I64(Box::new(&**tensor * (value as i64)))),
            _ => Err(KdezeroError::NotImplementedTypeError(
                "scalar_mul".to_string(),
                self.to_string()
            ).into()),
        }
    }

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
                "pow".to_string(),
                self.to_string()
            ).into()),
        }
    }

    pub fn exp(&self) -> Result<Self> {
        match self {
            VariableData::F32(tensor) =>
                Ok(VariableData::F32(Box::new(tensor.exp()))),
            VariableData::F64(tensor) =>
                Ok(VariableData::F64(Box::new(tensor.exp()))),
            _ => Err(KdezeroError::NotImplementedTypeError(
                "exp".to_string(),
                self.to_string()
            ).into()),
        }
    }

    pub fn sin(&self) -> Result<Self> {
        match self {
            VariableData::F32(tensor) =>
                Ok(VariableData::F32(Box::new(tensor.sin()))),
            VariableData::F64(tensor) =>
                Ok(VariableData::F64(Box::new(tensor.sin()))),
            _ => Err(KdezeroError::NotImplementedTypeError(
                "sin".to_string(),
                self.to_string()
            ).into()),
        }
    }

    pub fn cos(&self) -> Result<Self> {
        match self {
            VariableData::F32(tensor) =>
                Ok(VariableData::F32(Box::new(tensor.cos()))),
            VariableData::F64(tensor) =>
                Ok(VariableData::F64(Box::new(tensor.cos()))),
            _ => Err(KdezeroError::NotImplementedTypeError(
                "cos".to_string(),
                self.to_string()
            ).into()),
        }
    }

    pub fn tanh(&self) -> Result<Self> {
        match self {
            VariableData::F32(tensor) =>
                Ok(VariableData::F32(Box::new(tensor.tanh()))),
            VariableData::F64(tensor) =>
                Ok(VariableData::F64(Box::new(tensor.tanh()))),
            _ => Err(KdezeroError::NotImplementedTypeError(
                "tanh".to_string(),
                self.to_string()
            ).into()),
        }
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        match self {
            VariableData::F32(tensor) =>
                Ok(VariableData::F32(Box::new(tensor.reshape(shape)?))),
            VariableData::F64(tensor) =>
                Ok(VariableData::F64(Box::new(tensor.reshape(shape)?))),
            VariableData::USIZE(tensor) =>
                Ok(VariableData::USIZE(Box::new(tensor.reshape(shape)?))),
            VariableData::I32(tensor) =>
                Ok(VariableData::I32(Box::new(tensor.reshape(shape)?))),
            VariableData::I64(tensor) =>
                Ok(VariableData::I64(Box::new(tensor.reshape(shape)?))),
            _ => Err(KdezeroError::NotImplementedTypeError(
                "reshape".to_string(),
                self.to_string()
            ).into()),
        }
    }

    pub fn transpose(&self) -> Result<Self> {
        match self {
            VariableData::F32(tensor) =>
                Ok(VariableData::F32(Box::new(tensor.transpose()))),
            VariableData::F64(tensor) =>
                Ok(VariableData::F64(Box::new(tensor.transpose()))),
            VariableData::USIZE(tensor) =>
                Ok(VariableData::USIZE(Box::new(tensor.transpose()))),
            VariableData::I32(tensor) =>
                Ok(VariableData::I32(Box::new(tensor.transpose()))),
            VariableData::I64(tensor) =>
                Ok(VariableData::I64(Box::new(tensor.transpose()))),
            _ => Err(KdezeroError::NotImplementedTypeError(
                "transpose".to_string(),
                self.to_string()
            ).into()),
        }
    }
}
