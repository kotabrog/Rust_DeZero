mod getter;
mod check_state;
mod setter;
mod utility;

use std::collections::HashMap;
use super::{Variable, VariableData};

#[derive(Debug, Clone)]
pub struct Variables {
    variables: HashMap<usize, Variable>,
    next_id: usize,
}

impl Variables {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            next_id: 0,
        }
    }
}
