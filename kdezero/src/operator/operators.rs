mod getter;
mod check_state;
mod setter;
mod utility;

use std::collections::HashMap;
use super::Operator;

#[derive(Clone)]
pub struct Operators {
    operators: HashMap<usize, Operator>,
    next_id: usize,
}

impl Operators {
    pub fn new() -> Self {
        Self {
            operators: HashMap::new(),
            next_id: 0,
        }
    }
}
