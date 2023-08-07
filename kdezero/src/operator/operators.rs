use std::collections::HashMap;
use super::Operator;

pub struct Operators {
    operators: HashMap<usize, Operator>,
}

impl Operators {
    pub fn new() -> Self {
        Self {
            operators: HashMap::new(),
        }
    }

    pub fn get_operators(&self) -> &HashMap<usize, Operator> {
        &self.operators
    }

    pub fn add_operator(&mut self, operator: Operator) -> Option<Operator> {
        self.operators.insert(operator.get_id(), operator)
    }

    pub fn get_operator(&self, id: usize) -> Option<&Operator> {
        self.operators.get(&id)
    }
}
