use std::collections::HashMap;
use super::{Function, FunctionContents};
use crate::variable::VariableTable;

pub struct FunctionTable {
    table: HashMap<usize, Box<Function>>,
    id_max: usize,
}

impl FunctionTable {
    pub fn new() -> Self {
        Self { table: HashMap::new(), id_max: 0 }
    }

    fn insert(&mut self, function: Function) -> usize {
        let id = self.id_max;
        self.id_max += 1;
        self.table.insert(id, Box::new(function));
        id
    }

    pub fn get(&self, id: usize) -> Option<&Function> {
        self.table.get(&id).map(|v| v.as_ref())
    }

    pub fn get_mut(&mut self, id: usize) -> Option<&mut Function> {
        self.table.get_mut(&id).map(|v| v.as_mut())
    }

    pub fn generate_function_from_function_contents(&mut self, function_contents: Box<dyn FunctionContents>) -> usize {
        self.insert(Function::new(self.id_max, function_contents))
    }

    pub fn get_function(&self, id: usize) -> Option<&Function> {
        self.table.get(&id).map(|v| v.as_ref())
    }

    pub fn get_mut_function(&mut self, id: usize) -> Option<&mut Function> {
        self.table.get_mut(&id).map(|v| v.as_mut())
    }

    pub fn forward(&mut self, id: usize, inputs: Vec<usize>, variable_table: &mut VariableTable, no_grad: bool) -> Vec<usize> {
        let function = self.get_mut_function(id).expect("Invalid function id");
        function.forward(inputs, variable_table, no_grad)
    }

    pub fn backward(&mut self, id: usize, variable_table: &mut VariableTable) -> Vec<usize> {
        let function = self.get_function(id).expect("Invalid function id");
        let backward = function.get_backward();
        backward(id, self, variable_table)
    }
}
