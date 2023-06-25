use std::collections::HashMap;
use super::{Function, FunctionContents};
use crate::variable::VariableTable;

/// Function table
/// 
/// # Fields
/// 
/// * `table` - Function table
/// * `id_max` - The next id to adopt
pub struct FunctionTable {
    table: HashMap<usize, Box<Function>>,
    id_max: usize,
}

impl FunctionTable {
    /// Create a new FunctionTable instance.
    pub fn new() -> Self {
        Self { table: HashMap::new(), id_max: 0 }
    }

    /// Insert a new function into the table.
    /// 
    /// # Arguments
    /// 
    /// * `function` - Function
    fn insert(&mut self, function: Function) -> usize {
        let id = self.id_max;
        self.id_max += 1;
        self.table.insert(id, Box::new(function));
        id
    }

    /// Get the function with the specified id.
    pub fn get(&self, id: usize) -> Option<&Function> {
        self.table.get(&id).map(|v| v.as_ref())
    }

    /// Get the mutable function with the specified id.
    pub fn get_mut(&mut self, id: usize) -> Option<&mut Function> {
        self.table.get_mut(&id).map(|v| v.as_mut())
    }

    /// Generate a new function from the specified data and insert it into the table.
    pub fn generate_function_from_function_contents(&mut self, function_contents: Box<dyn FunctionContents>) -> usize {
        self.insert(Function::new(self.id_max, function_contents))
    }

    /// Forward the function with the specified id.
    /// 
    /// # Arguments
    /// 
    /// * `id` - Function ID
    /// * `inputs` - Input variable IDs
    /// * `variable_table` - Variable table
    /// * `no_grad` - Whether to calculate the gradient
    /// 
    /// # Returns
    /// 
    /// * Output variable IDs
    pub fn forward(&mut self, id: usize, inputs: Vec<usize>, variable_table: &mut VariableTable, no_grad: bool) -> Vec<usize> {
        let function = self.get_mut(id).expect("Invalid function id");
        function.forward(inputs, variable_table, no_grad)
    }

    /// Backward the function with the specified id.
    /// 
    /// # Arguments
    /// 
    /// * `id` - Function ID
    /// * `variable_table` - Variable table
    /// 
    /// # Returns
    /// 
    /// * Input variable IDs
    pub fn backward(&mut self, id: usize, variable_table: &mut VariableTable) -> Vec<usize> {
        let function = self.get(id).expect("Invalid function id");
        let backward = function.get_backward();
        backward(id, self, variable_table)
    }
}
