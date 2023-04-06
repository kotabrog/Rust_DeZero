pub mod sample;

use std::collections::HashMap;
use crate::variable::VariableTable;

/// Function info
/// 
/// # Fields
/// 
/// * `id` - function ID
/// * `input` - function input
/// * `output` - function output
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub id: usize,
    pub input: Option<usize>,
    pub output: Option<usize>,
}

impl FunctionInfo {
    /// Create a new FunctionInfo
    pub fn new() -> Self {
        Self { id: usize::MAX, input: None, output: None }
    }
}

/// FunctionWrapper
/// 
/// # Fields
/// 
/// * `info` - Function info
/// * `function` - Function
pub struct FunctionWrapper
{
    info: FunctionInfo,
    function: Box<dyn Function>,
}

impl FunctionWrapper {
    /// Create a new FunctionWrapper
    /// 
    /// # Arguments
    /// 
    /// * `function` - Function
    pub fn new(function: Box<dyn Function>) -> Self {
        Self { info: FunctionInfo::new(), function }
    }

    /// Get the function info
    pub fn get_info(&self) -> &FunctionInfo {
        &self.info
    }

    /// Get the ID
    pub fn get_id(&self) -> usize {
        self.info.id
    }

    /// Set the ID
    /// 
    /// # Arguments
    /// 
    /// * `id` - ID
    pub fn set_id(&mut self, id: usize) {
        self.info.id = id;
    }

    /// Get the input variable ID
    pub fn get_input(&self) -> Option<usize> {
        self.info.input
    }

    /// Set the input
    /// 
    /// # Arguments
    /// 
    /// * `input` - Input Variable ID
    pub fn set_input(&mut self, input: usize) {
        self.info.input = Some(input);
    }

    /// Get the output variable ID
    pub fn get_output(&self) -> Option<usize> {
        self.info.output
    }

    /// Set the output
    /// 
    /// # Arguments
    /// 
    /// * `output` - Output Variable ID
    pub fn set_output(&mut self, output: usize) {
        self.info.output = Some(output);
    }

    /// Get the function
    pub fn get_function(&self) -> &Box<dyn Function> {
        &self.function
    }

    /// Get the function (mutable)
    pub fn get_function_mut(&mut self) -> &mut Box<dyn Function> {
        &mut self.function
    }

    /// Call the function
    /// 
    /// # Arguments
    /// 
    /// * `input` - Input Variable ID
    /// * `variables` - Variable table
    /// 
    /// # Returns
    /// 
    /// * `usize` - Output Variable ID
    pub fn call_mut(&mut self, input: usize, variables: &mut VariableTable) -> usize {
        let y = self.function.forward(&self.info, &input, variables);
        self.info.input = Some(input);
        self.info.output = Some(y);
        variables.get_mut(y).unwrap().set_creator(self.info.id);
        y
    }

    /// backward the function
    /// 
    /// # Arguments
    /// 
    /// * `grad_id` - Gradient Variable ID
    /// * `variables` - Variable table
    /// 
    /// # Returns
    /// 
    /// * `usize` - Input Variable ID
    pub fn backward(&mut self, grad_id: usize, variables: &mut VariableTable) -> usize {
        self.function.backward(&self.info, &grad_id, variables)
    }
}

/// Function table
/// 
/// # Fields
/// 
/// * `table` - Function table
/// * `id_max` - Maximum ID
pub struct FunctionTable {
    table: HashMap<usize, Box<FunctionWrapper>>,
    id_max: usize,
}

impl FunctionTable {
    /// Create a new FunctionTable
    pub fn new() -> Self {
        Self { table: HashMap::new(), id_max: 0 }
    }

    /// Add a function wrapper
    /// 
    /// # Arguments
    /// 
    /// * `func` - Function wrapper
    /// 
    /// # Returns
    /// 
    /// * `usize` - function ID
    pub fn add(&mut self, func: FunctionWrapper) -> usize {
        let id = self.id_max;
        self.table.insert(self.id_max, Box::new(func));
        self.table.get_mut(&id).unwrap().set_id(id);
        self.id_max = self.id_max.checked_add(1).expect("FunctionTable::add: Overflow");
        id
    }

    /// Add a function
    /// 
    /// # Arguments
    /// 
    /// * `func` - Function
    /// 
    /// # Returns
    /// 
    /// * `usize` - function ID
    pub fn add_function(&mut self, func: Box<dyn Function>) -> usize {
        self.add(FunctionWrapper::new(func))
    }

    /// Get the function wrapper
    /// 
    /// # Arguments
    /// 
    /// * `id` - function ID
    pub fn get(&self, id: usize) -> Option<&Box<FunctionWrapper>> {
        self.table.get(&id)
    }

    /// Get the function wrapper (mutable)
    /// 
    /// # Arguments
    /// 
    /// * `id` - function ID
    pub fn get_mut(&mut self, id: usize) -> Option<&mut Box<FunctionWrapper>> {
        self.table.get_mut(&id)
    }
}

/// Function
/// 
/// # Methods
/// 
/// * `forward` - Forward propagation
/// * `backward` - Backward propagation
pub trait Function {
    fn forward(&self, info: &FunctionInfo, input: &usize, variables: &mut VariableTable) -> usize;
    fn backward(&self, info: &FunctionInfo, grad: &usize, variables: &mut VariableTable) -> usize;
}
