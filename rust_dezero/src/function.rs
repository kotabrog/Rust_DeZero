pub mod sample;

use std::collections::HashMap;
use crate::variable::VariableTable;

/// Function info
/// 
/// # Fields
/// 
/// * `id` - function ID
/// * `inputs` - function input
/// * `outputs` - function output
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub id: usize,
    pub inputs: Option<Vec<usize>>,
    pub outputs: Option<Vec<usize>>,
    pub generation: usize,
}

impl FunctionInfo {
    /// Create a new FunctionInfo
    pub fn new() -> Self {
        Self { id: usize::MAX, inputs: None, outputs: None, generation: 0 }
    }

    /// Get the input variable ID list
    /// 
    /// # Panics
    /// 
    /// Panics if inputs is None
    pub fn get_inputs_unchecked(&self) -> &Vec<usize> {
        self.inputs.as_ref().expect("inputs is None")
    }

    /// Get the output variable ID list
    /// 
    /// # Panics
    /// 
    /// Panics if outputs is None
    pub fn get_outputs_unchecked(&self) -> &Vec<usize> {
        self.outputs.as_ref().expect("outputs is None")
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

    /// Get the input variable ID list
    pub fn get_input(&self) -> Option<&Vec<usize>> {
        self.info.inputs.as_ref()
    }

    /// Set the input
    /// 
    /// # Arguments
    /// 
    /// * `inputs` - Input Variable ID list
    pub fn set_input(&mut self, inputs: Vec<usize>) {
        self.info.inputs = Some(inputs);
    }

    /// Get the output variable ID list
    pub fn get_output(&self) -> Option<&Vec<usize>> {
        self.info.outputs.as_ref()
    }

    /// Set the output
    /// 
    /// # Arguments
    /// 
    /// * `outputs` - Output Variable ID list
    pub fn set_output(&mut self, outputs: Vec<usize>) {
        self.info.outputs = Some(outputs);
    }

    /// Get the generation
    /// 
    pub fn get_generation(&self) -> usize {
        self.info.generation
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
    /// * `inputs` - Input Variable ID list
    /// * `variables` - Variable table
    /// 
    /// # Returns
    /// 
    /// * `Vec<usize>` - Output Variable ID list
    pub fn call_mut(&mut self, inputs: Vec<usize>, variables: &mut VariableTable) -> Vec<usize> {
        let outputs = self.function.forward(&self.info, &inputs, variables);
        self.info.inputs = Some(inputs);
        self.info.generation = self.info.inputs.as_ref().unwrap()
            .iter().map(|x| variables.get(*x)
            .expect("FunctionWrapper::call_mut: Variable not found")
            .get_generation())
            .max().expect("FunctionWrapper::call_mut: Generation not found");
        let generation = self.info.generation.checked_add(1).expect("FunctionWrapper::call_mut: Overflow");
        for y in outputs.iter() {
            variables.get_mut(*y).unwrap().set_creator(self.info.id, generation);
        }
        self.info.outputs = Some(outputs.clone());
        outputs
    }

    /// backward the function
    /// 
    /// # Arguments
    /// 
    /// * `variables` - Variable table
    /// 
    /// # Returns
    /// 
    /// * `usize` - Input Variable ID
    pub fn backward(&mut self, variables: &mut VariableTable) -> Vec<usize> {
        self.function.backward(&self.info, variables)
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
    fn forward(&self, info: &FunctionInfo, inputs: &Vec<usize>, variables: &mut VariableTable) -> Vec<usize>;
    fn backward(&self, info: &FunctionInfo, variables: &mut VariableTable) -> Vec<usize>;
}
