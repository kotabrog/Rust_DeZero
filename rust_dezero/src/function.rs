pub mod function_table;
pub mod operator;

pub use function_table::FunctionTable;

use crate::variable::VariableTable;

/// Function information
/// 
/// # Fields
/// 
/// * `id` - Function ID
/// * `inputs` - Input variable IDs
/// * `outputs` - Output variable IDs
/// * `generation` - Generation of this function
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub id: usize,
    pub inputs: Option<Vec<usize>>,
    pub outputs: Option<Vec<usize>>,
    pub generation: usize,
}

impl FunctionInfo {
    /// Create a new FunctionInfo instance.
    /// 
    /// # Arguments
    /// 
    /// * `id` - Function ID
    pub fn new(id: usize) -> Self {
        Self { id, inputs: None, outputs: None, generation: 0 }
    }
}

/// Function
/// 
/// # Fields
/// 
/// * `info` - Function information
/// * `function` - Function contents
pub struct Function {
    info: FunctionInfo,
    function: Box<dyn FunctionContents>
}

impl Function {
    /// Create a new Function instance.
    /// 
    /// # Arguments
    /// 
    /// * `id` - Function ID
    /// * `function` - Function contents
    pub fn new(id: usize, function: Box<dyn FunctionContents>) -> Self {
        Self { info: FunctionInfo::new(id), function }
    }

    /// Get the function information.
    pub fn get_info(&self) -> &FunctionInfo {
        &self.info
    }

    /// Get function ID.
    pub fn get_id(&self) -> usize {
        self.info.id
    }

    /// Get the function inputs.
    pub fn get_inputs(&self) -> Option<&Vec<usize>> {
        self.info.inputs.as_ref()
    }

    /// Get the function outputs.
    pub fn get_outputs(&self) -> Option<&Vec<usize>> {
        self.info.outputs.as_ref()
    }

    /// Set the function outputs.
    pub fn set_outputs(&mut self, outputs: Vec<usize>) {
        self.info.outputs = Some(outputs);
    }

    /// Get the function generation.
    pub fn get_generation(&self) -> usize {
        self.info.generation
    }

    /// forward
    /// 
    /// # Arguments
    /// 
    /// * `inputs` - Input variable IDs
    /// * `variable_table` - Variable table
    /// * `no_grad` - Whether to calculate gradients
    /// 
    /// # Returns
    /// 
    /// * `outputs` - Output variable IDs
    /// 
    /// # Panics
    /// 
    /// * `Invalid variable id` - If the variable ID is invalid
    /// * `Inputs are empty` - If the inputs are empty
    /// * `Generation overflow` - If the generation overflows
    pub fn forward(&mut self, inputs: Vec<usize>, variable_table: &mut VariableTable, no_grad: bool) -> Vec<usize> {
        let outputs = self.function.forward(&self.info, &inputs, variable_table);
        if !no_grad {
            self.info.inputs = Some(inputs);
            self.info.outputs = Some(outputs.clone());
            self.info.generation = self.info.inputs.as_ref().unwrap()
                .iter().map(
                    |&id| variable_table.get(id).expect("Invalid variable id")
                    .get_generation())
                .max().expect("Inputs are empty");
            let new_generation = self.info.generation.checked_add(1).expect("Generation overflow");
            for output in &outputs {
                variable_table.get_mut(*output).expect("Invalid variable id")
                    .set_creator(self.info.id, new_generation);
            }
        }
        outputs
    }

    /// Get the backward function.
    pub fn get_backward(&self) -> fn(usize, &mut FunctionTable, &mut VariableTable) -> Vec<usize> {
        self.function.get_backward()
    }

    /// Get the dot string
    pub fn to_dot_string(&self) -> String {
        let mut dot_string = String::new();
        dot_string.push_str(&format!("func_{} [label=\"{}\", color=lightblue, style=filled, shape=box]\n", self.info.id, self.function.name()));
        if let Some(inputs) = self.info.inputs.as_ref() {
            for input in inputs.iter() {
                dot_string.push_str(&format!("var_{} -> func_{};\n", input, self.info.id));
            }
        }
        if let Some(outputs) = self.info.outputs.as_ref() {
            for output in outputs.iter() {
                dot_string.push_str(&format!("func_{} -> var_{};\n", self.info.id, output));
            }
        }
        dot_string
    }
}

pub trait FunctionContents {
    fn name(&self) -> &str;
    fn forward(&self, info: &FunctionInfo, inputs: &Vec<usize>, variable_table: &mut VariableTable) -> Vec<usize>;
    fn get_backward(&self) -> fn(usize, &mut FunctionTable, &mut VariableTable) -> Vec<usize>;
}
