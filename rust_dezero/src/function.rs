pub mod function_table;
pub mod operator;

pub use function_table::FunctionTable;

use crate::variable::VariableTable;

#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub id: usize,
    pub inputs: Option<Vec<usize>>,
    pub outputs: Option<Vec<usize>>,
    pub generation: usize,
}

impl FunctionInfo {
    pub fn new(id: usize) -> Self {
        Self { id, inputs: None, outputs: None, generation: 0 }
    }
}

pub struct Function {
    info: FunctionInfo,
    function: Box<dyn FunctionContents>
}

impl Function {
    pub fn new(id: usize, function: Box<dyn FunctionContents>) -> Self {
        Self { info: FunctionInfo::new(id), function }
    }

    pub fn get_info(&self) -> &FunctionInfo {
        &self.info
    }

    pub fn get_id(&self) -> usize {
        self.info.id
    }

    pub fn get_inputs(&self) -> Option<&Vec<usize>> {
        self.info.inputs.as_ref()
    }

    pub fn get_outputs(&self) -> Option<&Vec<usize>> {
        self.info.outputs.as_ref()
    }

    pub fn set_outputs(&mut self, outputs: Vec<usize>) {
        self.info.outputs = Some(outputs);
    }

    pub fn get_generation(&self) -> usize {
        self.info.generation
    }

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

    pub fn get_backward(&self) -> fn(usize, &mut FunctionTable, &mut VariableTable) -> Vec<usize> {
        self.function.get_backward()
    }
}

pub trait FunctionContents {
    fn name(&self) -> &str;
    fn forward(&self, info: &FunctionInfo, inputs: &Vec<usize>, variable_table: &mut VariableTable) -> Vec<usize>;
    fn get_backward(&self) -> fn(usize, &mut FunctionTable, &mut VariableTable) -> Vec<usize>;
}
