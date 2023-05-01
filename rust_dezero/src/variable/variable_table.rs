mod function_generation_priority_queue;

use std::fs::File;
use std::io::prelude::*;
use std::process::Command;
use std::collections::{HashMap, HashSet};
use function_generation_priority_queue::FunctionGenerationPriorityQueue;
use super::{Variable, VariableContents};
use crate::{Tensor, function::{FunctionTable, operator::Add}};

/// Variable table
/// 
/// # Fields
/// 
/// * `table` - Variable table
/// * `id_max` - The next id to adopt
#[derive(Debug)]
pub struct VariableTable {
    table: HashMap<usize, Box<Variable>>,
    id_max: usize,
}

impl VariableTable {
    /// Create a new VariableTable instance.
    pub fn new() -> Self {
        Self { table: HashMap::new(), id_max: 0 }
    }

    /// Insert a new variable into the table.
    /// 
    /// # Arguments
    /// 
    /// * `variable` - Variable
    /// 
    /// # Returns
    /// 
    /// * Variable ID
    fn insert(&mut self, variable: Variable) -> usize {
        let id = self.id_max;
        self.id_max += 1;
        self.table.insert(id, Box::new(variable));
        id
    }

    /// Get the variable with the specified id.
    /// 
    /// # Arguments
    /// 
    /// * `id` - Variable ID
    pub fn get(&self, id: usize) -> Option<&Variable> {
        self.table.get(&id).map(|v| v.as_ref())
    }

    /// Get the mutable variable with the specified id.
    /// 
    /// # Arguments
    /// 
    /// * `id` - Variable ID
    pub fn get_mut(&mut self, id: usize) -> Option<&mut Variable> {
        self.table.get_mut(&id).map(|v| v.as_mut())
    }

    /// Generate a new variable from the specified data and insert it into the table.
    /// 
    /// # Arguments
    /// 
    /// * `data` - Variable contents
    /// * `name` - Variable name
    /// 
    /// # Returns
    /// 
    /// * Variable ID
    pub fn generate_variable_from_variable_contents(&mut self, data: VariableContents, name: &str) -> usize {
        self.insert(Variable::new(data, self.id_max, name))
    }

    /// Generate a new variable from the specified tensor and insert it into the table.
    /// 
    /// # Arguments
    /// 
    /// * `tensor` - Tensor
    /// * `name` - Variable name
    /// 
    /// # Returns
    /// 
    /// * Variable ID
    pub fn generate_variable_from_f64_tensor(&mut self, tensor: Tensor<f64>, name: &str) -> usize {
        self.generate_variable_from_variable_contents(VariableContents::F64(Box::new(tensor)), name)
    }

    /// Get the variable f64 contents of the specified variable id.
    /// 
    /// # Arguments
    /// 
    /// * `id` - Variable ID
    /// 
    /// # Returns
    /// 
    /// * variable f64 contents
    pub fn get_variable_contents_f64(&self, id: usize) -> Option<&Tensor<f64>> {
        self.table.get(&id).map(|v| v.to_f64_tensor()).flatten()
    }

    /// Get the variable grad id of the specified variable id.
    /// 
    /// # Arguments
    /// 
    /// * `id` - Variable ID
    /// 
    /// # Returns
    /// 
    /// * variable grad id
    pub fn get_variable_grad_id(&self, id: usize) -> Option<usize> {
        self.table.get(&id).map(|v| v.get_grad_id()).flatten()
    }

    /// Get the variable grad f64 contents of the specified variable id.
    /// 
    /// # Arguments
    /// 
    /// * `id` - Variable ID
    /// 
    /// # Returns
    /// 
    /// * variable grad f64 contents
    pub fn get_variable_grad_contents_f64(&self, id: usize) -> Option<&Tensor<f64>> {
        self.get_variable_grad_id(id)
            .map(|grad_id| self.get_variable_contents_f64(grad_id)).flatten()
    }

    /// Set the grad id of the specified variable id.
    /// 
    /// # Arguments
    /// 
    /// * `variable_id` - Variable ID
    /// * `grad_id` - Grad ID
    pub fn set_grad(&mut self, variable_id: usize, grad_id: usize) {
        let variable = self.get_mut(variable_id).expect("Invalid variable id");
        variable.set_grad_id(grad_id);
    }

    /// Set the grad f64 contents of the specified variable id.
    /// 
    /// # Arguments
    /// 
    /// * `variable_id` - Variable ID
    /// * `grad` - Grad f64 contents
    pub fn set_grad_from_f64_tensor(&mut self, variable_id: usize, grad: Tensor<f64>) {
        let grad_id = self.generate_variable_from_f64_tensor(grad, "");
        self.set_grad(variable_id, grad_id);
    }

    /// Set the grad f64 default contents of the specified variable id.
    /// 
    /// # Arguments
    /// 
    /// * `variable_id` - Variable ID
    pub fn set_grad_default(&mut self, variable_id: usize) {
        let variable = self.get(variable_id).expect("Invalid variable id");
        let grad = match variable.get_grad_id() {
            Some(_) => return,
            None => Tensor::ones_like(variable.to_f64_tensor().expect("Invalid variable data"))
        };
        self.set_grad_from_f64_tensor(variable_id, grad);
    }

    /// Sets the grad f64 default contents of the specified variable ids.
    pub fn sets_grad_default(&mut self, variable_ids: &Vec<usize>) {
        for &variable_id in variable_ids {
            self.set_grad_default(variable_id);
        }
    }

    /// Update the grad id of the specified variable id.
    /// 
    /// # Arguments
    /// 
    /// * `variable_id` - Variable ID
    /// * `grad_id` - Grad ID
    /// * `function_table` - Function table
    pub fn update_grad(&mut self, variable_id: usize, grad_id: usize, function_table: &mut FunctionTable) {
        let variable = self.get(variable_id).expect("Invalid variable id");
        let new_grad_id = match variable.get_grad_id() {
            Some(id) => {
                let add_id = function_table.generate_function_from_function_contents(Box::new(Add::new()));
                function_table.forward(add_id, vec![id, grad_id], self, false)[0]
            },
            None => grad_id,
        };
        self.set_grad(variable_id, new_grad_id);
    }

    /// Clear the grad of the specified variable id.
    /// 
    /// # Arguments
    /// 
    /// * `id` - Variable ID
    pub fn clear_grad(&mut self, id: usize) {
        self.get_mut(id).expect("Invalid variable id").clear_grad();
    }

    /// Clear the grads of the specified variable ids.
    /// 
    /// # Arguments
    /// 
    /// * `ids` - Variable IDs
    pub fn clear_grads(&mut self, ids: &Vec<usize>) {
        for &id in ids {
            self.clear_grad(id);
        }
    }

    /// Add function to queue for backward propagation.
    fn add_function_to_queue(&mut self, variable_id: usize,
            priority_queue: &mut FunctionGenerationPriorityQueue, function_table: &FunctionTable) {
        let variable = self.get(variable_id).expect("Invalid variable id");
        let function_id = match variable.get_creator() {
            Some(id) => id,
            None => return,
        };
        let function_generation = function_table.get(function_id).expect("Invalid function id")
            .get_generation();
        priority_queue.push(function_id, function_generation);
    }

    /// Add functions to queue for backward propagation.
    fn add_functions_to_queue(&mut self, variable_ids: &Vec<usize>,
            priority_queue: &mut FunctionGenerationPriorityQueue, function_table: &FunctionTable) {
        for &variable_id in variable_ids {
            self.add_function_to_queue(variable_id, priority_queue, function_table);
        }
    }

    /// Backward propagation.
    /// 
    /// # Arguments
    /// 
    /// * `ids` - Variable IDs
    /// * `function_table` - Function table
    /// * `retain_grad` - Whether to retain the grad
    /// 
    /// # Panics
    /// 
    /// * `Invalid function id` - If the function id is invalid
    /// * `Output not found` - If the output is not found
    /// * `Invalid variable id` - If the variable id is invalid
    pub fn backward(&mut self, ids: Vec<usize>, function_table: &mut FunctionTable, retain_grad: bool) {
        self.sets_grad_default(&ids);

        let mut function_queue = FunctionGenerationPriorityQueue::new();
        self.add_functions_to_queue(&ids, &mut function_queue, &function_table);

        while !function_queue.is_empty() {
            let function_id = function_queue.pop().unwrap();
            let input_ids = function_table.backward(function_id, self);
            self.add_functions_to_queue(&input_ids, &mut function_queue, &function_table);

            if !retain_grad {
                self.clear_grads(
                    function_table.get(function_id).expect("Invalid function id")
                        .get_outputs().expect("Output not found")
                )
            }
        }
    }

    /// Add a function to the function queue for dot graph
    fn add_function_for_dot_graph(&self, function_ids: &mut Vec<usize>, function_ids_set: &mut HashSet<usize>, id: usize) {
        let y = self.get(id).expect("Variable not found");
        let f_id = y.get_creator();
        let f_id = match f_id {
            Some(f_id) => f_id,
            None => return,
        };
        if function_ids_set.contains(&f_id) {
            return;
        }
        function_ids.push(f_id);
        function_ids_set.insert(f_id);
    }

    /// Add a variable to the variable queue for dot graph
    fn add_variable_for_dot_graph(&self, variable_ids_set: &mut HashSet<usize>, text: &mut String, id: usize) {
        if variable_ids_set.contains(&id) {
            return;
        }
        variable_ids_set.insert(id);
        let var = self.get(id).expect("Variable not found");
        text.push_str(&var.to_dot_string());
    }

    /// Get a dot graph
    /// 
    /// # Arguments
    /// 
    /// * `ids` - ID
    /// * `functions` - Function table
    /// 
    /// # Returns
    /// 
    /// * `String` - Dot graph
    /// 
    /// # Panics
    /// 
    /// * `Function not found` - If the function is not found
    pub fn get_dot_graph(&self, ids: Vec<usize>, functions: &FunctionTable) -> String {
        let mut text = String::new();
        let mut function_ids = vec![];
        let mut function_ids_set = HashSet::new();
        let mut variable_ids_set = HashSet::new();

        for id in &ids {
            self.add_variable_for_dot_graph(&mut variable_ids_set, &mut text, *id);
            self.add_function_for_dot_graph(&mut function_ids, &mut function_ids_set, *id);
        }

        while !function_ids.is_empty() {
            let f_id = function_ids.pop().unwrap();
            let f = functions.get(f_id).expect("Function not found");
            text.push_str(&f.to_dot_string());
            let input_ids = f.get_inputs();
            if let Some(input_ids) = input_ids {
                for id in input_ids {
                    self.add_variable_for_dot_graph(&mut variable_ids_set, &mut text, *id);
                    self.add_function_for_dot_graph(&mut function_ids, &mut function_ids_set, *id);
                }
            }
        }
        format!("digraph g {{\n{}}}", text)
    }

    /// Plot a dot graph
    /// 
    /// # Arguments
    /// 
    /// * `ids` - ID
    /// * `functions` - Function table
    /// * `path` - path without extension
    /// * `to_png` - If true, convert to png
    /// 
    /// # Panics
    /// 
    /// * `Failed to create file` - If failed to create file
    /// * `Failed to write file` - If failed to write file
    /// * `Failed to execute dot` - If failed to execute dot command
    pub fn plot_dot_graph(&self, ids: Vec<usize>, functions: &FunctionTable, path: &str, to_png: bool) {
        let dot_path = format!("{}.dot", path);
        {
            let text = self.get_dot_graph(ids, functions);
            let mut file = File::create(&dot_path).expect("Failed to create file");
            file.write_all(text.as_bytes()).expect("Failed to write file");
        }
        if to_png {
            Command::new("dot")
                .args([dot_path.as_str(), "-T", "png", "-o", format!("{}.png", path).as_str()])
                .status()
                .expect("Failed to execute dot");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_variable_from_f64_tensor_normal() {
        let mut table = VariableTable::new();
        let tensor = Tensor::new_from_num_vec(vec![1., 2., 3.], vec![3]);
        let id = table.generate_variable_from_f64_tensor(tensor.clone(), "x");
        let variable = table.get(id).unwrap();
        let data = variable.to_f64_tensor().unwrap();
        assert_eq!(data, &tensor);
        assert_eq!(variable.shape(), &vec![3]);
        assert_eq!(variable.data_type(), "f64");
        assert_eq!(variable.get_name(), "x");
    }
}
