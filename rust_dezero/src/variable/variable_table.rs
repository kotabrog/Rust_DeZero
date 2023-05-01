mod function_generation_priority_queue;

use std::collections::HashMap;
use function_generation_priority_queue::FunctionGenerationPriorityQueue;
use super::{Variable, VariableContents};
use crate::{Tensor, function::{FunctionTable, operator::Add}};

pub struct VariableTable {
    table: HashMap<usize, Box<Variable>>,
    id_max: usize,
}

impl VariableTable {
    pub fn new() -> Self {
        Self { table: HashMap::new(), id_max: 0 }
    }

    fn insert(&mut self, variable: Variable) -> usize {
        let id = self.id_max;
        self.id_max += 1;
        self.table.insert(id, Box::new(variable));
        id
    }

    pub fn get(&self, id: usize) -> Option<&Variable> {
        self.table.get(&id).map(|v| v.as_ref())
    }

    pub fn get_mut(&mut self, id: usize) -> Option<&mut Variable> {
        self.table.get_mut(&id).map(|v| v.as_mut())
    }

    pub fn generate_variable_from_variable_contents(&mut self, data: VariableContents, name: &str) -> usize {
        self.insert(Variable::new(data, self.id_max, name))
    }

    pub fn generate_variable_from_f64_tensor(&mut self, tensor: Tensor<f64>, name: &str) -> usize {
        self.generate_variable_from_variable_contents(VariableContents::F64(Box::new(tensor)), name)
    }

    pub fn get_variable_contents_f64(&self, id: usize) -> Option<&Tensor<f64>> {
        self.table.get(&id).map(|v| v.to_f64_tensor()).flatten()
    }

    pub fn get_variable_grad_id(&self, id: usize) -> Option<usize> {
        self.table.get(&id).map(|v| v.get_grad_id()).flatten()
    }

    pub fn get_variable_grad_contents_f64(&self, id: usize) -> Option<&Tensor<f64>> {
        self.get_variable_grad_id(id)
            .map(|grad_id| self.get_variable_contents_f64(grad_id)).flatten()
    }

    pub fn set_grad(&mut self, variable_id: usize, grad_id: usize) {
        let variable = self.get_mut(variable_id).expect("Invalid variable id");
        variable.set_grad_id(grad_id);
    }

    pub fn set_grad_from_f64_tensor(&mut self, variable_id: usize, grad: Tensor<f64>) {
        let grad_id = self.generate_variable_from_f64_tensor(grad, "");
        self.set_grad(variable_id, grad_id);
    }

    pub fn set_grad_default(&mut self, variable_id: usize) {
        let variable = self.get(variable_id).expect("Invalid variable id");
        let grad = match variable.get_grad_id() {
            Some(_) => return,
            None => Tensor::ones_like(variable.to_f64_tensor().expect("Invalid variable data"))
        };
        self.set_grad_from_f64_tensor(variable_id, grad);
    }

    pub fn sets_grad_default(&mut self, variable_ids: &Vec<usize>) {
        for &variable_id in variable_ids {
            self.set_grad_default(variable_id);
        }
    }

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

    pub fn clear_grads(&mut self, ids: &Vec<usize>) {
        for &id in ids {
            self.get_mut(id).expect("Invalid variable id").clear_grad();
        }
    }

    fn add_function_to_queue(&mut self, variable_id: usize,
            priority_queue: &mut FunctionGenerationPriorityQueue, function_table: &FunctionTable) {
        let variable = self.get(variable_id).expect("Invalid variable id");
        let function_id = match variable.get_creator() {
            Some(id) => id,
            None => return,
        };
        let function_generation = function_table.get_function(function_id).expect("Invalid function id")
            .get_generation();
        priority_queue.push(function_id, function_generation);
    }

    fn add_functions_to_queue(&mut self, variable_ids: &Vec<usize>,
            priority_queue: &mut FunctionGenerationPriorityQueue, function_table: &FunctionTable) {
        for &variable_id in variable_ids {
            self.add_function_to_queue(variable_id, priority_queue, function_table);
        }
    }

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
