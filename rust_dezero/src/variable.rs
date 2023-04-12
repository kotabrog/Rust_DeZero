use std::collections::{HashMap, HashSet, BinaryHeap};
use crate::Tensor;
use crate::function::FunctionTable;

/// Variable type
#[derive(Debug, Clone)]
pub enum VariableType {
    F64(Box<Variable<f64>>),
}

/// Variable wrapper
/// 
/// # Fields
/// 
/// * `id` - ID
/// * `variable` - Variable
/// * `creator` - Creator
/// * `generation` - Generation
#[derive(Debug, Clone)]
pub struct VariableWrapper {
    id: usize,
    variable: VariableType,
    creator: Option<usize>,
    generation: usize,
}

impl VariableWrapper {
    /// Create a new VariableWrapper from Variable<f64>
    /// 
    /// # Arguments
    /// 
    /// * `variable` - Variable<f64>
    pub fn from_variable_f64(variable: Variable<f64>) -> Self {
        Self { id: usize::MAX, variable: VariableType::F64(Box::new(variable)), creator: None, generation: 0 }
    }

    /// Get the ID
    pub fn get_id(&self) -> usize {
        self.id
    }

    /// Set the ID
    pub fn set_id(&mut self, id: usize) {
        self.id = id;
    }

    /// Get the variable
    pub fn get_variable(&self) -> &VariableType {
        &self.variable
    }

    /// Get the variable mutably
    pub fn get_variable_mut(&mut self) -> &mut VariableType {
        &mut self.variable
    }

    /// Get the creator
    pub fn get_creator(&self) -> Option<usize> {
        self.creator
    }

    /// Set the creator and generation
    pub fn set_creator(&mut self, creator: usize, generation: usize) {
        self.creator = Some(creator);
        self.generation = generation;
    }

    /// Clear the grad
    pub fn clear_grad(&mut self) {
        match &mut self.variable {
            VariableType::F64(x) => x.clear_grad(),
        }
    }

    pub fn get_generation(&self) -> usize {
        self.generation
    }
}

/// Variable
/// 
/// # Fields
/// 
/// * `data` - Contents of Variable
/// * `grad` - Gradient of Variable
#[derive(Debug, Clone)]
pub struct Variable<T>
{
    data: Tensor<T>,
    grad: Option<Tensor<T>>,
}

impl<T> Variable<T>
{
    /// Create a new Variable
    /// 
    /// # Arguments
    /// 
    /// * `data` - Contents of Variable
    pub fn new(data: Tensor<T>) -> Self {
        Self { data, grad: None }
    }

    /// Get the data
    pub fn data(&self) -> &Tensor<T> {
        &self.data
    }

    /// Get the shape
    pub fn shape(&self) -> &Vec<usize> {
        self.data.shape()
    }

    /// Get the data type
    pub fn data_type(&self) -> &str {
        self.data.data_type()
    }

    /// Set the data
    /// 
    /// # Arguments
    /// 
    /// * `data` - Contents of Variable
    pub fn set_data(&mut self, data: Tensor<T>) {
        self.data = data;
    }

    /// Get the grad
    pub fn grad(&self) -> Option<&Tensor<T>> {
        self.grad.as_ref()
    }

    /// Get the grad shape
    pub fn grad_shape(&self) -> Option<&Vec<usize>> {
        match &self.grad {
            Some(grad) => Some(grad.shape()),
            None => None,
        }
    }

    /// Set the grad
    /// 
    /// # Arguments
    /// 
    /// * `grad` - Gradient of Variable
    pub fn set_grad(&mut self, grad: Tensor<T>) {
        self.grad = Some(grad);
    }

    /// Clear the grad
    pub fn clear_grad(&mut self) {
        self.grad = None;
    }
}

impl<T> Variable<T>
where
    T: std::ops::AddAssign + Copy
{
    pub fn update_grad(&mut self, grad: Tensor<T>) {
        match &mut self.grad {
            Some(g) => {
                *g += &grad;
            },
            None => {
                self.grad = Some(grad);
            },
        }
    }
}

/// Structure for comparison in function generation
/// 
/// # Fields
/// 
/// * `id` - ID
/// * `generation` - Generation
#[derive(Debug, Eq)]
struct FunctionGeneration {
    id: usize,
    generation: usize,
}

impl FunctionGeneration {
    fn new(id: usize, generation: usize) -> Self {
        Self { id, generation }
    }

    fn get_id(&self) -> usize {
        self.id
    }
}

impl PartialEq for FunctionGeneration {
    fn eq(&self, other: &Self) -> bool {
        self.generation == other.generation
    }
}

impl PartialOrd for FunctionGeneration {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FunctionGeneration {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.generation.cmp(&other.generation)
    }
}

struct FunctionGenerationPriorityQueue {
    queue: BinaryHeap<FunctionGeneration>,
    id_set: HashSet<usize>,
}

impl FunctionGenerationPriorityQueue {
    fn new() -> Self {
        Self { queue: BinaryHeap::new(), id_set: HashSet::new() }
    }

    fn push(&mut self, id: usize, generation: usize) {
        if self.id_set.contains(&id) {
            return;
        }
        self.id_set.insert(id);
        self.queue.push(FunctionGeneration::new(id, generation));
    }

    fn pop(&mut self) -> Option<usize> {
        match self.queue.pop() {
            Some(x) => Some(x.get_id()),
            None => None,
        }
    }

    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

/// Variable table
/// 
/// # Fields
/// 
/// * `table` - Variable table
/// * `id_max` - Maximum ID
pub struct VariableTable {
    table: HashMap<usize, Box<VariableWrapper>>,
    id_max: usize,
}

impl VariableTable {
    /// Create a new VariableTable
    pub fn new() -> Self {
        Self { table: HashMap::new(), id_max: 0 }
    }

    /// Add a new variable
    /// 
    /// # Arguments
    /// 
    /// * `var` - Variable wrapper
    pub fn add(&mut self, var: Box<VariableWrapper>) -> usize {
        let id = self.id_max;
        self.table.insert(id, var);
        self.table.get_mut(&id).unwrap().set_id(id);
        self.id_max = self.id_max.checked_add(1).expect("VariableTable::add: id_max overflow");
        id
    }

    /// Get a variable
    /// 
    /// # Arguments
    /// 
    /// * `id` - ID
    pub fn get(&self, id: usize) -> Option<&Box<VariableWrapper>> {
        self.table.get(&id)
    }

    /// Get a variable mutably
    /// 
    /// # Arguments
    /// 
    /// * `id` - ID
    pub fn get_mut(&mut self, id: usize) -> Option<&mut Box<VariableWrapper>> {
        self.table.get_mut(&id)
    }

    /// Get a variable type
    /// 
    /// # Arguments
    /// 
    /// * `id` - ID
    pub fn get_variable_type(&self, id: usize) -> Option<&VariableType> {
        match self.table.get(&id) {
            Some(var) => Some(var.get_variable()),
            None => None,
        }
    }

    /// Get a variable type mutably
    /// 
    /// # Arguments
    /// 
    /// * `id` - ID
    pub fn get_variable_type_mut(&mut self, id: usize) -> Option<&mut VariableType> {
        match self.table.get_mut(&id) {
            Some(var) => Some(var.get_variable_mut()),
            None => None,
        }
    }

    /// If grad of the specified variable is not set, set to 1.
    fn set_grad_default(&mut self, id: usize) {
        let y = self.get_mut(id).expect("VariableTable::set_grad_default: variable not found");
        match y.get_variable_mut() {
            VariableType::F64(y) => {
                if y.grad().is_none() {
                    y.set_grad(Tensor::ones_like(&y.data()));
                }
            }
        }
    }

    /// Add a function to the function queue
    fn add_function(&self, functions: &mut FunctionTable,
        function_ids: &mut FunctionGenerationPriorityQueue, id: usize) {
        let y = self.get(id).expect("VariableTable::backward: variable not found");
        let f_id = y.get_creator();
        let f_id = match f_id {
            Some(f_id) => f_id,
            None => return,
        };
        let f_generation = functions
            .get(f_id).expect("VariableTable::backward: function not found")
            .get_generation();
        function_ids.push(f_id, f_generation);
    }

    /// Backward of the specified variable
    /// 
    /// # Arguments
    /// 
    /// * `ids` - ID
    /// * `functions` - Function table
    pub fn backward(&mut self, ids: Vec<usize>, functions: &mut FunctionTable) {
        for id in &ids {
            self.set_grad_default(*id);
        }

        let mut function_ids = FunctionGenerationPriorityQueue::new();
        for id in &ids {
            self.add_function(functions, &mut function_ids, *id);
        }

        while !function_ids.is_empty() {
            let f_id = function_ids.pop().unwrap();
            let f = functions.get_mut(f_id).expect("VariableWrapper::backward: function not found");
            let input_ids = f.backward(self);
            for id in &input_ids {
                self.add_function(functions, &mut function_ids, *id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_normal() {
        let tensor = Tensor::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let x = Variable::<f32>::new(tensor.clone());
        assert_eq!(*x.data(), tensor);
        assert_eq!(*x.shape(), vec![3]);
        assert_eq!(x.data_type(), "f32");
    }

    #[test]
    fn set_data_normal() {
        let tensor = Tensor::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let mut x = Variable::<f32>::new(tensor.clone());
        let tensor = Tensor::new_from_num_vec(vec![4.0, 5.0, 6.0], vec![3]);
        x.set_data(tensor.clone());
        assert_eq!(*x.data(), tensor);
        assert_eq!(*x.shape(), vec![3]);
        assert_eq!(x.data_type(), "f32");
    }

    #[test]
    fn set_grad_normal() {
        let tensor = Tensor::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let mut x = Variable::<f32>::new(tensor);
        let tensor = Tensor::new_from_num_vec(vec![4.0, 5.0, 6.0], vec![3]);
        x.set_grad(tensor.clone());
        assert_eq!(*x.grad().unwrap(), tensor);
        assert_eq!(*x.grad_shape().unwrap(), vec![3]);
    }
}
