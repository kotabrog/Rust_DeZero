use std::collections::{HashMap, VecDeque};
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
#[derive(Debug, Clone)]
pub struct VariableWrapper {
    id: usize,
    variable: VariableType,
    creator: Option<usize>,
}

impl VariableWrapper {
    /// Create a new VariableWrapper from Variable<f64>
    /// 
    /// # Arguments
    /// 
    /// * `variable` - Variable<f64>
    pub fn from_variable_f64(variable: Variable<f64>) -> Self {
        Self { id: usize::MAX, variable: VariableType::F64(Box::new(variable)), creator: None }
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

    /// Set the creator
    pub fn set_creator(&mut self, creator: usize) {
        self.creator = Some(creator);
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

    /// Backward of the specified variable
    /// 
    /// # Arguments
    /// 
    /// * `ids` - ID
    /// * `functions` - Function table
    pub fn backward(&mut self, ids: Vec<usize>, functions: &mut FunctionTable) {
        for id in &ids {
            let y =
                self.get_mut(*id).expect("VariableTable::backward: variable not found")
                    .get_variable_mut();
            match y {
                VariableType::F64(y) => {
                    if y.grad().is_none() {
                        y.set_grad(Tensor::ones_like(&y.data()));
                    }
                }
            }
        }

        let mut function_ids = VecDeque::new();
        function_ids.extend(ids);
        while !function_ids.is_empty() {
            let id = function_ids.pop_front().unwrap();
            let y = self.get(id).expect("VariableTable::backward: variable not found");
            let f_id = y.creator;
            let f_id = match f_id {
                Some(f_id) => f_id,
                None => continue,
            };
            let f = functions.get_mut(f_id).expect("VariableWrapper::backward: function not found");
            // wip
            f.backward(vec![y.get_id()], self);
            let id_list = f.get_input().expect("VariableWrapper::backward: input not found");
            function_ids.extend(id_list);
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
