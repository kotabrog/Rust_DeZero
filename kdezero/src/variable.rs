pub mod variable_table;

pub use variable_table::VariableTable;

use ktensor::tensor::Tensor;

/// Wrapper of Tensor
#[derive(Debug, Clone)]
pub enum VariableContents {
    F64(Box<Tensor<f64>>),
}

impl VariableContents {
    /// Returns the data type of the Tensor.
    pub fn data_type(&self) -> &str {
        match self {
            VariableContents::F64(_) => "f64",
        }
    }

    /// Returns the shape of the Tensor.
    pub fn shape(&self) -> &Vec<usize> {
        match self {
            VariableContents::F64(data) => data.shape(),
        }
    }

    /// Returns the size of the Tensor.
    pub fn size(&self) -> usize {
        match self {
            VariableContents::F64(data) => data.size(),
        }
    }

    /// Returns a reference to the Tensor<f64> if this is a VariableContents::F64 variant, otherwise None.
    pub fn to_f64_tensor(&self) -> Option<&Tensor<f64>> {
        match self {
            VariableContents::F64(data) => Some(data.as_ref()),
        }
    }
}

/// Variable
/// 
/// # Fields
/// 
/// * `data` - VariableContents
/// * `id` - Variable ID
/// * `name` - Variable name
/// * `creator` - Function ID that created this variable
/// * `generation` - Generation of this variable
/// * `grad_id` - Variable ID of the gradient of this variable
#[derive(Debug, Clone)]
pub struct Variable {
    data: VariableContents,
    id: usize,
    name: String,
    creator: Option<usize>,
    generation: usize,
    grad_id: Option<usize>,
}

impl Variable {
    /// Create a new Variable instance.
    /// 
    /// # Arguments
    /// 
    /// * `data` - VariableContents
    /// * `id` - Variable ID
    /// * `name` - Variable name
    pub fn new(data: VariableContents, id: usize, name: &str) -> Self {
        Self { data, id, name: name.to_string(), creator: None, generation: 0, grad_id: None }
    }

    /// Get the data of the Variable.
    pub fn get_data(&self) -> &VariableContents {
        &self.data
    }

    /// Get the mutable data of the Variable.
    pub fn get_mut_data(&mut self) -> &mut VariableContents {
        &mut self.data
    }

    /// Get the ID of the Variable.
    pub fn get_id(&self) -> usize {
        self.id
    }

    /// Get the name of the Variable.
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Set the name of the Variable.
    pub fn set_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    /// Get the creator of the Variable.
    pub fn get_creator(&self) -> Option<usize> {
        self.creator
    }

    /// Sets the creator of the Variable and its generation.
    /// 
    /// # Arguments
    /// 
    /// * `creator` - Function ID that created this variable
    /// * `generation` - Generation of this variable
    pub fn set_creator(&mut self, creator: usize, generation: usize) {
        self.creator = Some(creator);
        self.generation = generation;
    }

    /// Get the generation of the Variable.
    pub fn get_generation(&self) -> usize {
        self.generation
    }

    /// Get the ID of the gradient of the Variable.
    pub fn get_grad_id(&self) -> Option<usize> {
        self.grad_id
    }

    /// Sets the ID of the gradient of the Variable.
    pub fn set_grad_id(&mut self, grad_id: usize) {
        self.grad_id = Some(grad_id);
    }

    /// Get the shape of the Variable.
    pub fn shape(&self) -> &Vec<usize> {
        self.data.shape()
    }

    /// Get the size of the Variable.
    pub fn size(&self) -> usize {
        self.data.size()
    }

    /// Get the data type of the Variable.
    pub fn data_type(&self) -> &str {
        self.data.data_type()
    }

    /// Returns a reference to the Tensor<f64> if this is a VariableContents::F64 variant, otherwise None.
    pub fn to_f64_tensor(&self) -> Option<&Tensor<f64>> {
        self.data.to_f64_tensor()
    }

    /// Clear the gradient of the Variable.
    pub fn clear_grad(&mut self) {
        self.grad_id = None;
    }

    /// Get dot string
    pub fn to_dot_string(&self) -> String {
        format!("var_{} [label=\"{}\", color=orange, style=filled]\n", self.id, self.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_normal() {
        let tensor = Tensor::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let x = Variable::new(VariableContents::F64(Box::new(tensor.clone())), usize::MAX, "x");
        let data = x.to_f64_tensor().unwrap();

        assert_eq!(data, &tensor);
        assert_eq!(*x.shape(), vec![3]);
        assert_eq!(x.data_type(), "f64");
        assert_eq!(x.get_name(), "x");
    }
}
