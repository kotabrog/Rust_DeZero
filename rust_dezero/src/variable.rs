pub mod variable_table;

pub use variable_table::VariableTable;

use crate::tensor::Tensor;


#[derive(Debug, Clone)]
pub enum VariableContents {
    F64(Box<Tensor<f64>>),
}

impl VariableContents {
    pub fn data_type(&self) -> &str {
        match self {
            VariableContents::F64(_) => "f64",
        }
    }

    pub fn shape(&self) -> &Vec<usize> {
        match self {
            VariableContents::F64(data) => data.shape(),
        }
    }

    pub fn to_f64_tensor(&self) -> Option<&Tensor<f64>> {
        match self {
            VariableContents::F64(data) => Some(data.as_ref()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Variable {
    data: VariableContents,
    id: usize,
    name: String,
}

impl Variable {
    pub fn new(data: VariableContents, id: usize, name: &str) -> Self {
        Self { data, id, name: name.to_string() }
    }

    pub fn get_data(&self) -> &VariableContents {
        &self.data
    }

    pub fn get_mut_data(&mut self) -> &mut VariableContents {
        &mut self.data
    }

    pub fn get_id(&self) -> usize {
        self.id
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn shape(&self) -> &Vec<usize> {
        self.data.shape()
    }

    pub fn data_type(&self) -> &str {
        self.data.data_type()
    }

    pub fn to_f64_tensor(&self) -> Option<&Tensor<f64>> {
        self.data.to_f64_tensor()
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
