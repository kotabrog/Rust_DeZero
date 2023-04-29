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
}

#[derive(Debug, Clone)]
pub struct Variable {
    data: VariableContents,
    id: usize,
}

impl Variable {
    pub fn new(data: VariableContents, id: usize) -> Self {
        Self { data, id }
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

    pub fn shape(&self) -> &Vec<usize> {
        self.data.shape()
    }

    pub fn data_type(&self) -> &str {
        self.data.data_type()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_normal() {
        let tensor = Tensor::new_from_num_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let x = Variable::new(VariableContents::F64(Box::new(tensor.clone())), usize::MAX);
        let data: &Box<Tensor<f64>> = match x.get_data() {
            VariableContents::F64(data) => data,
        };

        assert_eq!(**data, tensor);
        assert_eq!(*x.shape(), vec![3]);
        assert_eq!(x.data_type(), "f64");
    }
}
