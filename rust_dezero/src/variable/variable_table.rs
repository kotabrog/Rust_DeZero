use std::collections::HashMap;
use super::{Variable, VariableContents};
use crate::Tensor;

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

    pub fn get_variable(&self, id: usize) -> Option<&Variable> {
        self.table.get(&id).map(|v| v.as_ref())
    }

    pub fn get_mut_variable(&mut self, id: usize) -> Option<&mut Variable> {
        self.table.get_mut(&id).map(|v| v.as_mut())
    }

    pub fn get_variable_contents_f64(&self, id: usize) -> Option<&Tensor<f64>> {
        self.table.get(&id).map(|v| v.to_f64_tensor()).flatten()
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
        let variable = table.get_variable(id).unwrap();
        let data = variable.to_f64_tensor().unwrap();
        assert_eq!(data, &tensor);
        assert_eq!(variable.shape(), &vec![3]);
        assert_eq!(variable.data_type(), "f64");
        assert_eq!(variable.get_name(), "x");
    }
}
