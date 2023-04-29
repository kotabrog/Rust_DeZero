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

    pub fn generate_variable_from_variable_contents(&mut self, data: VariableContents) -> usize {
        self.insert(Variable::new(data, self.id_max))
    }

    pub fn generate_variable_from_f64_tensor(&mut self, tensor: Tensor<f64>) -> usize {
        self.generate_variable_from_variable_contents(VariableContents::F64(Box::new(tensor)))
    }

    pub fn get_variable(&self, id: usize) -> Option<&Variable> {
        self.table.get(&id).map(|v| v.as_ref())
    }

    pub fn get_mut_variable(&mut self, id: usize) -> Option<&mut Variable> {
        self.table.get_mut(&id).map(|v| v.as_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_variable_from_f64_tensor_normal() {
        let mut table = VariableTable::new();
        let id = table.generate_variable_from_f64_tensor(Tensor::new_from_num_vec(vec![1., 2., 3.], vec![3]));
        let variable = table.get_variable(id).unwrap();
        assert_eq!(variable.shape(), &vec![3]);
        assert_eq!(variable.data_type(), "f64");
    }
}
