use anyhow::Result;
use super::{Variable, Variables, VariableData};

impl Variables {
    pub(crate) fn add_variable_no_check(&mut self, variable: Variable) {
        self.variables.insert(variable.get_id(), variable);
    }

    pub fn add_new_variable(
        &mut self, id: usize, node: Option<usize>, data: VariableData)
    -> Result<()> {
        self.check_id_in_variables(id)?;
        let variable = Variable::new(id, node, data);
        self.variables.insert(id, variable);
        self.update_next_id(id);
        Ok(())
    }

    pub(crate) fn set_node_id(&mut self, id: usize, node: Option<usize>) -> Result<()> {
        let variable = self.get_variable_mut(id)?;
        variable.set_node(node);
        Ok(())
    }

    pub(crate) fn set_grad(&mut self, id: usize, grad: Option<usize>) -> Result<()> {
        let variable = self.get_variable_mut(id)?;
        variable.set_grad(grad);
        Ok(())
    }

    pub(crate) fn set_next_id(&mut self, id: usize) {
        self.next_id = id;
    }

    pub(crate) fn change_variable_id(&mut self, id: usize, new_id: usize) -> Result<()> {
        self.check_id_not_in_variables(id)?;
        self.check_id_in_variables(new_id)?;
        let mut variable = self.variables.remove(&id).unwrap();
        variable.set_id(new_id);
        self.variables.insert(new_id, variable);
        Ok(())
    }

    pub fn clear_grads(&mut self) {
        for (_, variable) in self.variables.iter_mut() {
            variable.clear_grad();
        }
    }
}
