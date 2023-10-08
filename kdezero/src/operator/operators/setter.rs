use anyhow::Result;
use super::{Operator, Operators};
use super::super::Contents;

impl Operators {
    pub fn add_operator_no_check(&mut self, operator: Operator) {
        self.operators.insert(operator.get_id(), operator);
    }

    pub fn add_new_operator(
        &mut self, id: usize, node: Option<usize>, operator: Contents
    ) -> Result<()> {
        self.check_id_in_operators(id)?;
        let operator = Operator::new(id, node, operator);
        self.operators.insert(id, operator);
        self.update_next_id(id);
        Ok(())
    }

    pub(crate) fn set_node(&mut self, id: usize, node: Option<usize>) -> Result<()> {
        let operator = self.get_operator_mut(id)?;
        operator.set_node(node);
        Ok(())
    }

    pub(crate) fn set_next_id(&mut self, id: usize) {
        self.next_id = id;
    }

    pub(crate) fn change_operator_id(&mut self, old_id: usize, new_id: usize) -> Result<()> {
        self.check_id_not_in_operators(old_id)?;
        self.check_id_in_operators(new_id)?;
        let operator = self.operators.remove(&old_id).unwrap();
        self.operators.insert(new_id, operator);
        self.update_next_id(new_id);
        Ok(())
    }
}
