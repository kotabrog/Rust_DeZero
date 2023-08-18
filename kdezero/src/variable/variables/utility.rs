use anyhow::Result;
use super::Variables;

impl Variables {
    pub(crate) fn update_next_id(&mut self, id: usize) {
        self.next_id = self.next_id.max(id) + 1;
    }

    pub(crate) fn delete_variable(&mut self, id: usize) -> Result<()> {
        self.check_id_not_in_variables(id)?;
        self.variables.remove(&id);
        Ok(())
    }

    pub fn print_variables(&self) {
        println!("Variables:");
        println!("  next_id: {}", self.next_id);
        for i in self.variables.keys() {
            println!("  {:?}: {:?}", i, self.variables[i]);
        }
    }
}
