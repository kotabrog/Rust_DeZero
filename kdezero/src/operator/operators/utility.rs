use super::Operators;

impl Operators {
    pub(crate) fn update_next_id(&mut self, id: usize) {
        self.next_id = self.next_id.max(id) + 1;
    }
}
