use super::Graph;

impl Graph {
    pub(crate) fn update_next_id(&mut self, id: usize) {
        self.next_id = self.next_id.max(id) + 1;
    }

    pub fn print_graph(&self) {
        println!("Graph:");
        for i in self.nodes.keys() {
            println!("  {}: {:?}", i, self.nodes[i]);
        }
    }
}
