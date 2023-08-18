use super::Model;

impl Model {
    pub fn print_model(&self) {
        self.graph.print_graph();
        self.variables.print_variables();
        // println!("operators: {:?}", self.operators);
        println!("inputs: {:?}", self.inputs);
        println!("outputs: {:?}", self.outputs);
        println!("sorted_forward_nodes: {:?}", self.sorted_forward_nodes);
        println!("sorted_backward_nodes: {:?}", self.sorted_backward_nodes);
        if let Some(grad_model) = &self.grad_model {
            println!("grad_model: ");
            grad_model.print_model();
        }
    }
}
