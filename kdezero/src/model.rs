pub mod model_variable;
pub mod model_operator;
mod make;
mod run;
mod getter;
mod setter;
mod check_state;
mod print;

pub use model_variable::ModelVariable;
pub use model_operator::ModelOperator;

use crate::node::Graph;
use crate::variable::Variables;
use crate::operator::Operators;

pub struct Model {
    graph: Graph,
    variables: Variables,
    operators: Operators,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    sorted_forward_nodes: Vec<usize>,
    sorted_backward_nodes: Vec<usize>,
    grad_model: Option<Box<Model>>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            variables: Variables::new(),
            operators: Operators::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            sorted_forward_nodes: Vec::new(),
            sorted_backward_nodes: Vec::new(),
            grad_model: None,
        }
    }
}
