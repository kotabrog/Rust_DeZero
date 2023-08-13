pub mod square;

pub use square::Square;

use anyhow::Result;
use crate::variable::Variables;
use crate::node::Graph;
use crate::model::Model;

pub trait OperatorContents {
    fn forward(
        &self, node_id: usize,
        graph: &Graph,
        variables: &mut Variables,
    ) -> Result<Vec<usize>>;

    fn backward(
        &self, node_id: usize,
        graph: &Graph, variables: &mut Variables,
        grad_model: &mut Model,
    ) -> Result<Vec<usize>>;
}
