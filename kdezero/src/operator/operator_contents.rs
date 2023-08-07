pub mod square;

use anyhow::Result;
use crate::variable::Variables;
use crate::node::Graph;

pub trait OperatorContents {
    fn forward(
        &self, node_id: usize,
        graph: &Graph,
        variables: &mut Variables,
    ) -> Result<Vec<usize>>;
}
