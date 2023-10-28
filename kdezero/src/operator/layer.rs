use anyhow::Result;
use super::OperatorContents;
use crate::model::Model;

#[derive(Clone)]
pub struct Layer {
    model: Model,
}

impl Layer {
    pub fn new(model: Model) -> Self {
        Self {
            model,
        }
    }
}

impl OperatorContents for Layer {
    fn forward(
            &mut self, node_id: usize,
            model: &mut Model,
        ) -> anyhow::Result<Vec<usize>> {
        todo!()
    }

    fn backward(
            &mut self, node_id: usize,
            model: &mut Model,
        ) -> anyhow::Result<Vec<usize>> {
        todo!()
    }
}
