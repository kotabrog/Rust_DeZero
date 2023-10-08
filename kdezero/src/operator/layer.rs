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
