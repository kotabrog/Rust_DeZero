use super::operator_contents::{OperatorContents, OperatorContentsWrapper};
use super::layer::Layer;

#[derive(Clone)]
pub enum Contents {
    Operator(OperatorContentsWrapper),
    Layer(Layer),
    None,
}

impl Contents {
    pub fn make_operator(operator: Box<dyn OperatorContents>) -> Self {
        Self::Operator(operator.into())
    }

    pub fn take(&mut self) -> Self {
        let content = std::mem::replace(self, Self::None);
        content
    }
}

impl From<OperatorContentsWrapper> for Contents {
    fn from(operator: OperatorContentsWrapper) -> Self {
        Self::Operator(operator)
    }
}

impl From<Layer> for Contents {
    fn from(layer: Layer) -> Self {
        Self::Layer(layer)
    }
}
