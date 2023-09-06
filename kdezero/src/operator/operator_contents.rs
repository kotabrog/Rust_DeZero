pub mod square;
pub mod add;
pub mod identity;
pub mod mul;
pub mod exp;
pub mod sin;
pub mod cos;
pub mod neg;
pub mod sub;
pub mod div;
pub mod pow;
pub mod scalar_mul;

pub use square::Square;
pub use add::Add;
pub use identity::Identity;
pub use mul::Mul;
pub use exp::Exp;
pub use sin::Sin;
pub use cos::Cos;
pub use neg::Neg;
pub use sub::Sub;
pub use div::Div;
pub use pow::Pow;
pub use scalar_mul::ScalarMul;

use anyhow::Result;
use crate::model::Model;

/// Trait to give clone method to operator contents.
pub trait CloneOperator {
    fn clone_operator(&self) -> Box<dyn OperatorContents>;
}

pub trait OperatorContents: CloneOperator {
    fn forward(
        &self, node_id: usize,
        model: &mut Model,
    ) -> Result<Vec<usize>>;

    fn backward(
        &self, node_id: usize,
        model: &mut Model,
    ) -> Result<Vec<usize>>;
}

impl<T> CloneOperator for T
where
    T: OperatorContents + Clone + 'static,
{
    fn clone_operator(&self) -> Box<dyn OperatorContents> {
        Box::new(self.clone())
    }
}

/// Wrapper struct to operator contents.
pub struct OperatorContentsWrapper {
    operator: Box<dyn OperatorContents>,
}

impl OperatorContentsWrapper {
    pub fn new(operator: Box<dyn OperatorContents>) -> Self {
        Self { operator }
    }

    pub fn get_operator(&self) -> &Box<dyn OperatorContents> {
        &self.operator
    }

    pub fn forward(
        &self, node_id: usize,
        model: &mut Model,
    ) -> Result<Vec<usize>> {
        self.operator.forward(node_id, model)
    }

    pub fn backward(
        &self, node_id: usize,
        model: &mut Model,
    ) -> Result<Vec<usize>> {
        self.operator.backward(node_id, model)
    }
}

impl Clone for OperatorContentsWrapper {
    fn clone(&self) -> Self {
        Self {
            operator: self.operator.clone_operator(),
        }
    }
}
