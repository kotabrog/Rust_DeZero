extern crate ktensor;

#[test]
fn sample() {
    use ktensor::Tensor;

    let x = Tensor::<f32>::arrange([2, 3]).unwrap();
    let y = Tensor::<f32>::arrange([3, 2]).unwrap();
    let z = x.matmul(&y).unwrap();

    let z = z + 1.0;

    let mut x = Tensor::<f32>::arrange([2, 2]).unwrap();
    x += &z;

    let sum = x.sum(Some([0, 1]), false);
    assert_eq!(sum.at([]).unwrap(), &101.0);
}

// #[test]
// fn sample2() {
//     use std::collections::HashMap;

//     struct Model {
//         nodes: HashMap<usize, Node>,
//         variables: HashMap<usize, usize>,
//     }

//     struct Node {
//         id: usize
//     }

//     impl Node {
//         pub fn new(id: usize) -> Self {
//             Self {
//                 id
//             }
//         }

//         pub fn variable(&mut self, variables: &mut HashMap<usize, usize>) -> () {
//             self.id = variables[&self.id];
//         }
//     }

//     impl Model
//     {
//         pub fn new() -> Self {
//             Self {
//                 nodes: HashMap::new(),
//                 variables: HashMap::new(),
//             }
//         }

//         pub fn test(&mut self) {
//             self.nodes.insert(0, Node::new(0));
//             self.variables.insert(0, 1);
//             let mut node = self.nodes.get_mut(&0).unwrap();
//             node.variable(&mut self.variables);
//         }
//     }
// }
