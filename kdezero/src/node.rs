pub mod graph;

#[derive(Debug, Clone)]
pub enum NodeData {
    None,
    Variable(usize),
    Operator(usize),
}

impl NodeData {
    pub fn to_string(&self) -> String {
        match self {
            NodeData::None => "None".to_string(),
            NodeData::Variable(_) => "Variable".to_string(),
            NodeData::Operator(_) => "Operator".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    id: usize,
    name: String,
    data: NodeData,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

impl Node {
    /// Create a new Node instance.
    /// 
    /// # Arguments
    /// 
    /// * `id` - Node ID
    /// * `name` - Node name
    pub fn new(id: usize, name: String, data: NodeData, inputs: Vec<usize>, outputs: Vec<usize>) -> Self {
        Self {
            id,
            name,
            data,
            inputs,
            outputs,
        }
    }

    /// Get the ID of the Node.
    pub fn get_id(&self) -> usize {
        self.id
    }

    /// Get the name of the Node.
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Get the data of the Node.
    pub fn get_data(&self) -> &NodeData {
        &self.data
    }

    /// Get inputs of the Node.
    pub fn get_inputs(&self) -> &Vec<usize> {
        &self.inputs
    }

    /// Get outputs of the Node.
    pub fn get_outputs(&self) -> &Vec<usize> {
        &self.outputs
    }
}
