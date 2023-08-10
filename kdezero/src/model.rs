pub mod model_variable;
pub mod model_operator;

pub use model_variable::ModelVariable;
pub use model_operator::ModelOperator;

use std::collections::HashMap;
use anyhow::Result;
use crate::node::{NodeData, Graph};
use crate::variable::{Variable, Variables, VariableData};
use crate::operator::{Operators, OperatorContents};
use crate::error::KdezeroError;

pub struct Model {
    graph: Graph,
    variables: Variables,
    operators: Operators,
    sorted_forward_nodes: Vec<usize>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            variables: Variables::new(),
            operators: Operators::new(),
            sorted_forward_nodes: Vec::new(),
        }
    }

    pub fn get_graph(&self) -> &Graph {
        &self.graph
    }

    pub fn get_variables(&self) -> &Variables {
        &self.variables
    }

    pub fn get_operators(&self) -> &Operators {
        &self.operators
    }

    pub fn add_new_node(
        &mut self, id: usize, name: String,
        data: NodeData, inputs: Vec<usize>, outputs: Vec<usize>)
    -> Result<()> {
        self.graph.add_new_node(id, name, data, inputs, outputs)?;
        if !self.sorted_forward_nodes.is_empty() {
            self.sorted_forward_nodes = Vec::new();
        }
        Ok(())
    }

    pub fn add_new_variable(
        &mut self, id: usize, node: Option<usize>, data: VariableData)
    -> Result<()> {
        self.variables.add_new_variable(id, node, data)
    }

    pub fn add_new_operator(
        &mut self, id: usize, node: Option<usize>, params: Vec<usize>, operator: Box<dyn OperatorContents>
    ) -> Result<()> {
        self.operators.add_new_operator(id, node, params, operator)
    }

    pub fn get_variable_from_name(&self, name: &str) -> Result<&Variable> {
        let node = self.graph.get_node_from_name(name)?;
        let variable_id = node.get_data().get_variable_id()?;
        self.variables.get_variable(variable_id)
    }

    fn set_node_inputs(&mut self, node_id: usize, inputs: Vec<usize>) -> Result<()> {
        self.graph.set_node_inputs(node_id, inputs)
    }

    fn add_node_input(&mut self, node_id: usize, input: usize) -> Result<()> {
        self.graph.add_node_input(node_id, input)
    }

    fn add_node_output(&mut self, node_id: usize, output: usize) -> Result<()> {
        self.graph.add_node_output(node_id, output)
    }

    pub fn make_model(
        inputs: Vec<ModelVariable>, outputs: Vec<ModelVariable>,
        operators: Vec<ModelOperator>, inits: Vec<ModelVariable>
    ) -> Result<Self> {
        fn check_name_in_nodes(name: &String, nodes: &HashMap<String, usize>) -> Result<()> {
            if nodes.contains_key(name) {
                return Err(
                    KdezeroError::DuplicateError(
                        name.clone(),
                        "Model".to_string()
                    ).into()
                );
            }
            Ok(())
        }

        fn model_variable_to_model_element(
            model: &mut Model, model_variable: ModelVariable, node_id: usize, variable_id: usize
        ) {
            let name = model_variable.name.clone();
            model.add_new_node(
                node_id, name.clone(), NodeData::Variable(variable_id),
                vec![], vec![]
            ).unwrap();
            model.add_new_variable(
                variable_id, Some(node_id), model_variable.data
            ).unwrap();
        }

        fn model_variable_init(
            model: &mut Model, model_variable: ModelVariable, id_counter: &mut usize,
            variable_nodes: &mut HashMap<String, usize>
        ) -> Result<()> {
            let name = model_variable.name.clone();
            check_name_in_nodes(&name, variable_nodes)?;
            model_variable_to_model_element(
                model, model_variable, *id_counter, *id_counter
            );
            variable_nodes.insert(name, *id_counter);
            *id_counter += 1;
            Ok(())
        }

        fn model_variables_init(
            model: &mut Model, model_variables: Vec<ModelVariable>, id_counter: &mut usize,
            variable_nodes: &mut HashMap<String, usize>
        ) -> Result<()> {
            for model_variable in model_variables {
                model_variable_init(
                    model, model_variable, id_counter, variable_nodes
                )?;
            }
            Ok(())
        }

        fn param_init(
            model: &mut Model, param: VariableData, id_counter: &mut usize,
            variable_counter: &mut usize, param_ids: &mut Vec<usize>
        ) {
            let param_name = (*id_counter).to_string();
            model_variable_to_model_element(
                model, ModelVariable::new(param_name, param),
                *id_counter, *variable_counter
            );
            param_ids.push(*id_counter);
            *id_counter += 1;
            *variable_counter += 1;
        }

        fn model_output_case(
            model: &mut Model, output: String, node_id: usize,
            variable_nodes: &HashMap<String, usize>, output_range: (usize, usize),
            output_ids: &mut Vec<usize>
        ) -> Result<()> {
            let output_id = *variable_nodes.get(&output).unwrap();
            if output_id < output_range.0 || output_range.1 <= output_id {
                return Err(
                    KdezeroError::DuplicateError(
                        output.clone(),
                        "Model".to_string()
                    ).into()
                );
            }
            model.add_node_input(output_id, node_id)?;
            output_ids.push(output_id);
            Ok(())
        }

        fn output_init(
            model: &mut Model, output: String, id_counter: &mut usize,
            variable_counter: &mut usize, node_id: usize,
            variable_nodes: &mut HashMap<String, usize>, output_ids: &mut Vec<usize>,
            output_range: (usize, usize)
        ) -> Result<()> {
            if variable_nodes.contains_key(&output) {
                return model_output_case(
                    model, output, node_id, variable_nodes, output_range, output_ids)
            }
            model.add_new_node(
                *id_counter, output.clone(), NodeData::Variable(*variable_counter),
                vec![node_id], vec![]
            ).unwrap();
            model.add_new_variable(
                *variable_counter, Some(*id_counter), VariableData::None
            ).unwrap();
            variable_nodes.insert(output, *id_counter);
            output_ids.push(*id_counter);
            *id_counter += 1;
            *variable_counter += 1;
            Ok(())
        }

        fn operator_init(
            model: &mut Model, node_id: usize,
            param_ids: Vec<usize>, operator_counter: usize,
            operator_data: Box<dyn OperatorContents>,
            name: String, output_ids: Vec<usize>
        ) {
            model.add_new_operator(
                operator_counter, Some(node_id),
                param_ids, operator_data
            ).unwrap();
            model.add_new_node(
                node_id, name,
                NodeData::Operator(operator_counter),
                vec![], output_ids
            ).unwrap();
        }

        fn model_operator_init(
            model: &mut Model, id_counter: &mut usize,
            variable_nodes: &mut HashMap<String, usize>, model_operator: ModelOperator,
            operator_counter: usize, variable_counter: &mut usize,
            output_range: (usize, usize)
        ) -> Result<(usize, Vec<String>)> {
            let name = model_operator.name.clone();
            let node_id = *id_counter;
            *id_counter += 1;
            check_name_in_nodes(&name, variable_nodes)?;
            let mut param_ids: Vec<usize> = vec![];
            for param in model_operator.params {
                param_init(
                    model, param, id_counter, variable_counter, &mut param_ids)
            }
            let mut output_ids: Vec<usize> = vec![];
            for output in model_operator.outputs {
                output_init(
                    model, output, id_counter, variable_counter,
                    node_id, variable_nodes, &mut output_ids,
                    output_range
                )?;
            }
            operator_init(
                model, node_id, param_ids, operator_counter,
                model_operator.data, name, output_ids
            );
            Ok((node_id, model_operator.inputs))
        }

        fn operator_input_init(
            model: &mut Model, node_id: usize, inputs: Vec<String>,
            variable_nodes: &HashMap<String, usize>
        ) -> Result<()> {
            let mut input_ids: Vec<usize> = vec![];
            for input in inputs {
                if !variable_nodes.contains_key(&input) {
                    return Err(
                        KdezeroError::NotFoundError(
                            input.clone(),
                            "nodes".to_string()
                        ).into()
                    );
                }
                input_ids.push(*variable_nodes.get(&input).unwrap());
            }
            for input_id in input_ids.iter() {
                model.add_node_output(*input_id, node_id)?;
            }
            model.set_node_inputs(node_id, input_ids)
        }

        fn model_operators_init(
            model: &mut Model, model_operators: Vec<ModelOperator>, id_counter: &mut usize,
            variable_nodes: &mut HashMap<String, usize>, output_range: (usize, usize)
        ) -> Result<()> {
            let mut remaining_operators = Vec::new();
            let mut operator_counter = 0;
            let mut variable_counter = *id_counter;
            for model_operator in model_operators {
                let remaining_operator = model_operator_init(
                    model, id_counter, variable_nodes, model_operator,
                    operator_counter, &mut variable_counter,
                    output_range
                )?;
                operator_counter += 1;
                remaining_operators.push(remaining_operator);
            }
            for (node_id, inputs) in remaining_operators {
                operator_input_init(model, node_id, inputs, variable_nodes)?;
            }
            Ok(())
        }

        let mut model = Model::new();
        let mut id_counter = 0;
        let mut variable_nodes: HashMap<String, usize> = HashMap::new();
        model_variables_init(&mut model, inputs, &mut id_counter, &mut variable_nodes)?;
        let output_start_id = id_counter;
        model_variables_init(&mut model, outputs, &mut id_counter, &mut variable_nodes)?;
        let output_end_id = id_counter;
        model_variables_init(&mut model, inits, &mut id_counter, &mut variable_nodes)?;
        model_operators_init(
            &mut model, operators, &mut id_counter,
            &mut variable_nodes, (output_start_id, output_end_id)
        )?;
        Ok(model)
    }

    pub fn forward(&mut self) -> Result<()> {
        if self.sorted_forward_nodes.is_empty() {
            self.sorted_forward_nodes = self.graph.topological_sort()?;
        }
        for id in self.sorted_forward_nodes.iter() {
            let node = self.graph.get_node(*id)?;
            match node.get_data() {
                NodeData::Operator(operator_id) => {
                    let operator = self.operators.get_operator(*operator_id)?;
                    operator.forward(&self.graph, &mut self.variables)?;
                },
                _ => (),
            }
        }
        Ok(())
    }
}
