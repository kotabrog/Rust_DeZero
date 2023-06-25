use std::collections::{HashSet, BinaryHeap};

#[derive(Debug, Eq)]
struct FunctionGeneration {
    id: usize,
    generation: usize,
}

impl FunctionGeneration {
    fn new(id: usize, generation: usize) -> Self {
        Self { id, generation }
    }

    fn get_id(&self) -> usize {
        self.id
    }
}

impl PartialEq for FunctionGeneration {
    fn eq(&self, other: &Self) -> bool {
        self.generation == other.generation
    }
}

impl PartialOrd for FunctionGeneration {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FunctionGeneration {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.generation.cmp(&other.generation)
    }
}

pub struct FunctionGenerationPriorityQueue {
    queue: BinaryHeap<FunctionGeneration>,
    id_set: HashSet<usize>,
}

impl FunctionGenerationPriorityQueue {
    pub fn new() -> Self {
        Self { queue: BinaryHeap::new(), id_set: HashSet::new() }
    }

    pub fn push(&mut self, id: usize, generation: usize) {
        if self.id_set.contains(&id) {
            return;
        }
        self.id_set.insert(id);
        self.queue.push(FunctionGeneration::new(id, generation));
    }

    pub fn pop(&mut self) -> Option<usize> {
        match self.queue.pop() {
            Some(x) => Some(x.get_id()),
            None => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}
