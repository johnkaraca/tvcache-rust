use serde::Serialize;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::mem::{size_of, size_of_val};

#[derive(Clone, Debug, Default, Serialize)]
pub struct ToolCallEntry {
    pub value: Option<String>,
    pub runtime: u32, // milliseconds
    pub embedding: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Node {
    pub children: HashMap<String, Node>,
    pub sandbox_id: Option<String>,
    pub tool_calls: HashMap<String, ToolCallEntry>,
    pub settled: bool, // if true, this node has been compared with every other settled sibling during contraction
    pub ttl_present: bool,
    pub ttl: u32, // seconds
    pub inserted_at_seconds: u64,
    pub number_of_cache_hits: u64,
    pub number_of_lpm_hits: u64,
    pub size_in_bytes: u32, // A single entry can't be more than 4GB, need to update this if we remove sandbox id later.
}

impl Node {
    pub fn child(&self, key: &str) -> Option<&Node> {
        self.children.get(key)
    }

    pub fn child_mut(&mut self, key: &str) -> Option<&mut Node> {
        self.children.get_mut(key)
    }

    /// Returns the child and whether it was newly created.
    pub fn get_or_create_child(&mut self, key: String) -> (&mut Node, bool) {
        match self.children.entry(key) {
            Entry::Occupied(entry) => (entry.into_mut(), false),
            Entry::Vacant(entry) => {
                let mut node = Node::default();
                node.tool_calls
                    .insert(entry.key().clone(), ToolCallEntry::default());
                (entry.insert(node), true)
            }
        }
    }

    pub fn remove_child(&mut self, key: &str) -> Option<Node> {
        self.children.remove(key)
    }

    pub fn retain_children<F>(&mut self, mut keep: F)
    where
        F: FnMut(&mut Node) -> bool,
    {
        self.children.retain(|_, child| keep(child));
    }

    /// Yields one representative tool-call key and node per child.
    pub fn iter_children(&self) -> impl Iterator<Item = (&String, &Node)> {
        self.children.iter()
    }

    pub fn walk_subtcg<T, F>(&self, initial: T, f: &mut F) -> T
    where
        F: FnMut(T, &Node) -> T,
    {
        let mut acc = f(initial, self);

        for (_, child) in self.iter_children() {
            acc = child.walk_subtcg(acc, f);
        }

        return acc;
    }

    pub fn update_size_in_bytes(&mut self) {
        let entries_size_in_bytes: u32 = self
            .tool_calls
            .iter()
            .map(|(key, entry)| {
                key.len() as u32
                    + entry.value.as_ref().map_or(0, |value| value.len() as u32)
                    + size_of_val(&entry.runtime) as u32
                    + entry
                        .embedding
                        .as_ref()
                        .map_or(0, |embedding| (embedding.len() * size_of::<f32>()) as u32)
            })
            .sum();

        let sandbox_id_size_in_bytes = self
            .sandbox_id
            .as_ref()
            .map_or(0, |sandbox_id| sandbox_id.len() as u32);

        self.size_in_bytes = entries_size_in_bytes
            + sandbox_id_size_in_bytes
            + size_of_val(&self.settled) as u32
            + size_of_val(&self.ttl_present) as u32
            + size_of_val(&self.ttl) as u32
            + size_of_val(&self.inserted_at_seconds) as u32
            + size_of_val(&self.number_of_cache_hits) as u32
            + size_of_val(&self.number_of_lpm_hits) as u32;
    }
}
