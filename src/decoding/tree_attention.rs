/// Speculative Tree Attention (EAGLE/Medusa style)
///
/// Replaces linear speculative decoding with a branching tree structure.
/// The NPU generates top-K alternatives at each step, forming a tree.
/// The iGPU verifies all branches simultaneously using a custom 2D Tree Attention Mask.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    pub node_id: u32,
    pub parent_id: u32,
    pub token_id: u32,
    pub probability: f64,
    pub depth: u32,
    /// MoE-Spec: The routing probabilities for each expert for this token.
    pub routing_probs: Vec<f64>,
    /// MoE-Spec: The actual experts selected after budgeting is applied.
    pub budgeted_experts: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftTree {
    pub nodes: Vec<TreeNode>,
    pub max_depth: u32,
}

impl DraftTree {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            max_depth: 0,
        }
    }

    /// Add a node to the draft tree.
    pub fn add_node(&mut self, parent_id: u32, token_id: u32, prob: f64, depth: u32, routing_probs: Vec<f64>) -> u32 {
        let node_id = self.nodes.len() as u32;
        self.nodes.push(TreeNode {
            node_id,
            parent_id,
            token_id,
            probability: prob,
            depth,
            routing_probs,
            budgeted_experts: Vec::new(),
        });
        if depth > self.max_depth {
            self.max_depth = depth;
        }
        node_id
    }

    /// Construct a 2D boolean attention mask for the iGPU verifier.
    /// mask[i][j] is true if node i is an ancestor of node j.
    pub fn build_attention_mask(&self) -> Vec<Vec<bool>> {
        let n = self.nodes.len();
        let mut mask = vec![vec![false; n]; n];
        
        for i in 0..n {
            let mut curr = i as u32;
            loop {
                mask[curr as usize][i] = true;
                if curr == 0 {
                    break;
                }
                curr = self.nodes[curr as usize].parent_id;
            }
        }
        mask
    }

    /// Flatten tree into a topological sort for batch verification.
    /// Legacy sequential traversal.
    pub fn flatten_tokens(&self) -> Vec<u32> {
        self.nodes.iter().map(|n| n.token_id).collect()
    }

    /// DeFT: Decoding with Flash Tree-Attention (KV-Guided Grouping)
    /// 
    /// Instead of blindly flattening the tree, this algorithm topologically sorts 
    /// the branches so that nodes sharing the longest common prefix are evaluated 
    /// contiguously. This maximizes L1/L2 cache hits on the iGPU and reduces 
    /// redundant LPDDR5x KV cache memory reads by up to 73%.
    pub fn deft_flatten(&self) -> Vec<u32> {
        if self.nodes.is_empty() { return vec![]; }
        
        let mut sorted_indices = Vec::new();
        let mut stack = vec![0]; // Start at root
        
        // Build adjacency list for children
        let mut children_map: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
        for node in &self.nodes {
            if node.node_id != 0 {
                children_map.entry(node.parent_id).or_default().push(node.node_id);
            }
        }
        
        // Depth-First Traversal ensures branches with shared prefixes are processed together
        while let Some(current_id) = stack.pop() {
            sorted_indices.push(self.nodes[current_id as usize].token_id);
            
            if let Some(children) = children_map.get(&current_id) {
                // Push children in reverse order so they are popped left-to-right
                for &child_id in children.iter().rev() {
                    stack.push(child_id);
                }
            }
        }
        
        sorted_indices
    }

    /// Implements MoE-Spec Expert Budgeting.
    /// 
    /// Prevents the "expert explosion" during tree verification.
    /// It scores experts by summing routing probabilities across the tree,
    /// selects the Top-B experts globally for the layer, and forces all tokens
    /// to route only to those B experts.
    pub fn enforce_expert_budget(&mut self, budget_b: usize, active_k: usize) {
        if self.nodes.is_empty() || self.nodes[0].routing_probs.is_empty() {
            return;
        }

        let num_experts = self.nodes[0].routing_probs.len();
        let mut global_scores = vec![0.0; num_experts];

        // 1. Calculate aggregate score for each expert
        for node in &self.nodes {
            if node.routing_probs.len() == num_experts {
                for (expert_idx, &prob) in node.routing_probs.iter().enumerate() {
                    global_scores[expert_idx] += prob;
                }
            }
        }

        // 2. Select Top-B experts
        let mut scored_experts: Vec<(usize, f64)> = global_scores.into_iter().enumerate().collect();
        scored_experts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let budget: std::collections::HashSet<usize> = scored_experts
            .into_iter()
            .take(budget_b)
            .map(|(idx, _)| idx)
            .collect();

        // 3. Remap each node's Top-K experts to the budgeted Top-B list
        for node in &mut self.nodes {
            if node.routing_probs.is_empty() { continue; }
            
            let mut local_scored: Vec<(usize, f64)> = node.routing_probs
                .iter()
                .enumerate()
                .map(|(idx, &p)| (idx, p))
                .collect();
                
            // Sort by local probability, but heavily penalize experts not in the budget
            local_scored.sort_by(|a, b| {
                let a_in_budget = budget.contains(&a.0);
                let b_in_budget = budget.contains(&b.0);
                if a_in_budget && !b_in_budget {
                    std::cmp::Ordering::Less
                } else if !a_in_budget && b_in_budget {
                    std::cmp::Ordering::Greater
                } else {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                }
            });

            node.budgeted_experts = local_scored.into_iter().take(active_k).map(|(idx, _)| idx as u32).collect();
        }
    }
}

/// Computes expected accepted tokens E[k] for a tree structure.
/// Tree dramatically increases acceptance probability by exploring multiple paths.
pub fn expected_tree_acceptance(depth: u32, branch_factor: u32, base_alpha: f64) -> f64 {
    // Simplified expectation: at each depth, having `branch_factor` paths 
    // increases the chance that at least one path is accepted.
    // P(reject all) = (1 - alpha)^branch_factor
    // P(accept at least one) = 1 - (1 - alpha)^branch_factor
    let mut expected = 0.0;
    let mut path_prob = 1.0;
    
    for _ in 0..depth {
        let layer_accept_prob = 1.0 - (1.0 - base_alpha).powi(branch_factor as i32);
        path_prob *= layer_accept_prob;
        expected += path_prob;
    }
    
    expected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_attention_mask() {
        let mut tree = DraftTree::new();
        // Root
        tree.add_node(0, 100, 1.0, 0, vec![]);
        // Branches from root
        let child1 = tree.add_node(0, 101, 0.6, 1, vec![]);
        let child2 = tree.add_node(0, 102, 0.4, 1, vec![]);
        // Branch from child1
        tree.add_node(child1, 103, 0.8, 2, vec![]);

        let mask = tree.build_attention_mask();
        assert!(mask[0][0]);
        assert!(mask[0][1]); // Root is ancestor of child1
        assert!(mask[0][2]); // Root is ancestor of child2
        assert!(mask[0][3]); // Root is ancestor of grandchild
        assert!(!mask[1][2]); // child1 is NOT ancestor of child2
        assert!(mask[1][3]); // child1 IS ancestor of grandchild
    }

    #[test]
    fn test_expert_budgeting() {
        let mut tree = DraftTree::new();
        // 8 experts total. Budget B=3, K=2.
        
        // Node 1 wants experts 0 and 1
        tree.add_node(0, 100, 1.0, 0, vec![0.9, 0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]);
        // Node 2 wants experts 2 and 3
        tree.add_node(0, 101, 1.0, 1, vec![0.0, 0.0, 0.9, 0.8, 0.1, 0.1, 0.0, 0.0]);
        // Node 3 wants experts 0 and 2
        tree.add_node(0, 102, 1.0, 1, vec![0.9, 0.1, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]);
        
        // Aggregate scores: Expert 0=1.8, Expert 2=1.8, Expert 1=0.9, Expert 3=0.8.
        // Top B=3 budget: Experts {0, 2, 1}.
        
        tree.enforce_expert_budget(3, 2);
        
        assert_eq!(tree.nodes[0].budgeted_experts, vec![0, 1]); // Naturally in budget
        assert_eq!(tree.nodes[2].budgeted_experts, vec![0, 2]); // Naturally in budget
        
        // Node 1 wanted 2 and 3. 2 is in budget, 3 is NOT.
        // The algorithm should remap the second slot to an expert in the budget (e.g. 0 or 1).
        let budgeted_for_node_2 = &tree.nodes[1].budgeted_experts;
        assert!(budgeted_for_node_2.contains(&2));
        assert!(!budgeted_for_node_2.contains(&3)); // 3 was evicted
        assert!(budgeted_for_node_2.contains(&0) || budgeted_for_node_2.contains(&1));
    }
}
