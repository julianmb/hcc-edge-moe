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
    pub fn add_node(&mut self, parent_id: u32, token_id: u32, prob: f64, depth: u32) -> u32 {
        let node_id = self.nodes.len() as u32;
        self.nodes.push(TreeNode {
            node_id,
            parent_id,
            token_id,
            probability: prob,
            depth,
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
    pub fn flatten_tokens(&self) -> Vec<u32> {
        self.nodes.iter().map(|n| n.token_id).collect()
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
        tree.add_node(0, 100, 1.0, 0);
        // Branches from root
        let child1 = tree.add_node(0, 101, 0.6, 1);
        let child2 = tree.add_node(0, 102, 0.4, 1);
        // Branch from child1
        tree.add_node(child1, 103, 0.8, 2);

        let mask = tree.build_attention_mask();
        assert!(mask[0][0]);
        assert!(mask[0][1]); // Root is ancestor of child1
        assert!(mask[0][2]); // Root is ancestor of child2
        assert!(mask[0][3]); // Root is ancestor of grandchild
        assert!(!mask[1][2]); // child1 is NOT ancestor of child2
        assert!(mask[1][3]); // child1 IS ancestor of grandchild
    }
}
