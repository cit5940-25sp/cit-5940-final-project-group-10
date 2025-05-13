package graph.core;

import java.util.ArrayList;
import java.util.List;

/**
 * A basic tree data structure implementation.
 * @param <T> The type of data stored in the tree nodes
 */
public class Tree<T> {
    private TreeNode<T> root;
    
    /**
     * Creates a new tree with the specified root node.
     * @param root The root node of the tree
     */
    public Tree(TreeNode<T> root) {
        this.root = root;
    }
    
    /**
     * Gets the root node of the tree.
     * @return The root node
     */
    public TreeNode<T> getRoot() {
        return root;
    }
    
    /**
     * Sets a new root node for the tree.
     * @param root The new root node
     */
    public void setRoot(TreeNode<T> root) {
        this.root = root;
    }
    
    /**
     * Gets all leaf nodes in the tree.
     * @return A list of all leaf nodes
     * @throws NullPointerException if the root is null
     */
    public List<TreeNode<T>> getLeaves() {
        if (root == null) {
            throw new NullPointerException("Tree root is null");
        }
        
        List<TreeNode<T>> leaves = new ArrayList<>();
        collectLeaves(root, leaves);
        return leaves;
    }
    
    /**
     * Recursively collects leaf nodes starting from the specified node.
     * @param node The current node to check
     * @param leaves The list to collect leaf nodes
     */
    private void collectLeaves(TreeNode<T> node, List<TreeNode<T>> leaves) {
        if (node.isLeaf()) {
            leaves.add(node);
        } else {
            for (TreeNode<T> child : node.getChildren()) {
                collectLeaves(child, leaves);
            }
        }
    }
}