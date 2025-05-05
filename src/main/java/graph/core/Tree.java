package graph.core;

import java.util.List;
import java.util.ArrayList;

/**
 * Represents a tree data structure.
 * @param <T> The type of data stored in the tree nodes
 */
public class Tree<T> {
    private TreeNode<T> root;
    
    /**
     * Creates a new tree with the specified root
     * @param root The root node of the tree
     */
    public Tree(TreeNode<T> root) {
        this.root = root;
    }
    
    /**
     * Gets the root of the tree
     * @return The root node
     */
    public TreeNode<T> getRoot() {
        return root;
    }
    
    /**
     * Sets the root of the tree
     * @param root The new root node
     */
    public void setRoot(TreeNode<T> root) {
        this.root = root;
    }
    
    /**
     * Gets all leaf nodes in the tree
     * @return List of leaf nodes
     */
    public List<TreeNode<T>> getLeaves() {
        List<TreeNode<T>> leaves = new ArrayList<>();
        collectLeaves(root, leaves);
        return leaves;
    }
    
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