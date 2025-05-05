package graph.core;

import java.util.List;

/**
 * Interface for a node in a tree structure.
 * @param <T> The type of data stored in the node
 */
public interface TreeNode<T> {
    /**
     * Gets the data stored in this node
     * @return The data
     */
    T getData();
    
    /**
     * Gets all children of this node
     * @return List of child nodes
     */
    List<? extends TreeNode<T>> getChildren();
    
    /**
     * Gets the parent of this node
     * @return The parent node, or null if this is the root
     */
    TreeNode<T> getParent();
    
    /**
     * Adds a child to this node
     * @param child The child node to add
     */
    void addChild(TreeNode<T> child);
    
    /**
     * Checks if this node is a leaf (has no children)
     * @return true if this node is a leaf, false otherwise
     */
    boolean isLeaf();
}