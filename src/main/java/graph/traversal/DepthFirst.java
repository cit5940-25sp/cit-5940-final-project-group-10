package graph.traversal;

import graph.core.TreeNode;
import java.util.*;
import java.util.function.Consumer;

/**
 * Implements depth-first traversal for trees.
 */
public class DepthFirst {
    
    /**
     * Traverses a tree in depth-first order, applying the given action to each node.
     * @param <T> The type of data stored in the tree nodes
     * @param root The root node to start traversal from
     * @param action The action to apply to each node
     */
    public static <T> void traverse(TreeNode<T> root, Consumer<TreeNode<T>> action) {
        if (root == null) return;
        
        action.accept(root);
        
        for (TreeNode<T> child : root.getChildren()) {
            traverse(child, action);
        }
    }
    
    /**
     * Collects all nodes in depth-first order.
     * @param <T> The type of data stored in the tree nodes
     * @param root The root node to start traversal from
     * @return List of nodes in depth-first order
     */
    public static <T> List<TreeNode<T>> collectNodes(TreeNode<T> root) {
        List<TreeNode<T>> result = new ArrayList<>();
        traverse(root, result::add);
        return result;
    }
}