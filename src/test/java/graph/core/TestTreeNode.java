package graph.core;

import java.util.ArrayList;
import java.util.List;

/**
 * Simple implementation of TreeNode for testing purposes.
 */
public class TestTreeNode<T> implements TreeNode<T> {
    private T data;
    private TreeNode<T> parent;
    private List<TreeNode<T>> children;
    
    public TestTreeNode(T data) {
        this.data = data;
        this.children = new ArrayList<>();
    }
    
    @Override
    public T getData() {
        return data;
    }
    
    @Override
    public List<TreeNode<T>> getChildren() {
        return children;
    }
    
    @Override
    public TreeNode<T> getParent() {
        return parent;
    }
    
    @Override
    public void addChild(TreeNode<T> child) {
        children.add(child);
        if (child instanceof TestTreeNode) {
            ((TestTreeNode<T>) child).setParent(this);
        }
    }
    
    @Override
    public boolean isLeaf() {
        return children.isEmpty();
    }
    
    public void setParent(TreeNode<T> parent) {
        this.parent = parent;
    }
    
    @Override
    public String toString() {
        return "Node(" + data + ")";
    }
    
    /**
     * Creates a test tree with specified structure.
     * @return Root node of a test tree
     */
    public static TestTreeNode<String> createTestTree() {
        /*
         * Creates a tree with this structure:
         *       A
         *     / | \
         *    B  C  D
         *   /|  |
         *  E F  G
         */
        TestTreeNode<String> root = new TestTreeNode<>("A");
        TestTreeNode<String> nodeB = new TestTreeNode<>("B");
        TestTreeNode<String> nodeC = new TestTreeNode<>("C");
        TestTreeNode<String> nodeD = new TestTreeNode<>("D");
        TestTreeNode<String> nodeE = new TestTreeNode<>("E");
        TestTreeNode<String> nodeF = new TestTreeNode<>("F");
        TestTreeNode<String> nodeG = new TestTreeNode<>("G");
        
        root.addChild(nodeB);
        root.addChild(nodeC);
        root.addChild(nodeD);
        
        nodeB.addChild(nodeE);
        nodeB.addChild(nodeF);
        nodeC.addChild(nodeG);
        
        return root;
    }
}