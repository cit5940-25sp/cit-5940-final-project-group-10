package graph.core;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Tests for the Tree class.
 */
public class TreeTest {
    
    private Tree<String> tree;
    private TestTreeNode<String> root;
    
    @BeforeEach
    public void setUp() {
        // Create a test tree
        root = TestTreeNode.createTestTree();
        tree = new Tree<>(root);
    }
    
    @Test
    public void testGetRoot() {
        // Check that we can get the tree root
        TreeNode<String> retrievedRoot = tree.getRoot();
        
        assertNotNull(retrievedRoot);
        assertEquals("A", retrievedRoot.getData());
        assertEquals(3, retrievedRoot.getChildren().size());
    }
    
    @Test
    public void testSetRoot() {
        // Create a new root
        TestTreeNode<String> newRoot = new TestTreeNode<>("Z");
        
        // Set it as the root
        tree.setRoot(newRoot);
        
        // Check that the root was updated
        TreeNode<String> retrievedRoot = tree.getRoot();
        assertEquals("Z", retrievedRoot.getData());
        assertTrue(retrievedRoot.isLeaf());
    }
    
    @Test
    public void testGetLeaves() {
        // Get leaf nodes
        List<TreeNode<String>> leaves = tree.getLeaves();
        
        // Extract leaf node data for easier comparison
        List<String> leafData = leaves.stream()
                .map(TreeNode::getData)
                .collect(Collectors.toList());
        
        // Should have 4 leaves: D, E, F, and G
        assertEquals(4, leaves.size());
        assertTrue(leafData.contains("D"));
        assertTrue(leafData.contains("E"));
        assertTrue(leafData.contains("F"));
        assertTrue(leafData.contains("G"));
    }
    
    @Test
    public void testGetLeavesWithSingleNodeTree() {
        // Create a tree with just one node
        TestTreeNode<String> singleNode = new TestTreeNode<>("Root");
        Tree<String> singleNodeTree = new Tree<>(singleNode);
        
        // Get leaves
        List<TreeNode<String>> leaves = singleNodeTree.getLeaves();
        
        // Should have one leaf (the root)
        assertEquals(1, leaves.size());
        assertEquals("Root", leaves.get(0).getData());
    }
    
    @Test
    public void testGetLeavesWithNullRoot() {
        // This is an edge case that may cause NullPointerException
        // It's good practice to test such cases
        
        // Create a tree with null root
        Tree<String> nullTree = new Tree<>(null);
        
        // This should cause an exception when trying to get leaves
        Exception exception = assertThrows(NullPointerException.class, () -> {
            nullTree.getLeaves();
        });
        
        // Verify exception type (this test will fail unless the method throws NullPointerException)
        assertNotNull(exception);
    }
    
    @Test
    public void testTreeWithComplexStructure() {
        // Create a more complex tree structure for testing
        TestTreeNode<String> rootNode = new TestTreeNode<>("Root");
        TestTreeNode<String> a1 = new TestTreeNode<>("A1");
        TestTreeNode<String> a2 = new TestTreeNode<>("A2");
        TestTreeNode<String> b1 = new TestTreeNode<>("B1");
        TestTreeNode<String> b2 = new TestTreeNode<>("B2");
        TestTreeNode<String> b3 = new TestTreeNode<>("B3");
        TestTreeNode<String> c1 = new TestTreeNode<>("C1");
        
        rootNode.addChild(a1);
        rootNode.addChild(a2);
        a1.addChild(b1);
        a1.addChild(b2);
        a2.addChild(b3);
        b1.addChild(c1);
        
        Tree<String> complexTree = new Tree<>(rootNode);
        
        // Test leaf collection
        List<TreeNode<String>> leaves = complexTree.getLeaves();
        List<String> leafData = leaves.stream()
                .map(TreeNode::getData)
                .collect(Collectors.toList());
        
        // Should have 3 leaves: C1, B2, and B3
        assertEquals(3, leaves.size());
        assertTrue(leafData.contains("C1"));
        assertTrue(leafData.contains("B2"));
        assertTrue(leafData.contains("B3"));
    }
}