package graph.traversal;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import graph.core.TreeNode;
import graph.core.TestTreeNode;

import java.util.ArrayList;
import java.util.List;

/**
 * Tests for the DepthFirst traversal algorithm.
 */
public class DepthFirstTest {
    
    private TestTreeNode<String> root;
    
    @BeforeEach
    public void setUp() {
        // Create a test tree
        root = TestTreeNode.createTestTree();
    }
    
    @Test
    public void testTraverseOrder() {
        // Test depth-first traversal
        List<String> visitedNodes = new ArrayList<>();
        
        DepthFirst.traverse(root, node -> visitedNodes.add((String)node.getData()));
        
        // Expected DFS order: A, B, E, F, C, G, D
        List<String> expected = List.of("A", "B", "E", "F", "C", "G", "D");
        assertEquals(expected, visitedNodes);
    }
    
    @Test
    public void testTraverseWithNullRoot() {
        // Test traversal with null root
        List<String> visitedNodes = new ArrayList<>();
        
        // Should not throw exception
        DepthFirst.traverse(null, node -> visitedNodes.add((String)node.getData()));
        
        // Nothing should be visited
        assertTrue(visitedNodes.isEmpty());
    }
    
    @Test
    public void testCollectNodes() {
        // Test collecting nodes
        List<TreeNode<String>> nodes = DepthFirst.collectNodes(root);
        
        // Check size
        assertEquals(7, nodes.size());
        
        // Check order
        assertEquals("A", (String)nodes.get(0).getData());
        assertEquals("B", (String)nodes.get(1).getData());
        assertEquals("E", (String)nodes.get(2).getData());
        assertEquals("F", (String)nodes.get(3).getData());
        assertEquals("C", (String)nodes.get(4).getData());
        assertEquals("G", (String)nodes.get(5).getData());
        assertEquals("D", (String)nodes.get(6).getData());
    }
    
    @Test
    public void testCollectNodesWithNullRoot() {
        // Test collecting nodes with null root
        List<TreeNode<String>> nodes = DepthFirst.collectNodes(null);
        
        // Should return empty list
        assertNotNull(nodes);
        assertTrue(nodes.isEmpty());
    }
    
    @Test
    public void testTraverseEmptyTree() {
        // Create a tree with just a root
        TestTreeNode<String> emptyRoot = new TestTreeNode<>("Root");
        
        List<String> visitedNodes = new ArrayList<>();
        DepthFirst.traverse(emptyRoot, node -> visitedNodes.add((String)node.getData()));
        
        // Only root should be visited
        assertEquals(1, visitedNodes.size());
        assertEquals("Root", visitedNodes.get(0));
    }
}