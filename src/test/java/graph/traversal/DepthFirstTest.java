package graph.traversal;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;

import graph.core.TreeNode;
import graph.core.TestTreeNode;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Comprehensive tests for the DepthFirst traversal algorithm.
 */
@DisplayName("DepthFirst Traversal Tests")
public class DepthFirstTest {
    
    private TestTreeNode<String> root;
    private TestTreeNode<String> customRoot;
    private TestTreeNode<String> nodeB;
    private TestTreeNode<String> nodeC;
    private TestTreeNode<String> nodeD;
    private TestTreeNode<String> nodeE;
    private TestTreeNode<String> nodeF;
    private TestTreeNode<String> nodeG;
    private TestTreeNode<String> nodeH;
    
    @BeforeEach
    public void setUp() {
        // Create a standard test tree
        root = TestTreeNode.createTestTree();
        
        // Create a custom test tree:
        //         A
        //       / | \
        //      B  C  D
        //     /\    /\
        //    E  F  G  H
        customRoot = new TestTreeNode<>("A");
        nodeB = new TestTreeNode<>("B");
        nodeC = new TestTreeNode<>("C");
        nodeD = new TestTreeNode<>("D");
        nodeE = new TestTreeNode<>("E");
        nodeF = new TestTreeNode<>("F");
        nodeG = new TestTreeNode<>("G");
        nodeH = new TestTreeNode<>("H");
        
        customRoot.addChild(nodeB);
        customRoot.addChild(nodeC);
        customRoot.addChild(nodeD);
        
        nodeB.addChild(nodeE);
        nodeB.addChild(nodeF);
        nodeD.addChild(nodeG);
        nodeD.addChild(nodeH);
    }
    
    @Test
    @DisplayName("Test standard depth-first traversal order")
    public void testTraverseOrder() {
        // Test depth-first traversal
        List<String> visitedNodes = new ArrayList<>();
        
        DepthFirst.traverse(root, node -> visitedNodes.add((String)node.getData()));
        
        // Expected DFS order: A, B, E, F, C, G, D
        List<String> expected = List.of("A", "B", "E", "F", "C", "G", "D");
        assertEquals(expected, visitedNodes);
    }
    
    @Test
    @DisplayName("Test traversal with null root")
    public void testTraverseWithNullRoot() {
        // Test traversal with null root
        List<String> visitedNodes = new ArrayList<>();
        
        // Should not throw exception
        DepthFirst.traverse(null, node -> visitedNodes.add((String)node.getData()));
        
        // Nothing should be visited
        assertTrue(visitedNodes.isEmpty());
    }
    
    @Test
    @DisplayName("Test collectNodes basic functionality")
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
    @DisplayName("Test collecting nodes with null root")
    public void testCollectNodesWithNullRoot() {
        // Test collecting nodes with null root
        List<TreeNode<String>> nodes = DepthFirst.collectNodes(null);
        
        // Should return empty list
        assertNotNull(nodes);
        assertTrue(nodes.isEmpty());
    }
    
    @Test
    @DisplayName("Test traversal of empty tree")
    public void testTraverseEmptyTree() {
        // Create a tree with just a root
        TestTreeNode<String> emptyRoot = new TestTreeNode<>("Root");
        
        List<String> visitedNodes = new ArrayList<>();
        DepthFirst.traverse(emptyRoot, node -> visitedNodes.add((String)node.getData()));
        
        // Only root should be visited
        assertEquals(1, visitedNodes.size());
        assertEquals("Root", visitedNodes.get(0));
    }
    
    @Test
    @DisplayName("Test custom tree traversal")
    public void testTraverseCustomTree() {
        List<String> visitedNodes = new ArrayList<>();
        DepthFirst.traverse(customRoot, node -> visitedNodes.add((String)node.getData()));
        
        // Expected depth-first order: A, B, E, F, C, D, G, H
        assertEquals(8, visitedNodes.size());
        assertEquals("A", visitedNodes.get(0));
        assertEquals("B", visitedNodes.get(1));
        assertEquals("E", visitedNodes.get(2));
        assertEquals("F", visitedNodes.get(3));
        assertEquals("C", visitedNodes.get(4));
        assertEquals("D", visitedNodes.get(5));
        assertEquals("G", visitedNodes.get(6));
        assertEquals("H", visitedNodes.get(7));
    }
    
    @Test
    @DisplayName("Test traversal with modifying action")
    public void testTraverseWithModifyingAction() {
        // Create action that appends a suffix to each node's data
        DepthFirst.traverse(customRoot, node -> {
            if (node instanceof TestTreeNode) {
                TestTreeNode<String> testNode = (TestTreeNode<String>)node;
                testNode.setData(testNode.getData() + "_visited");
            }
        });
        
        // Verify that all nodes were modified
        assertEquals("A_visited", customRoot.getData());
        assertEquals("B_visited", nodeB.getData());
        assertEquals("C_visited", nodeC.getData());
        assertEquals("D_visited", nodeD.getData());
        assertEquals("E_visited", nodeE.getData());
        assertEquals("F_visited", nodeF.getData());
        assertEquals("G_visited", nodeG.getData());
        assertEquals("H_visited", nodeH.getData());
    }
    
    @Test
    @DisplayName("Test traversal of deep tree")
    public void testTraverseDeepTree() {
        // Create a deeper tree: A -> B -> C -> D -> E -> F
        TestTreeNode<String> deepRoot = new TestTreeNode<>("A");
        TestTreeNode<String> level1 = new TestTreeNode<>("B");
        TestTreeNode<String> level2 = new TestTreeNode<>("C");
        TestTreeNode<String> level3 = new TestTreeNode<>("D");
        TestTreeNode<String> level4 = new TestTreeNode<>("E");
        TestTreeNode<String> level5 = new TestTreeNode<>("F");
        
        deepRoot.addChild(level1);
        level1.addChild(level2);
        level2.addChild(level3);
        level3.addChild(level4);
        level4.addChild(level5);
        
        List<String> visited = new ArrayList<>();
        DepthFirst.traverse(deepRoot, node -> visited.add((String)node.getData()));
        
        // Expected depth-first order: A, B, C, D, E, F
        assertEquals(6, visited.size(), "Should visit all 6 nodes");
        assertEquals("A", visited.get(0));
        assertEquals("B", visited.get(1));
        assertEquals("C", visited.get(2));
        assertEquals("D", visited.get(3));
        assertEquals("E", visited.get(4));
        assertEquals("F", visited.get(5));
    }
    
    @Test
    @DisplayName("Test traversal of wide tree")
    public void testTraverseWideTree() {
        // Create a wider tree: A -> B1, B2, B3, B4, B5
        TestTreeNode<String> wideRoot = new TestTreeNode<>("A");
        TestTreeNode<String> child1 = new TestTreeNode<>("B1");
        TestTreeNode<String> child2 = new TestTreeNode<>("B2");
        TestTreeNode<String> child3 = new TestTreeNode<>("B3");
        TestTreeNode<String> child4 = new TestTreeNode<>("B4");
        TestTreeNode<String> child5 = new TestTreeNode<>("B5");
        
        wideRoot.addChild(child1);
        wideRoot.addChild(child2);
        wideRoot.addChild(child3);
        wideRoot.addChild(child4);
        wideRoot.addChild(child5);
        
        List<String> visited = new ArrayList<>();
        DepthFirst.traverse(wideRoot, node -> visited.add((String)node.getData()));
        
        // Expected depth-first order: A, B1, B2, B3, B4, B5
        assertEquals(6, visited.size(), "Should visit all 6 nodes");
        assertEquals("A", visited.get(0));
        assertTrue(visited.subList(1, 6).containsAll(List.of("B1", "B2", "B3", "B4", "B5")), 
                 "Should contain all children");
    }
    
    @Test
    @DisplayName("Test comparison with BreadthFirst traversal")
    public void testComparisonWithBreadthFirst() {
        // BFS and DFS should visit all nodes but in different orders
        List<String> bfsVisited = new ArrayList<>();
        BreadthFirst.traverse(customRoot, node -> bfsVisited.add((String)node.getData()));
        
        List<String> dfsVisited = new ArrayList<>();
        DepthFirst.traverse(customRoot, node -> dfsVisited.add((String)node.getData()));
        
        // Both should visit all 8 nodes
        assertEquals(8, bfsVisited.size());
        assertEquals(8, dfsVisited.size());
        
        // Both should start with the root
        assertEquals(bfsVisited.get(0), dfsVisited.get(0));
        
        // But the orders should be different for trees with >1 level
        assertFalse(bfsVisited.equals(dfsVisited), 
                  "BFS and DFS should visit nodes in different orders");
    }
    
    @Test
    @DisplayName("Test counter with the traversal")
    public void testTraverseWithCounter() {
        AtomicInteger counter = new AtomicInteger(0);
        DepthFirst.traverse(customRoot, node -> counter.incrementAndGet());
        
        assertEquals(8, counter.get(), "Should count 8 nodes in the tree");
    }
    
    @Test
    @DisplayName("Test stack overflow prevention")
    public void testRecursionLimit() {
        // Test a reasonably deep tree to ensure no stack overflow
        int depth = 1000;  // Deep enough to test, but not too deep to cause real overflow
        
        TestTreeNode<String> deepRoot = new TestTreeNode<>("Root");
        TestTreeNode<String> current = deepRoot;
        
        // Create a deep linear tree
        for (int i = 0; i < depth; i++) {
            TestTreeNode<String> child = new TestTreeNode<>("Level" + i);
            current.addChild(child);
            current = child;
        }
        
        // Should not throw StackOverflowError
        AtomicInteger counter = new AtomicInteger(0);
        DepthFirst.traverse(deepRoot, node -> counter.incrementAndGet());
        
        assertEquals(depth + 1, counter.get(), "Should count all nodes in the deep tree");
    }
}