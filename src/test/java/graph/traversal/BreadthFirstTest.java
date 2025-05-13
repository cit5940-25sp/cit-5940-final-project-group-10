package graph.traversal;

import graph.core.TreeNode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for BreadthFirst traversal algorithm.
 */
@DisplayName("BreadthFirst Traversal Tests")
public class BreadthFirstTest {
    
    private TestTreeNode<String> root;
    private TestTreeNode<String> child1;
    private TestTreeNode<String> child2;
    private TestTreeNode<String> child3;
    private TestTreeNode<String> grandChild1;
    private TestTreeNode<String> grandChild2;
    private TestTreeNode<String> grandChild3;
    private TestTreeNode<String> grandChild4;
    
    @BeforeEach
    public void setUp() {
        // Create a test tree:
        //         A
        //       / | \
        //      B  C  D
        //     /\    /\
        //    E  F  G  H
        root = new TestTreeNode<>("A");
        child1 = new TestTreeNode<>("B");
        child2 = new TestTreeNode<>("C");
        child3 = new TestTreeNode<>("D");
        grandChild1 = new TestTreeNode<>("E");
        grandChild2 = new TestTreeNode<>("F");
        grandChild3 = new TestTreeNode<>("G");
        grandChild4 = new TestTreeNode<>("H");
        
        root.addChild(child1);
        root.addChild(child2);
        root.addChild(child3);
        
        child1.addChild(grandChild1);
        child1.addChild(grandChild2);
        child3.addChild(grandChild3);
        child3.addChild(grandChild4);
    }
    
    @Test
    @DisplayName("Test traversal with null root")
    public void testTraverseNullRoot() {
        // Traversal with null root should not throw exceptions
        AtomicInteger counter = new AtomicInteger(0);
        BreadthFirst.traverse(null, node -> counter.incrementAndGet());
        
        assertEquals(0, counter.get(), "Traversal of null tree should not visit any nodes");
    }
    
    @Test
    @DisplayName("Test traversal with single node")
    public void testTraverseSingleNode() {
        TestTreeNode<String> singleNode = new TestTreeNode<>("Single");
        
        List<String> visited = new ArrayList<>();
        BreadthFirst.traverse(singleNode, node -> visited.add(node.getData()));
        
        assertEquals(1, visited.size(), "Should visit exactly one node");
        assertEquals("Single", visited.get(0), "Should visit the node with correct data");
    }
    
    @Test
    @DisplayName("Test breadth-first order with complete tree")
    public void testTraverseCompleteTree() {
        List<String> visited = new ArrayList<>();
        BreadthFirst.traverse(root, node -> visited.add(node.getData()));
        
        // Expected breadth-first order: A, B, C, D, E, F, G, H
        assertEquals(8, visited.size(), "Should visit all 8 nodes");
        assertEquals("A", visited.get(0), "Root should be visited first");
        
        // Level 1: B, C, D (order of same-level nodes depends on how they were added)
        assertEquals("B", visited.get(1), "Child 1 should be visited second");
        assertEquals("C", visited.get(2), "Child 2 should be visited third");
        assertEquals("D", visited.get(3), "Child 3 should be visited fourth");
        
        // Level 2: E, F, G, H (all grandchildren)
        assertEquals("E", visited.get(4), "Grandchild 1 should be visited fifth");
        assertEquals("F", visited.get(5), "Grandchild 2 should be visited sixth");
        assertEquals("G", visited.get(6), "Grandchild 3 should be visited seventh");
        assertEquals("H", visited.get(7), "Grandchild 4 should be visited eighth");
    }
    
    @Test
    @DisplayName("Test collectNodes method")
    public void testCollectNodes() {
        List<TreeNode<String>> nodes = BreadthFirst.collectNodes(root);
        
        assertEquals(8, nodes.size(), "Should collect all 8 nodes");
        assertEquals("A", nodes.get(0).getData(), "Root should be first in the list");
        assertEquals("B", nodes.get(1).getData(), "Child 1 should be second in the list");
        assertEquals("C", nodes.get(2).getData(), "Child 2 should be third in the list");
        assertEquals("D", nodes.get(3).getData(), "Child 3 should be fourth in the list");
        assertEquals("E", nodes.get(4).getData(), "Grandchild 1 should be fifth in the list");
        assertEquals("F", nodes.get(5).getData(), "Grandchild 2 should be sixth in the list");
        assertEquals("G", nodes.get(6).getData(), "Grandchild 3 should be seventh in the list");
        assertEquals("H", nodes.get(7).getData(), "Grandchild 4 should be eighth in the list");
    }
    
    @Test
    @DisplayName("Test collectNodes with null root")
    public void testCollectNodesNullRoot() {
        List<TreeNode<String>> nodes = BreadthFirst.collectNodes(null);
        
        assertTrue(nodes.isEmpty(), "Collecting nodes from null tree should return empty list");
    }
    
    @Test
    @DisplayName("Test traversal with deep tree")
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
        BreadthFirst.traverse(deepRoot, node -> visited.add(node.getData()));
        
        // Expected breadth-first order: A, B, C, D, E, F
        assertEquals(6, visited.size(), "Should visit all 6 nodes");
        assertEquals("A", visited.get(0));
        assertEquals("B", visited.get(1));
        assertEquals("C", visited.get(2));
        assertEquals("D", visited.get(3));
        assertEquals("E", visited.get(4));
        assertEquals("F", visited.get(5));
    }
    
    @Test
    @DisplayName("Test traversal with wide tree")
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
        BreadthFirst.traverse(wideRoot, node -> visited.add(node.getData()));
        
        // Expected breadth-first order: A, B1, B2, B3, B4, B5
        assertEquals(6, visited.size(), "Should visit all 6 nodes");
        assertEquals("A", visited.get(0));
        assertTrue(visited.subList(1, 6).containsAll(List.of("B1", "B2", "B3", "B4", "B5")), 
                  "Should contain all children at level 1");
    }
    
    @Test
    @DisplayName("Test traversal with action that modifies nodes")
    public void testTraverseWithModifyingAction() {
        // Create action that appends a suffix to each node's data
        BreadthFirst.traverse(root, node -> {
            if (node instanceof TestTreeNode) {
                TestTreeNode<String> testNode = (TestTreeNode<String>)node;
                testNode.setData(testNode.getData() + "_visited");
            }
        });
        
        // Verify that all nodes were modified
        assertEquals("A_visited", root.getData());
        assertEquals("B_visited", child1.getData());
        assertEquals("C_visited", child2.getData());
        assertEquals("D_visited", child3.getData());
        assertEquals("E_visited", grandChild1.getData());
        assertEquals("F_visited", grandChild2.getData());
        assertEquals("G_visited", grandChild3.getData());
        assertEquals("H_visited", grandChild4.getData());
    }
    
    @Test
    @DisplayName("Test comparison with DepthFirst traversal")
    public void testComparisonWithDepthFirst() {
        // BFS and DFS should visit all nodes but in different orders
        List<String> bfsVisited = new ArrayList<>();
        BreadthFirst.traverse(root, node -> bfsVisited.add(node.getData()));
        
        List<String> dfsVisited = new ArrayList<>();
        DepthFirst.traverse(root, node -> dfsVisited.add(node.getData()));
        
        // Both should visit all 8 nodes
        assertEquals(8, bfsVisited.size());
        assertEquals(8, dfsVisited.size());
        
        // Both should start with the root
        assertEquals(bfsVisited.get(0), dfsVisited.get(0));
        
        // But the orders should be different
        assertFalse(bfsVisited.equals(dfsVisited), 
                  "BFS and DFS should visit nodes in different orders");
        
        // BFS should visit all level 1 nodes before any level 2 nodes
        List<String> level1Nodes = List.of("B", "C", "D");
        List<String> level2Nodes = List.of("E", "F", "G", "H");
        
        // For BFS, all level1 nodes should come before all level2 nodes
        for (String level1Node : level1Nodes) {
            for (String level2Node : level2Nodes) {
                int level1Index = bfsVisited.indexOf(level1Node);
                int level2Index = bfsVisited.indexOf(level2Node);
                assertTrue(level1Index < level2Index, 
                        "BFS should visit level 1 nodes before level 2 nodes");
            }
        }
    }
    
    @Test
    @DisplayName("Test traversal of a very large tree")
    public void testLargeTreeTraversal() {
        // Create a tree with many nodes to test performance and memory usage
        TestTreeNode<String> largeRoot = new TestTreeNode<>("Root");
        
        // Add 100 children to the root
        for (int i = 0; i < 100; i++) {
            TestTreeNode<String> child = new TestTreeNode<>("Child" + i);
            largeRoot.addChild(child);
            
            // Add 10 grandchildren to each child
            for (int j = 0; j < 10; j++) {
                child.addChild(new TestTreeNode<>("GrandChild" + i + "_" + j));
            }
        }
        
        // This tree should have 1 + 100 + 1000 = 1101 nodes
        AtomicInteger counter = new AtomicInteger(0);
        BreadthFirst.traverse(largeRoot, node -> counter.incrementAndGet());
        
        assertEquals(1101, counter.get(), "Should visit all 1101 nodes in the large tree");
    }
    
    @Test
    @DisplayName("Test traversal when node has multiple parents")
    public void testNodeWithMultipleParents() {
        // Create a test tree where a node appears in multiple places
        TestTreeNode<String> commonNode = new TestTreeNode<>("Common");
        
        // Root tree
        TestTreeNode<String> rootA = new TestTreeNode<>("A");
        TestTreeNode<String> nodeB = new TestTreeNode<>("B");
        TestTreeNode<String> nodeC = new TestTreeNode<>("C");
        
        rootA.addChild(nodeB);
        rootA.addChild(nodeC);
        
        // Both B and C point to the same common node
        // This creates a diamond pattern A -> B -> Common and A -> C -> Common
        nodeB.addChild(commonNode);
        nodeC.addChild(commonNode);
        
        List<String> visited = new ArrayList<>();
        BreadthFirst.traverse(rootA, node -> visited.add(node.getData()));
        
        // Expected order: A, B, C, Common, Common
        // BFS should visit the common node twice because it's seen from two parents
        assertEquals(4, visited.size(), "Should visit 4 nodes, with Common counted only once");
        assertEquals("A", visited.get(0));
        assertTrue(visited.subList(1, 3).containsAll(List.of("B", "C")));
        assertEquals("Common", visited.get(3));
    }
    
    /**
     * Simple implementation of TreeNode for testing.
     */
    private static class TestTreeNode<T> implements TreeNode<T> {
        private T data;
        private final List<TestTreeNode<T>> children = new ArrayList<>();
        private TreeNode<T> parent;
        
        public TestTreeNode(T data) {
            this.data = data;
        }
        
        @Override
        public T getData() {
            return data;
        }
        
        public void setData(T data) {
            this.data = data;
        }
        
        @Override
        public List<TestTreeNode<T>> getChildren() {
            return children;
        }
        
        @Override
        public TreeNode<T> getParent() {
            return parent;
        }
        
        @Override
        public void addChild(TreeNode<T> child) {
            if (child instanceof TestTreeNode) {
                TestTreeNode<T> testChild = (TestTreeNode<T>)child;
                children.add(testChild);
                testChild.parent = this;
            }
        }
        
        @Override
        public boolean isLeaf() {
            return children.isEmpty();
        }
    }
}