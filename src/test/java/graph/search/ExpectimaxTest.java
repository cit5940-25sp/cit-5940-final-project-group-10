package graph.search;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.function.BiFunction;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the Expectimax algorithm.
 */
public class ExpectimaxTest {
    
    private GameTreeNode<Integer> root;
    private BiFunction<Integer, Boolean, Double> evaluator;
    
    @BeforeEach
    public void setUp() {
        // Create a simple game tree
        // The nodes contain integers that will be directly used as scores
        //      0
        //    /   \
        //   1     2
        //  / \   / \
        // 3   4 5   6
        root = new GameTreeNode<>(0);
        
        GameTreeNode<Integer> node1 = root.addChild(1);
        GameTreeNode<Integer> node2 = root.addChild(2);
        
        node1.addChild(3);
        node1.addChild(4);
        
        node2.addChild(5);
        node2.addChild(6);
        
        // Simple evaluator that just returns the node's value
        evaluator = (value, maximizing) -> (double) value;
    }
    
    @Test
    public void testExpectimaxMaximizingPlayer() {
        // Perform expectimax search starting with maximizing player
        double score = Expectimax.search(root, 2, true, evaluator);
        
        // Verify the result
        // For maximizing player at root:
        // Node 1: Expectation of [3, 4] = (3 + 4)/2 = 3.5
        // Node 2: Expectation of [5, 6] = (5 + 6)/2 = 5.5
        // Root: Max of [3.5, 5.5] = 5.5
        assertEquals(5.5, score, 0.001);
        assertEquals(5.5, root.getScore(), 0.001);
    }
    
    @Test
    public void testExpectimaxChanceNode() {
        // Perform expectimax search starting with chance node
        double score = Expectimax.search(root, 2, false, evaluator);
        
        // Verify the result
        // For chance node at root:
        // Node 1: Max of [3, 4] = 4
        // Node 2: Max of [5, 6] = 6
        // Root: Expectation of [4, 6] = (4 + 6)/2 = 5
        assertEquals(5.0, score, 0.001);
        assertEquals(5.0, root.getScore(), 0.001);
    }
    
    @Test
    public void testExpectimaxWithDepthLimit() {
        // Perform expectimax search with depth limit of 1 (don't go to leaf nodes)
        double score = Expectimax.search(root, 1, true, evaluator);
        
        // Verify the result
        // With depth limit 1, we evaluate node1 and node2 directly
        // Root: Max of [1, 2] = 2
        assertEquals(2.0, score, 0.001);
        assertEquals(2.0, root.getScore(), 0.001);
    }
    
    @Test
    public void testExpectimaxWithLeafNode() {
        // Create a leaf node
        GameTreeNode<Integer> leaf = new GameTreeNode<>(42);
        
        // Perform expectimax search on leaf node
        double score = Expectimax.search(leaf, 2, true, evaluator);
        
        // For a leaf node, the score should be the evaluated value
        assertEquals(42.0, score, 0.001);
        assertEquals(42.0, leaf.getScore(), 0.001);
    }
    
    @Test
    public void testExpectimaxWithEmptyChanceNode() {
        // Create a node with no children to test the edge case in the chance node logic
        GameTreeNode<Integer> emptyChanceNode = new GameTreeNode<>(7);
        
        // Perform expectimax search on the empty chance node
        double score = Expectimax.search(emptyChanceNode, 2, false, evaluator);
        
        // For an empty chance node, it should return the evaluated value
        assertEquals(7.0, score, 0.001);
        assertEquals(7.0, emptyChanceNode.getScore(), 0.001);
    }
    
    @Test
    public void testExpectimaxDifferentProbabilities() {
        // Create a tree with different number of children in different branches
        //      0
        //    /   \
        //   1     2
        //  /     / \
        // 3     5   6
        GameTreeNode<Integer> unbalancedRoot = new GameTreeNode<>(0);
        
        GameTreeNode<Integer> unbalancedNode1 = unbalancedRoot.addChild(1);
        GameTreeNode<Integer> unbalancedNode2 = unbalancedRoot.addChild(2);
        
        unbalancedNode1.addChild(3);
        
        unbalancedNode2.addChild(5);
        unbalancedNode2.addChild(6);
        
        // Perform expectimax search starting with chance node
        double score = Expectimax.search(unbalancedRoot, 2, false, evaluator);
        
        // Verify the result
        // For chance node at root:
        // Node 1: Max of [3] = 3
        // Node 2: Max of [5, 6] = 6
        // Root: Expectation of [3, 6] = (3 + 6)/2 = 4.5
        assertEquals(4.5, score, 0.001);
        assertEquals(4.5, unbalancedRoot.getScore(), 0.001);
    }
}