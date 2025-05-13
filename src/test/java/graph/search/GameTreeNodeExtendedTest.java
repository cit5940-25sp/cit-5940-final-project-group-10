package graph.search;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;

/**
 * Extended tests for the GameTreeNode class.
 */
public class GameTreeNodeExtendedTest {
    
    @Test
    public void testConstructor() {
        Integer data = 42;
        GameTreeNode<Integer> node = new GameTreeNode<>(data);
        
        assertEquals(data, node.getData());
        assertTrue(node.getChildren().isEmpty());
        assertNull(node.getParent());
        assertEquals(0, node.getVisits());
        assertEquals(0.0, node.getTotalScore(), 0.001);
        assertEquals(Double.NEGATIVE_INFINITY, node.getAlpha());
        assertEquals(Double.POSITIVE_INFINITY, node.getBeta());
    }
    
    @Test
    public void testSetData() {
        GameTreeNode<Integer> node = new GameTreeNode<>(42);
        node.setData(84);
        
        assertEquals(84, node.getData());
    }
    
    @Test
    public void testAddChildNode() {
        GameTreeNode<Integer> parent = new GameTreeNode<>(42);
        GameTreeNode<Integer> child = new GameTreeNode<>(43);
        
        parent.addChild(child);
        
        assertEquals(1, parent.getChildren().size());
        assertSame(child, parent.getChildren().get(0));
        assertSame(parent, child.getParent());
    }
    
    @Test
    public void testAddChildData() {
        GameTreeNode<Integer> parent = new GameTreeNode<>(42);
        GameTreeNode<Integer> child = parent.addChild(43);
        
        assertEquals(1, parent.getChildren().size());
        assertSame(child, parent.getChildren().get(0));
        assertSame(parent, child.getParent());
        assertEquals(43, child.getData());
    }
    
    @Test
    public void testAddMultipleChildren() {
        GameTreeNode<Integer> parent = new GameTreeNode<>(42);
        GameTreeNode<Integer> child1 = parent.addChild(43);
        GameTreeNode<Integer> child2 = parent.addChild(44);
        GameTreeNode<Integer> child3 = parent.addChild(45);
        
        assertEquals(3, parent.getChildren().size());
        assertSame(child1, parent.getChildren().get(0));
        assertSame(child2, parent.getChildren().get(1));
        assertSame(child3, parent.getChildren().get(2));
        
        assertSame(parent, child1.getParent());
        assertSame(parent, child2.getParent());
        assertSame(parent, child3.getParent());
    }
    
    @Test
    public void testIsLeaf() {
        GameTreeNode<Integer> leafNode = new GameTreeNode<>(42);
        assertTrue(leafNode.isLeaf());
        
        GameTreeNode<Integer> parentNode = new GameTreeNode<>(42);
        parentNode.addChild(43);
        assertFalse(parentNode.isLeaf());
    }
    
    @Test
    public void testScoreGetterSetter() {
        GameTreeNode<Integer> node = new GameTreeNode<>(42);
        
        // Default value
        assertEquals(0.0, node.getScore(), 0.001);
        
        // Set and get
        node.setScore(42.5);
        assertEquals(42.5, node.getScore(), 0.001);
    }
    
    @Test
    public void testAlphaBetaGetterSetter() {
        GameTreeNode<Integer> node = new GameTreeNode<>(42);
        
        // Default values
        assertEquals(Double.NEGATIVE_INFINITY, node.getAlpha());
        assertEquals(Double.POSITIVE_INFINITY, node.getBeta());
        
        // Set and get alpha
        node.setAlpha(10.5);
        assertEquals(10.5, node.getAlpha(), 0.001);
        
        // Set and get beta
        node.setBeta(20.5);
        assertEquals(20.5, node.getBeta(), 0.001);
    }
    
    @Test
    public void testVisitsAndTotalScore() {
        GameTreeNode<Integer> node = new GameTreeNode<>(42);
        
        // Default values
        assertEquals(0, node.getVisits());
        assertEquals(0.0, node.getTotalScore(), 0.001);
        
        // Increment visits
        node.incrementVisits();
        assertEquals(1, node.getVisits());
        
        // Add score
        node.addScore(5.5);
        assertEquals(5.5, node.getTotalScore(), 0.001);
        
        // Add more
        node.incrementVisits();
        node.addScore(4.5);
        assertEquals(2, node.getVisits());
        assertEquals(10.0, node.getTotalScore(), 0.001);
    }
    
    @ParameterizedTest
    @ValueSource(doubles = {0.1, 0.5, 1.0, 1.414, 2.0})
    public void testUCB1WithDifferentExplorationParams(double explorationParam) {
        // Create a parent with some visits
        GameTreeNode<Integer> parent = new GameTreeNode<>(0);
        for (int i = 0; i < 10; i++) {
            parent.incrementVisits();
        }
        
        // Create a child with some visits and score
        GameTreeNode<Integer> child = parent.addChild(1);
        for (int i = 0; i < 5; i++) {
            child.incrementVisits();
        }
        child.addScore(10.0);
        
        // Calculate UCB1
        double ucb1 = child.getUCB1(explorationParam);
        
        // Expected UCB1 values
        // Exploitation = 10.0 / 5 = 2.0
        // Exploration = explorationParam * sqrt(ln(10) / 5)
        double expectedExploitation = 2.0;
        double expectedExploration = explorationParam * Math.sqrt(Math.log(10) / 5);
        double expectedUCB1 = expectedExploitation + expectedExploration;
        
        assertEquals(expectedUCB1, ucb1, 0.001);
    }
    
    @Test
    public void testUCB1WithZeroVisits() {
        // Create a parent with some visits
        GameTreeNode<Integer> parent = new GameTreeNode<>(0);
        parent.incrementVisits();
        
        // Create an unvisited child
        GameTreeNode<Integer> child = parent.addChild(1);
        
        // Calculate UCB1 - should be infinity for unvisited nodes
        double ucb1 = child.getUCB1(1.0);
        assertEquals(Double.POSITIVE_INFINITY, ucb1);
    }
    
    @Test
    public void testMultiLevelTree() {
        // Create a multi-level tree
        GameTreeNode<Integer> root = new GameTreeNode<>(0);
        
        GameTreeNode<Integer> child1 = root.addChild(1);
        GameTreeNode<Integer> child2 = root.addChild(2);
        
        GameTreeNode<Integer> grandchild1 = child1.addChild(3);
        GameTreeNode<Integer> grandchild2 = child1.addChild(4);
        
        GameTreeNode<Integer> grandchild3 = child2.addChild(5);
        GameTreeNode<Integer> grandchild4 = child2.addChild(6);
        
        // Test structure
        assertEquals(2, root.getChildren().size());
        assertSame(child1, root.getChildren().get(0));
        assertSame(child2, root.getChildren().get(1));
        
        assertEquals(2, child1.getChildren().size());
        assertSame(grandchild1, child1.getChildren().get(0));
        assertSame(grandchild2, child1.getChildren().get(1));
        
        assertEquals(2, child2.getChildren().size());
        assertSame(grandchild3, child2.getChildren().get(0));
        assertSame(grandchild4, child2.getChildren().get(1));
        
        // Test parent references
        assertNull(root.getParent());
        assertSame(root, child1.getParent());
        assertSame(root, child2.getParent());
        assertSame(child1, grandchild1.getParent());
        assertSame(child1, grandchild2.getParent());
        assertSame(child2, grandchild3.getParent());
        assertSame(child2, grandchild4.getParent());
        
        // Test leaf status
        assertFalse(root.isLeaf());
        assertFalse(child1.isLeaf());
        assertFalse(child2.isLeaf());
        assertTrue(grandchild1.isLeaf());
        assertTrue(grandchild2.isLeaf());
        assertTrue(grandchild3.isLeaf());
        assertTrue(grandchild4.isLeaf());
        
        // Test data values
        assertEquals(0, root.getData());
        assertEquals(1, child1.getData());
        assertEquals(2, child2.getData());
        assertEquals(3, grandchild1.getData());
        assertEquals(4, grandchild2.getData());
        assertEquals(5, grandchild3.getData());
        assertEquals(6, grandchild4.getData());
    }
    
    @Test
    public void testSetScoreWithHierarchy() {
        // Create a small tree
        GameTreeNode<Integer> root = new GameTreeNode<>(0);
        GameTreeNode<Integer> child1 = root.addChild(1);
        GameTreeNode<Integer> child2 = root.addChild(2);
        
        // Set scores
        root.setScore(10.0);
        child1.setScore(20.0);
        child2.setScore(30.0);
        
        // Verify scores - each node's score should be independent
        assertEquals(10.0, root.getScore(), 0.001);
        assertEquals(20.0, child1.getScore(), 0.001);
        assertEquals(30.0, child2.getScore(), 0.001);
    }
    
    @Test
    public void testAddScoreWithHierarchy() {
        // Create a small tree
        GameTreeNode<Integer> root = new GameTreeNode<>(0);
        GameTreeNode<Integer> child1 = root.addChild(1);
        GameTreeNode<Integer> child2 = root.addChild(2);
        
        // Add scores
        root.addScore(10.0);
        child1.addScore(20.0);
        child2.addScore(30.0);
        
        // Add more scores
        root.addScore(5.0);
        child1.addScore(5.0);
        child2.addScore(5.0);
        
        // Verify scores
        assertEquals(15.0, root.getTotalScore(), 0.001);
        assertEquals(25.0, child1.getTotalScore(), 0.001);
        assertEquals(35.0, child2.getTotalScore(), 0.001);
    }
}