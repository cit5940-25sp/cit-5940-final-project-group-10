package graph.search;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import graph.core.TreeNode;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the GameTreeNode class.
 */
public class GameTreeNodeTest {
    
    private GameTreeNode<String> root;
    
    @BeforeEach
    public void setUp() {
        root = new GameTreeNode<>("root");
    }
    
    @Test
    public void testCreateNodeWithData() {
        assertEquals("root", root.getData());
        assertTrue(root.getChildren().isEmpty());
        assertTrue(root.isLeaf());
        assertNull(root.getParent());
    }
    
    @Test
    public void testSetData() {
        root.setData("new data");
        assertEquals("new data", root.getData());
    }
    
    @Test
    public void testAddChildNode() {
        GameTreeNode<String> child = new GameTreeNode<>("child");
        root.addChild(child);
        
        assertFalse(root.isLeaf());
        assertEquals(1, root.getChildren().size());
        assertEquals(root, child.getParent());
        assertEquals("child", root.getChildren().get(0).getData());
    }
    
    @Test
    public void testAddChildData() {
        GameTreeNode<String> child = root.addChild("child");
        
        assertFalse(root.isLeaf());
        assertEquals(1, root.getChildren().size());
        assertEquals(root, child.getParent());
        assertEquals("child", child.getData());
        assertEquals("child", root.getChildren().get(0).getData());
    }
    
    @Test
    public void testIsLeaf() {
        assertTrue(root.isLeaf());
        root.addChild("child");
        assertFalse(root.isLeaf());
    }
    
    @Test
    public void testSetAndGetScore() {
        root.setScore(42.0);
        assertEquals(42.0, root.getScore(), 0.001);
    }
    
    @Test
    public void testSetAndGetAlphaBeta() {
        assertEquals(Double.NEGATIVE_INFINITY, root.getAlpha(), 0.001);
        assertEquals(Double.POSITIVE_INFINITY, root.getBeta(), 0.001);
        
        root.setAlpha(10.0);
        root.setBeta(20.0);
        
        assertEquals(10.0, root.getAlpha(), 0.001);
        assertEquals(20.0, root.getBeta(), 0.001);
    }
    
    @Test
    public void testMCTSProperties() {
        // Test initial values
        assertEquals(0, root.getVisits());
        assertEquals(0.0, root.getTotalScore(), 0.001);
        
        // Test increment visits
        root.incrementVisits();
        root.incrementVisits();
        assertEquals(2, root.getVisits());
        
        // Test add score
        root.addScore(5.0);
        root.addScore(3.0);
        assertEquals(8.0, root.getTotalScore(), 0.001);
    }
    
    @Test
    public void testUCB1WithZeroVisits() {
        // Without parent, this should not matter, but testing the infinity case
        double value = root.getUCB1(1.0);
        assertEquals(Double.POSITIVE_INFINITY, value);
    }
    
    @Test
    public void testUCB1WithParentAndVisits() {
        // Create parent with visits
        GameTreeNode<String> parent = new GameTreeNode<>("parent");
        parent.incrementVisits();
        parent.incrementVisits();
        
        // Create child and link to parent
        GameTreeNode<String> child = new GameTreeNode<>("child");
        parent.addChild(child);
        
        // Set visits and score for child
        child.incrementVisits();
        child.addScore(5.0);
        
        // Calculate expected UCB1
        double explorationParam = 1.0;
        double exploitation = 5.0 / 1.0;
        double exploration = explorationParam * Math.sqrt(Math.log(2) / 1.0);
        double expectedUCB1 = exploitation + exploration;
        
        double actualUCB1 = child.getUCB1(explorationParam);
        assertEquals(expectedUCB1, actualUCB1, 0.001);
    }
}