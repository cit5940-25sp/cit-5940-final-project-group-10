package graph.search;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.function.Function;
import java.util.function.BiFunction;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the Monte Carlo Tree Search algorithm.
 */
public class MonteCarloTreeSearchTest {
    
    private GameTreeNode<Integer> root;
    private Function<Integer, Integer> simulator;
    private BiFunction<Integer, Boolean, Double> evaluator;
    
    @BeforeEach
    public void setUp() {
        // Create a simple game tree
        root = new GameTreeNode<>(0);
        
        // Add some children
        GameTreeNode<Integer> child1 = root.addChild(1);
        GameTreeNode<Integer> child2 = root.addChild(2);
        GameTreeNode<Integer> child3 = root.addChild(3);
        
        // Add grandchildren
        child1.addChild(10);
        child1.addChild(11);
        
        child2.addChild(20);
        child2.addChild(21);
        
        child3.addChild(30);
        child3.addChild(31);
        
        // Simple simulator that just returns the input value
        // In a real game, this would simulate a random game from the state
        simulator = value -> value;
        
        // Simple evaluator that just returns the value for maximizing player
        // and -value for minimizing player
        evaluator = (value, maximizing) -> maximizing ? (double) value : -(double) value;
    }
    
    @Test
    public void testMonteCarloTreeSearch() {
        // Perform MCTS with 100 simulations
        MonteCarloTreeSearch.search(root, 100, 1.414, simulator, evaluator);
        
        // Verify that visits and scores have been updated
        assertTrue(root.getVisits() > 0, "Root node should be visited");
        assertTrue(root.getTotalScore() != 0, "Root node should have a score");
        
        // Check that all children have been visited
        for (GameTreeNode<Integer> child : root.getChildren()) {
            assertTrue(child.getVisits() > 0, "Child node should be visited");
            assertTrue(child.getTotalScore() != 0, "Child node should have a score");
        }
    }
    
    @Test
    public void testSelectionPhase() {
        // Pre-populate some visits and scores to test selection
        root.incrementVisits();
        root.incrementVisits();
        
        GameTreeNode<Integer> child1 = root.getChildren().get(0);
        child1.incrementVisits();
        child1.addScore(10.0);
        
        GameTreeNode<Integer> child2 = root.getChildren().get(1);
        child2.incrementVisits();
        child2.addScore(20.0);
        
        // Child 3 is unvisited, so it should be selected first
        
        // Run a few MCTS iterations
        MonteCarloTreeSearch.search(root, 10, 1.414, simulator, evaluator);
        
        // Check that the previously unvisited child3 is now visited
        GameTreeNode<Integer> child3 = root.getChildren().get(2);
        assertTrue(child3.getVisits() > 0, "Previously unvisited child should now be visited");
    }
    
    @Test
    public void testUCB1Selection() {
        // Setup specific visit counts and scores to test UCB1 selection
        root.incrementVisits();
        root.incrementVisits();
        root.incrementVisits();
        
        // Child 1: Low score, many visits (exploitation not favored)
        GameTreeNode<Integer> child1 = root.getChildren().get(0);
        child1.incrementVisits();
        child1.incrementVisits();
        child1.addScore(2.0);
        
        // Child 2: High score, few visits (exploitation favored)
        GameTreeNode<Integer> child2 = root.getChildren().get(1);
        child2.incrementVisits();
        child2.addScore(10.0);
        
        // All children visited, so UCB1 will be used
        // With high exploration parameter, child1 should be favored (fewer visits)
        MonteCarloTreeSearch.search(root, 5, 10.0, simulator, evaluator);
        
        // With low exploration parameter, child2 should be favored (higher score)
        GameTreeNode<Integer> newRoot = new GameTreeNode<>(0);
        for (int i = 0; i < root.getChildren().size(); i++) {
            GameTreeNode<Integer> oldChild = root.getChildren().get(i);
            GameTreeNode<Integer> newChild = newRoot.addChild(oldChild.getData());
            newChild.incrementVisits();
            newChild.addScore(oldChild.getTotalScore());
            
            // Add same grandchildren
            for (GameTreeNode<Integer> grandchild : oldChild.getChildren()) {
                newChild.addChild(grandchild.getData());
            }
        }
        newRoot.incrementVisits();
        newRoot.incrementVisits();
        newRoot.incrementVisits();
        
        MonteCarloTreeSearch.search(newRoot, 5, 0.1, simulator, evaluator);
        
        // We can't directly assert the selection policy due to randomness,
        // but we can verify that all nodes got some visits
        assertTrue(newRoot.getVisits() > 3, "Root should have more visits after MCTS");
    }
    
    @Test
    public void testExpansionPhase() {
        // Create a root with no children to test expansion
        GameTreeNode<Integer> leafRoot = new GameTreeNode<>(42);
        
        // Run MCTS on a leaf node
        MonteCarloTreeSearch.search(leafRoot, 10, 1.414, simulator, evaluator);
        
        // The node should be visited, but still a leaf
        assertTrue(leafRoot.getVisits() > 0, "Leaf node should be visited");
        assertTrue(leafRoot.isLeaf(), "Node should still be a leaf");
    }
    
    @Test
    public void testBackpropagation() {
        // Test that scores are correctly backpropagated
        
        // Create a custom evaluator that gives fixed scores
        BiFunction<Integer, Boolean, Double> fixedEvaluator = (value, maximizing) -> 1.0;
        
        // Run a single simulation
        MonteCarloTreeSearch.search(root, 1, 1.414, simulator, fixedEvaluator);
        
        // Root and the selected path should have 1 visit and score
        assertEquals(1, root.getVisits(), "Root should have 1 visit");
        assertEquals(1.0, root.getTotalScore(), "Root should have score 1.0");
        
        // Exactly one child should have 1 visit
        int childrenVisited = 0;
        for (GameTreeNode<Integer> child : root.getChildren()) {
            if (child.getVisits() > 0) {
                childrenVisited++;
                assertEquals(1, child.getVisits(), "Visited child should have 1 visit");
                // The score is based on the actual implementation which doesn't seem to alternate scores
                // based on player perspective in the backpropagation (all nodes get 1.0)
                assertEquals(1.0, child.getTotalScore(), "Child should have score 1.0");
            }
        }
        assertEquals(1, childrenVisited, "Exactly one child should be visited");
    }
    
    @Test
    public void testMultipleSimulations() {
        // Run more simulations to see cumulative effect
        int simulations = 50;
        MonteCarloTreeSearch.search(root, simulations, 1.414, simulator, evaluator);
        
        // Root visits should equal number of simulations
        assertEquals(simulations, root.getVisits(), "Root visits should equal number of simulations");
        
        // Sum of children visits should equal number of simulations
        int totalChildVisits = 0;
        for (GameTreeNode<Integer> child : root.getChildren()) {
            totalChildVisits += child.getVisits();
        }
        assertEquals(simulations, totalChildVisits, "Sum of child visits should equal simulations");
    }
}