package graph.search;

import java.util.Comparator;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;
import java.util.function.Function;
import java.util.function.BiFunction;

/**
 * Implements the Monte Carlo Tree Search algorithm.
 */
public class MonteCarloTreeSearch {
    private static final Random random = new Random();
    
    /**
     * Performs Monte Carlo Tree Search on the game tree.
     * @param <T> The type of game state
     * @param root The root node of the game tree
     * @param simulations Number of simulations to run
     * @param explorationParam The exploration parameter for UCB1
     * @param simulator Function to simulate a random game from a state
     * @param evaluator Function to evaluate the result of a simulation
     */
    public static <T> void search(
            GameTreeNode<T> root,
            int simulations,
            double explorationParam,
            Function<T, T> simulator,
            BiFunction<T, Boolean, Double> evaluator) {
        
        for (int i = 0; i < simulations; i++) {
            // Selection
            GameTreeNode<T> selectedNode = select(root, explorationParam);
            
            // Expansion
            GameTreeNode<T> expandedNode = expand(selectedNode);
            
            // Simulation
            T simulatedResult = simulator.apply(expandedNode.getData());
            
            // Backpropagation
            backpropagate(expandedNode, simulatedResult, evaluator);
        }
    }
    
    /**
     * Selects a node to expand using UCB1.
     * @param <T> The type of game state
     * @param node The current node
     * @param explorationParam The exploration parameter for UCB1
     * @return The selected node
     */
    private static <T> GameTreeNode<T> select(GameTreeNode<T> node, double explorationParam) {
        while (!node.isLeaf()) {
            if (node.getChildren().stream().anyMatch(child -> ((GameTreeNode<T>)child).getVisits() == 0)) {
                // If there are unvisited children, select one of them
                List<GameTreeNode<T>> unvisitedChildren = new ArrayList<>();
                for (GameTreeNode<T> child : node.getChildren()) {
                    if (child.getVisits() == 0) {
                        unvisitedChildren.add(child);
                    }
                }
                return unvisitedChildren.get(random.nextInt(unvisitedChildren.size()));
            }
            
            // Select best child according to UCB1
            GameTreeNode<T> bestChild = null;
            double bestUCB1 = Double.NEGATIVE_INFINITY;
            
            for (GameTreeNode<T> child : node.getChildren()) {
                double ucb1 = child.getUCB1(explorationParam);
                if (ucb1 > bestUCB1) {
                    bestUCB1 = ucb1;
                    bestChild = child;
                }
            }
            
            if (bestChild != null) {
                node = bestChild;
            } else {
                break;
            }
        }
        
        return node;
    }
    
    /**
     * Expands the selected node by adding a child.
     * @param <T> The type of game state
     * @param node The node to expand
     * @return The newly created child node
     */
    private static <T> GameTreeNode<T> expand(GameTreeNode<T> node) {
        // In a real implementation, we would generate all possible moves
        // and add them as children. For this skeleton, we just add a dummy child.
        GameTreeNode<T> child = new GameTreeNode<>(node.getData());
        node.addChild(child);
        return child;
    }
    
    /**
     * Backpropagates the simulation result up the tree.
     * @param <T> The type of game state
     * @param node The leaf node that was expanded
     * @param result The result of the simulation
     * @param evaluator Function to evaluate the simulation result
     */
    private static <T> void backpropagate(
            GameTreeNode<T> node, 
            T result, 
            BiFunction<T, Boolean, Double> evaluator) {
        
        boolean isMaximizingPlayer = true; // Start with maximizing player at the leaf
        
        GameTreeNode<T> current = node;
        while (current != null) {
            current.incrementVisits();
            double score = evaluator.apply(result, isMaximizingPlayer);
            current.addScore(score);
            
            current = (GameTreeNode<T>) current.getParent();
            isMaximizingPlayer = !isMaximizingPlayer; // Alternate players
        }
    }
}