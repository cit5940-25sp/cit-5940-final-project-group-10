package graph.search;

import java.util.function.BiFunction;

/**
 * Implements the Expectimax algorithm for game tree search.
 */
public class Expectimax {
    
    /**
     * Performs an expectimax search on the game tree.
     * @param <T> The type of game state
     * @param node The current node in the game tree
     * @param depth The maximum depth to search
     * @param maximizingPlayer Whether the current player is maximizing
     * @param evaluator Function to evaluate a leaf node or depth-limited node
     * @return The expected score for the current player
     */
    public static <T> double search(
            GameTreeNode<T> node, 
            int depth, 
            boolean maximizingPlayer,
            BiFunction<T, Boolean, Double> evaluator) {
        
        // If node is a leaf or depth limit reached, evaluate the node
        if (depth == 0 || node.isLeaf()) {
            double score = evaluator.apply(node.getData(), maximizingPlayer);
            node.setScore(score);
            return score;
        }
        
        if (maximizingPlayer) {
            double maxScore = Double.NEGATIVE_INFINITY;
            
            for (GameTreeNode<T> child : node.getChildren()) {
                double score = search(child, depth - 1, false, evaluator);
                maxScore = Math.max(maxScore, score);
            }
            
            node.setScore(maxScore);
            return maxScore;
        } else {
            // Chance node (expectation)
            double expectedScore = 0;
            int numChildren = node.getChildren().size();
            
            if (numChildren == 0) {
                return evaluator.apply(node.getData(), maximizingPlayer);
            }
            
            double probability = 1.0 / numChildren;
            
            for (GameTreeNode<T> child : node.getChildren()) {
                double score = search(child, depth - 1, true, evaluator);
                expectedScore += probability * score;
            }
            
            node.setScore(expectedScore);
            return expectedScore;
        }
    }
}