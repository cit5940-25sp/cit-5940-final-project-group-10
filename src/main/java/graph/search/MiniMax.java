package graph.search;

import java.util.function.BiFunction;

/**
 * Implements the Minimax algorithm with alpha-beta pruning for game tree search.
 */
public class MiniMax {
    
    /**
     * Performs a minimax search with alpha-beta pruning on the game tree.
     * @param <T> The type of game state
     * @param node The current node in the game tree
     * @param depth The maximum depth to search
     * @param maximizingPlayer Whether the current player is maximizing
     * @param alpha The alpha value for pruning
     * @param beta The beta value for pruning
     * @param evaluator Function to evaluate a leaf node or depth-limited node
     * @return The best score for the current player
     */
    public static <T> double search(
            GameTreeNode<T> node, 
            int depth, 
            boolean maximizingPlayer,
            double alpha,
            double beta,
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
                double score = search(child, depth - 1, false, alpha, beta, evaluator);
                maxScore = Math.max(maxScore, score);
                alpha = Math.max(alpha, score);
                
                if (beta <= alpha) {
                    break; // Beta cutoff
                }
            }
            
            node.setScore(maxScore);
            return maxScore;
        } else {
            double minScore = Double.POSITIVE_INFINITY;
            
            for (GameTreeNode<T> child : node.getChildren()) {
                double score = search(child, depth - 1, true, alpha, beta, evaluator);
                minScore = Math.min(minScore, score);
                beta = Math.min(beta, score);
                
                if (beta <= alpha) {
                    break; // Alpha cutoff
                }
            }
            
            node.setScore(minScore);
            return minScore;
        }
    }
}