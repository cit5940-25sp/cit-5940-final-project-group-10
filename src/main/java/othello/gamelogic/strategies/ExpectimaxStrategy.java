package othello.gamelogic.strategies;

import graph.search.GameTreeNode;
import graph.search.Expectimax;
import othello.gamelogic.*;
import java.util.Map;
import java.util.List;

/**
 * Implements a strategy using the Expectimax algorithm.
 */
public class ExpectimaxStrategy implements Strategy {
    private final BoardEvaluator evaluator;
    private final int maxDepth;
    
    public ExpectimaxStrategy() {
        this.evaluator = new WeightedEvaluator();
        this.maxDepth = 3; // May need to be lower than minimax due to branching
    }
    
    @Override
    public BoardSpace getBestMove(OthelloGame game, Player currentPlayer, Player opponent) {
        // Get available moves
        Map<BoardSpace, List<BoardSpace>> availableMoves = game.getAvailableMoves(currentPlayer);
        
        if (availableMoves.isEmpty()) {
            return null; // No valid moves
        }
        
        // Create game state representation
        GameState initialState = new GameState(game.getBoard(), currentPlayer, opponent);
        
        // Create root node for the search tree
        GameTreeNode<GameState> rootNode = new GameTreeNode<>(initialState);
        
        // Create child nodes for each available move
        for (BoardSpace move : availableMoves.keySet()) {
            GameState childState = initialState.applyMove(move);
            GameTreeNode<GameState> childNode = new GameTreeNode<>(childState);
            rootNode.addChild(childNode);
            
            // Store the move in the node for retrieval later
            childNode.setData(new GameStateWithMove(childState, move));
        }
        
        // Use graph package's Expectimax implementation
        Expectimax.search(
            rootNode, 
            maxDepth, 
            true, // maximizing player
            (state, isMax) -> evaluator.evaluate(
                ((GameState)state).getBoard(), 
                isMax ? currentPlayer : opponent,
                isMax ? opponent : currentPlayer)
        );
        
        // Find child with the best score
        return getBestChildMove(rootNode);
    }
    
    /**
     * Helper class to store a move with a game state
     */
    private static class GameStateWithMove extends GameState {
        private final BoardSpace move;
        
        public GameStateWithMove(GameState state, BoardSpace move) {
            super(state.getBoard(), state.getCurrentPlayer(), state.getOpponent());
            this.move = move;
        }
        
        public BoardSpace getMove() {
            return move;
        }
    }
    
    /**
     * Gets the move from the best child of the root
     */
    private BoardSpace getBestChildMove(GameTreeNode<GameState> root) {
        if (root.getChildren().isEmpty()) {
            return null;
        }
        
        GameTreeNode<GameState> bestChild = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        
        for (GameTreeNode<GameState> child : root.getChildren()) {
            if (child.getScore() > bestScore) {
                bestScore = child.getScore();
                bestChild = child;
            }
        }
        
        if (bestChild == null) {
            return null;
        }
        
        // Extract the move from the best child
        if (bestChild.getData() instanceof GameStateWithMove) {
            return ((GameStateWithMove) bestChild.getData()).getMove();
        }
        
        return null;
    }
}