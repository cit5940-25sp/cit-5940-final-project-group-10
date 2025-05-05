package othello.gamelogic.strategies;

import graph.search.GameTreeNode;
import graph.search.MonteCarloTreeSearch;
import othello.Constants;
import othello.gamelogic.*;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;

/**
 * Implements a strategy using Monte Carlo Tree Search.
 */
public class MCTSStrategy implements Strategy {
    private final double explorationParameter;
    private final int simulationCount;
    private final Random random = new Random();
    
    public MCTSStrategy() {
        this.explorationParameter = Constants.EXPLORATION_PARAM;
        this.simulationCount = 1000; // Number of simulations to run
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
        
        // Use graph package's MCTS implementation
        MonteCarloTreeSearch.search(
            rootNode,
            simulationCount,
            explorationParameter,
            this::simulateRandomGame,  // Game simulation function
            this::getGameResult        // Result evaluation function
        );
        
        // Find child with the highest visit count
        return getMostVisitedChildMove(rootNode);
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
     * Simulates a random game from the given state
     */
    private GameState simulateRandomGame(GameState state) {
        GameState currentState = state;
        int moveLimit = 100; // Prevent infinite loops
        
        while (!currentState.isGameOver() && moveLimit > 0) {
            // Get available moves
            Map<BoardSpace, List<BoardSpace>> availableMoves = 
                currentState.getCurrentPlayer().getAvailableMoves(currentState.getBoard());
            
            if (availableMoves.isEmpty()) {
                // No moves, swap players and continue
                currentState = currentState.swapPlayers();
                continue;
            }
            
            // Choose a random move
            List<BoardSpace> moves = new ArrayList<>(availableMoves.keySet());
            if (!moves.isEmpty()) {
                BoardSpace randomMove = moves.get(random.nextInt(moves.size()));
                // Apply the move
                currentState = currentState.applyMove(randomMove);
            }
            moveLimit--;
        }
        
        return currentState;
    }
    
    /**
     * Evaluates the result of a simulation
     */
    private double getGameResult(GameState finalState, boolean isMaximizingPlayer) {
        // Count pieces for both players
        BoardSpace[][] board = finalState.getBoard();
        int playerOneCount = 0;
        int playerTwoCount = 0;
        
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j].getType() == BoardSpace.SpaceType.BLACK) {
                    playerOneCount++;
                } else if (board[i][j].getType() == BoardSpace.SpaceType.WHITE) {
                    playerTwoCount++;
                }
            }
        }
        
        // Determine if maximizing player won
        boolean playerOneWon = playerOneCount > playerTwoCount;
        Player currentPlayer = finalState.getCurrentPlayer();
        Player opponent = finalState.getOpponent();
        boolean isMaximizingPlayerOne = isMaximizingPlayer && 
            currentPlayer.getColor() == BoardSpace.SpaceType.BLACK ||
            !isMaximizingPlayer && opponent.getColor() == BoardSpace.SpaceType.BLACK;
        
        boolean maximizingPlayerWon = (isMaximizingPlayerOne && playerOneWon) || 
            (!isMaximizingPlayerOne && !playerOneWon);
        
        return maximizingPlayerWon ? 1.0 : 0.0;
    }
    
    /**
     * Gets the move from the most visited child of the root
     */
    private BoardSpace getMostVisitedChildMove(GameTreeNode<GameState> root) {
        if (root.getChildren().isEmpty()) {
            return null;
        }
        
        GameTreeNode<GameState> bestChild = null;
        int mostVisits = -1;
        
        for (GameTreeNode<GameState> child : root.getChildren()) {
            if (child.getVisits() > mostVisits) {
                mostVisits = child.getVisits();
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