package othello.gamelogic;

import java.util.List;
import java.util.Map;

/**
 * Represents the state of an Othello game for use in search algorithms.
 */
public class GameState {
    private BoardSpace[][] board;
    private Player currentPlayer;
    private Player opponent;
    
    /**
     * Creates a new game state
     * @param board The current board configuration
     * @param currentPlayer The player whose turn it is
     * @param opponent The opposing player
     */
    public GameState(BoardSpace[][] board, Player currentPlayer, Player opponent) {
        // Deep copy the board
        this.board = new BoardSpace[OthelloGame.GAME_BOARD_SIZE][OthelloGame.GAME_BOARD_SIZE];
        for (int i = 0; i < OthelloGame.GAME_BOARD_SIZE; i++) {
            for (int j = 0; j < OthelloGame.GAME_BOARD_SIZE; j++) {
                this.board[i][j] = new BoardSpace(board[i][j]);
            }
        }
        
        this.currentPlayer = currentPlayer;
        this.opponent = opponent;
    }
    
    /**
     * Gets the board configuration
     * @return The current board
     */
    public BoardSpace[][] getBoard() {
        return board;
    }
    
    /**
     * Gets the current player
     * @return The current player
     */
    public Player getCurrentPlayer() {
        return currentPlayer;
    }
    
    /**
     * Gets the opponent player
     * @return The opponent player
     */
    public Player getOpponent() {
        return opponent;
    }
    
    /**
     * Creates a new game state with the roles of current player and opponent swapped
     * @return A new game state with players swapped
     */
    public GameState swapPlayers() {
        return new GameState(board, opponent, currentPlayer);
    }
    
    /**
     * Applies a move to the game state and returns the resulting state
     * @param move The move to apply
     * @return The resulting game state
     */
    public GameState applyMove(BoardSpace move) {
        // Create a new state
        GameState newState = new GameState(board, currentPlayer, opponent);
        
        // Get available moves for this state
        Map<BoardSpace, List<BoardSpace>> availableMoves = currentPlayer.getAvailableMoves(board);
        
        // Find the move in the available moves
        BoardSpace correspondingMove = null;
        for (BoardSpace destination : availableMoves.keySet()) { // this is broken???
            if (destination.getX() == move.getX() && destination.getY() == move.getY()) {
                correspondingMove = destination;
                break;
            }
        }
        
        if (correspondingMove == null) {
            // Invalid move, return current state
            return this;
        }
        
        // Apply the move
        // Note: In a real implementation, we would create a simplified version of takeSpaces
        // that works with our copied board rather than the actual game
        
        // For this skeleton, we'll leave this method incomplete
        
        // Swap players for the next turn
        return newState.swapPlayers();
    }
    
    /**
     * Checks if the game is over (no valid moves for either player)
     * @return true if the game is over, false otherwise
     */
    public boolean isGameOver() {
        Map<BoardSpace, List<BoardSpace>> currentPlayerMoves = currentPlayer.getAvailableMoves(board);
        
        if (!currentPlayerMoves.isEmpty()) {
            return false;
        }
        
        // If current player has no moves, check opponent
        GameState swappedState = swapPlayers();
        Map<BoardSpace, List<BoardSpace>> opponentMoves = swappedState.getCurrentPlayer().getAvailableMoves(swappedState.getBoard());
        
        return opponentMoves.isEmpty();
    }
}