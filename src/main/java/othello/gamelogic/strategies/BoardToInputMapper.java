package othello.gamelogic.strategies;

import othello.gamelogic.BoardSpace;
import othello.gamelogic.OthelloGame;
import othello.gamelogic.Player;
import java.util.Map;
import java.util.List;

/**
 * Utility to convert Othello board states to neural network inputs.
 */
public class BoardToInputMapper {
    
    /**
     * Maps a board state to neural network input array.
     * @param board The board to convert
     * @param currentPlayer The current player
     * @return An array of input values for the neural network
     */
    public static double[] mapToInput(BoardSpace[][] board, Player currentPlayer) {
        // For Othello, we can represent the board as 3 channels:
        // 1. Current player's pieces (1 where present, 0 elsewhere)
        // 2. Opponent's pieces (1 where present, 0 elsewhere)
        // 3. Empty spaces (1 where empty, 0 elsewhere)
        
        int boardSize = OthelloGame.GAME_BOARD_SIZE;
        double[] input = new double[boardSize * boardSize * 3];
        
        BoardSpace.SpaceType currentPlayerColor = currentPlayer.getColor();
        BoardSpace.SpaceType opponentColor = (currentPlayerColor == BoardSpace.SpaceType.BLACK) 
            ? BoardSpace.SpaceType.WHITE 
            : BoardSpace.SpaceType.BLACK;
        
        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                BoardSpace.SpaceType spaceType = board[i][j].getType();
                
                // Calculate indices for each channel
                int currentPlayerIndex = i * boardSize + j;
                int opponentIndex = boardSize * boardSize + i * boardSize + j;
                int emptyIndex = 2 * boardSize * boardSize + i * boardSize + j;
                
                // Set values based on space type
                if (spaceType == currentPlayerColor) {
                    input[currentPlayerIndex] = 1.0;
                    input[opponentIndex] = 0.0;
                    input[emptyIndex] = 0.0;
                } else if (spaceType == opponentColor) {
                    input[currentPlayerIndex] = 0.0;
                    input[opponentIndex] = 1.0;
                    input[emptyIndex] = 0.0;
                } else { // Empty
                    input[currentPlayerIndex] = 0.0;
                    input[opponentIndex] = 0.0;
                    input[emptyIndex] = 1.0;
                }
            }
        }
        
        return input;
    }
    
    /**
     * Maps neural network output to a board space.
     * @param output The neural network output
     * @param availableMoves Map of available moves
     * @return The selected board space
     */
    public static BoardSpace mapToMove(double[] output, Map<BoardSpace, List<BoardSpace>> availableMoves) {
        // Assuming output is a probability distribution over all board positions
        int boardSize = OthelloGame.GAME_BOARD_SIZE;
        
        // Find the highest probability among available moves
        BoardSpace bestMove = null;
        double bestProb = -1;
        
        for (BoardSpace move : availableMoves.keySet()) {
            int index = move.getX() * boardSize + move.getY();
            
            if (index < output.length && output[index] > bestProb) {
                bestProb = output[index];
                bestMove = move;
            }
        }
        
        return bestMove;
    }
}