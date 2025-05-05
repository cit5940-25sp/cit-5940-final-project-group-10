package othello.gamelogic.strategies;

import othello.Constants;
import othello.gamelogic.BoardSpace;
import othello.gamelogic.Player;

/**
 * Evaluates board positions based on position weights.
 */
public class WeightedEvaluator implements BoardEvaluator {
    
    @Override
    public double evaluate(BoardSpace[][] board, Player player, Player opponent) {
        double score = 0;
        
        // Calculate weighted score for player and opponent
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j].getType() == player.getColor()) {
                    score += Constants.BOARD_WEIGHTS[i][j];
                } else if (board[i][j].getType() == opponent.getColor()) {
                    score -= Constants.BOARD_WEIGHTS[i][j];
                }
            }
        }
        
        return score;
    }
}