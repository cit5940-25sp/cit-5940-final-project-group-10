package othello.gamelogic.strategies;

import othello.gamelogic.BoardSpace;
import othello.gamelogic.Player;

/**
 * Interface for evaluating Othello board positions.
 */
public interface BoardEvaluator {
    /**
     * Evaluates a board position from a player's perspective.
     * @param board The board to evaluate
     * @param player The player from whose perspective to evaluate
     * @param opponent The opposing player
     * @return A score, with higher values being better for the player
     */
    double evaluate(BoardSpace[][] board, Player player, Player opponent);
}