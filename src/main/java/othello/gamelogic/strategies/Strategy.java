package othello.gamelogic.strategies;

import othello.gamelogic.BoardSpace;
import othello.gamelogic.OthelloGame;
import othello.gamelogic.Player;

/**
 * Interface for Othello game strategies.
 */
public interface Strategy {
    /**
     * Gets the best move for the current player.
     * @param game The current game
     * @param currentPlayer The player whose turn it is
     * @param opponent The opposing player
     * @return The best move as a BoardSpace
     */
    BoardSpace getBestMove(OthelloGame game, Player currentPlayer, Player opponent);
}