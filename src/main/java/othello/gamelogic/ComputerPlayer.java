package othello.gamelogic;

import othello.gamelogic.strategies.Strategy;
import othello.gamelogic.strategies.StrategyFactory;

/**
 * Represents a computer player that will make decisions autonomously during their turns.
 * Employs a specific computer strategy passed in through program arguments.
 */
public class ComputerPlayer extends Player {
    private final Strategy strategy;
    
    public ComputerPlayer(String strategyName) {
        // Use the strategyName input to create a specific strategy
        this.strategy = StrategyFactory.createStrategy(strategyName);
    }
    
    /**
     * Gets the best move for this computer player.
     * @param game The current game
     * @param opponent The opposing player
     * @return The board space representing the best move
     */
    public BoardSpace getBestMove(OthelloGame game, Player opponent) {
        return strategy.getBestMove(game, this, opponent);
    }
}