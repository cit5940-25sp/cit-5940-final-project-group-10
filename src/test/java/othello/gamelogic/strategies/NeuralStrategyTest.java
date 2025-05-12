package othello.gamelogic.strategies;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import deeplearningjava.api.Network;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.network.FeedForwardNetwork;
import deeplearningjava.factory.NetworkFactory;
import othello.gamelogic.*;

import java.io.File;
import java.util.Map;
import java.util.List;

/**
 * Tests for the NeuralStrategy class.
 */
public class NeuralStrategyTest {
    
    private OthelloGame game;
    private Player blackPlayer;
    private Player whitePlayer;
    
    @BeforeEach
    public void setUp() {
        // Set up players
        blackPlayer = new HumanPlayer();
        blackPlayer.setColor(BoardSpace.SpaceType.BLACK);
        
        whitePlayer = new ComputerPlayer("minimax");
        whitePlayer.setColor(BoardSpace.SpaceType.WHITE);
        
        // Set up a fresh game state
        game = new OthelloGame(blackPlayer, whitePlayer);
    }
    
    @Test
    public void testCreateNetworkFromDefaults() {
        // Create a neural network with default parameters
        int[] layerSizes = {64 * 3, 128, 64, 32, 1};
        Network network = new FeedForwardNetwork(
            layerSizes,
            ActivationFunctions.relu(),
            ActivationFunctions.tanh(),
            false
        );
        
        // Wrap the network
        NetworkWrapper wrapper = new NetworkWrapper(network);
        
        // Create a neural strategy with this network
        NeuralStrategy strategy = new NeuralStrategy(wrapper);
        
        // Ensure the strategy has the network
        assertNotNull(strategy.getNetwork());
        assertEquals(wrapper, strategy.getNetwork());
        
        // Test that it can make a move
        BoardSpace move = strategy.getBestMove(game, blackPlayer, whitePlayer);
        
        // At the start of the game, there should be valid moves
        assertNotNull(move);
    }
    
    @Test
    public void testNetworkEvaluation() {
        // Create a simple network
        int[] layerSizes = {64 * 3, 16, 1};
        Network network = new FeedForwardNetwork(
            layerSizes,
            ActivationFunctions.relu(),
            ActivationFunctions.tanh(),
            false
        );
        
        // Wrap the network
        NetworkWrapper wrapper = new NetworkWrapper(network);
        
        // Create a neural strategy with this network
        NeuralStrategy strategy = new NeuralStrategy(wrapper);
        
        // Test evaluation by making a move
        BoardSpace move = strategy.getBestMove(game, blackPlayer, whitePlayer);
        
        // Check that a move is returned
        assertNotNull(move);
        
        // Apply the move
        BoardSpace[][] board = game.getBoard();
        board[move.getX()][move.getY()].setType(blackPlayer.getColor());
        
        // Make sure there's still a valid move after the first move
        BoardSpace nextMove = strategy.getBestMove(game, whitePlayer, blackPlayer);
        assertNotNull(nextMove);
    }
    
    @Test
    public void testNetworkFactory() {
        // Create a default neural network using the factory
        NetworkWrapper wrapper = new NetworkWrapper(NetworkFactory.createOthelloNetwork());
        
        // Create a neural strategy with this network
        NeuralStrategy strategy = new NeuralStrategy(wrapper);
        
        // Test that it can make a move
        BoardSpace move = strategy.getBestMove(game, blackPlayer, whitePlayer);
        
        // At the start of the game, there should be valid moves
        assertNotNull(move);
        
        // Test the factory's default network creation method
        int[] layerSizes = {2, 4, 1};
        NetworkWrapper customWrapper = new NetworkWrapper(
            NetworkFactory.createDenseNetwork(layerSizes, "sigmoid", "sigmoid", false)
        );
        
        // The network should be properly configured
        assertNotNull(customWrapper);
        
        // Test feedforward works
        double[] outputs = customWrapper.feedForward(new double[] {0.5, 0.5});
        assertNotNull(outputs);
        assertEquals(1, outputs.length);
    }
    
    @Test
    public void testDefaultStrategyCreation() {
        // Test the static factory method in NeuralStrategy
        NeuralStrategy strategy = NeuralStrategy.createDefault();
        
        // Ensure the strategy is created with a valid network
        assertNotNull(strategy);
        assertNotNull(strategy.getNetwork());
        
        // Test that it can make a move
        BoardSpace move = strategy.getBestMove(game, blackPlayer, whitePlayer);
        
        // At the start of the game, there should be valid moves
        assertNotNull(move);
    }
}