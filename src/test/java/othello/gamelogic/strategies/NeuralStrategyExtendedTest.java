package othello.gamelogic.strategies;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import deeplearningjava.api.Network;
import deeplearningjava.api.TensorNetwork;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.network.FeedForwardNetwork;
import deeplearningjava.network.DenseNetwork;
import deeplearningjava.factory.NetworkFactory;
import othello.gamelogic.*;

import java.util.Map;
import java.util.List;

/**
 * Extended tests for the NeuralStrategy class to improve code coverage.
 */
public class NeuralStrategyExtendedTest {
    
    private OthelloGame game;
    private Player blackPlayer;
    private Player whitePlayer;
    private GameState gameState;
    
    @BeforeEach
    public void setUp() {
        // Set up players
        blackPlayer = new HumanPlayer();
        blackPlayer.setColor(BoardSpace.SpaceType.BLACK);
        
        whitePlayer = new ComputerPlayer("minimax");
        whitePlayer.setColor(BoardSpace.SpaceType.WHITE);
        
        // Set up a fresh game state
        game = new OthelloGame(blackPlayer, whitePlayer);
        
        // Create a game state
        gameState = new GameState(game.getBoard(), blackPlayer, whitePlayer);
    }
    
    @Test
    public void testTensorNetworkStrategy() {
        // Create a tensor network with Othello dimensions
        int boardSize = 8;
        int channels = 1;
        
        // Create a dense network for Othello
        DenseNetwork network = DenseNetwork.createForBoardGame(
                new int[]{channels, boardSize, boardSize},
                new int[]{128, 64, 32},
                1,
                ActivationFunctions.relu(),
                ActivationFunctions.tanh(),
                false
        );
        
        // Create a tensor network wrapper
        TensorNetworkWrapper wrapper = new TensorNetworkWrapper(network, boardSize, channels);
        
        // Create a neural strategy using tensor network
        NeuralStrategy strategy = new NeuralStrategy(wrapper);
        
        // Verify that it uses tensor network
        assertTrue(strategy.usesTensorNetwork());
        assertNull(strategy.getNetwork());
        assertNotNull(strategy.getTensorNetwork());
        assertEquals(wrapper, strategy.getTensorNetwork());
        
        // Test making a move
        BoardSpace move = strategy.getBestMove(game, blackPlayer, whitePlayer);
        assertNotNull(move);
    }
    
    @Test
    public void testCreateWithTensorNetwork() {
        // Test the static factory method for tensor networks
        int boardSize = 8;
        int channels = 1;
        
        NeuralStrategy strategy = NeuralStrategy.createWithTensorNetwork(boardSize, channels);
        
        // Verify the strategy is properly created
        assertTrue(strategy.usesTensorNetwork());
        assertNull(strategy.getNetwork());
        assertNotNull(strategy.getTensorNetwork());
        
        // Test making a move
        BoardSpace move = strategy.getBestMove(game, blackPlayer, whitePlayer);
        assertNotNull(move);
    }
    
    @Test
    public void testNoAvailableMoves() {
        // Create a board state with no available moves
        // This is a simplified test case - in a real game this would require a specific board setup
        
        // Create a mock game where getAvailableMoves returns empty map
        OthelloGame mockGame = new OthelloGame(blackPlayer, whitePlayer) {
            @Override
            public Map<BoardSpace, List<BoardSpace>> getAvailableMoves(Player player) {
                return Map.of(); // Empty map, no available moves
            }
        };
        
        // Create strategy
        NeuralStrategy strategy = NeuralStrategy.createDefault();
        
        // Get best move should return null when no moves available
        BoardSpace move = strategy.getBestMove(mockGame, blackPlayer, whitePlayer);
        assertNull(move);
    }
    
    @Test
    public void testTrainMethod() {
        // Test the training method
        NeuralStrategy strategy = NeuralStrategy.createDefault();
        
        // Training method is just a placeholder, but we want to ensure it runs
        strategy.train(100);
        
        // No assertion needed, just making sure it doesn't throw an exception
        assertTrue(true);
    }
    
    @Test
    public void testEvaluateWithTensorNetwork() {
        // Create a tensor network with Othello dimensions
        int boardSize = 8;
        int channels = 1;
        
        // Create a dense network for Othello
        DenseNetwork network = DenseNetwork.createForBoardGame(
                new int[]{channels, boardSize, boardSize},
                new int[]{128, 64, 32},
                1,
                ActivationFunctions.relu(),
                ActivationFunctions.tanh(),
                false
        );
        
        // Create a tensor network wrapper
        TensorNetworkWrapper wrapper = new TensorNetworkWrapper(network, boardSize, channels);
        
        // Create a neural strategy using tensor network
        NeuralStrategy strategy = new NeuralStrategy(wrapper);
        
        // The evaluateWithTensorNetwork method is private, so we need to test it indirectly
        // We'll use getBestMove, which internally uses evaluateWithTensorNetwork
        BoardSpace move = strategy.getBestMove(game, blackPlayer, whitePlayer);
        
        // Assert that we got a valid move
        assertNotNull(move);
    }
}