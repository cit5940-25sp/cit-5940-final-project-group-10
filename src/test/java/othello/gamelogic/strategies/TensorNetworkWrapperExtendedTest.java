package othello.gamelogic.strategies;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import deeplearningjava.network.DenseNetwork;
import deeplearningjava.api.TensorNetwork;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.core.tensor.Tensor;
import othello.gamelogic.BoardSpace;

/**
 * Extended tests for the TensorNetworkWrapper class to improve code coverage.
 */
public class TensorNetworkWrapperExtendedTest {
    
    private TensorNetworkWrapper wrapper;
    private BoardSpace[][] boardState;
    private int boardSize = 8;
    private int channels = 1;
    
    @BeforeEach
    public void setUp() {
        // Create a tensor network wrapper
        wrapper = TensorNetworkWrapper.createDefault(boardSize, channels);
        
        // Create a sample board state
        boardState = new BoardSpace[boardSize][boardSize];
        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                boardState[i][j] = new BoardSpace(i, j, BoardSpace.SpaceType.EMPTY);
            }
        }
        
        // Set up the initial Othello position
        boardState[3][3].setType(BoardSpace.SpaceType.WHITE);
        boardState[3][4].setType(BoardSpace.SpaceType.BLACK);
        boardState[4][3].setType(BoardSpace.SpaceType.BLACK);
        boardState[4][4].setType(BoardSpace.SpaceType.WHITE);
    }
    
    @Test
    public void testBoardToTensor() {
        // Test that we can convert a board to a tensor
        // We can't directly call boardToTensor as it's private, but we can test it indirectly
        // through evaluateBoard which uses it
        try {
            wrapper.evaluateBoard(boardState);
            // If we reach here without exception, the test passes
            assertTrue(true);
        } catch (Exception e) {
            fail("Exception thrown: " + e.getMessage());
        }
    }
    
    @Test
    public void testTrainOnBoardWithBoardSpace() {
        // Test the trainOnBoard method with BoardSpace array
        // Create a target evaluation
        double targetEval = 0.5;
        
        try {
            double actualOutput = wrapper.trainOnBoard(boardState, targetEval);
            // The actual value doesn't matter, just ensure it runs
            assertTrue(actualOutput >= -1.0 && actualOutput <= 1.0);
        } catch (Exception e) {
            fail("Exception thrown: " + e.getMessage());
        }
    }
    
    @Test
    public void testInvalidBoardStateSize() {
        // Create a board state with wrong dimensions
        BoardSpace[][] invalidBoard = new BoardSpace[6][6];
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                invalidBoard[i][j] = new BoardSpace(i, j, BoardSpace.SpaceType.EMPTY);
            }
        }
        
        // Should throw an exception
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            wrapper.evaluateBoard(invalidBoard);
        });
        
        assertTrue(exception.getMessage().contains("Board state dimensions"));
    }
    
    @Test
    public void testMultiChannelBoardToTensor() {
        // Create a wrapper with multiple channels
        int multiChannels = 3;
        
        // Create a tensor network for multi-channel input
        DenseNetwork network = DenseNetwork.createForBoardGame(
                new int[]{multiChannels, boardSize, boardSize},
                new int[]{64, 32},
                1,
                ActivationFunctions.relu(),
                ActivationFunctions.tanh(),
                false
        );
        
        TensorNetworkWrapper multiChannelWrapper = new TensorNetworkWrapper(network, boardSize, multiChannels);
        
        try {
            // This implicitly tests the multi-channel conversion in boardToTensor
            multiChannelWrapper.evaluateBoard(boardState);
            // If we reach here without exception, the test passes
            assertTrue(true);
        } catch (Exception e) {
            fail("Exception thrown: " + e.getMessage());
        }
    }
    
    @Test
    public void testConstructorWithInvalidDimensions() {
        // Create a network with mismatched dimensions
        DenseNetwork network = DenseNetwork.createForBoardGame(
                new int[]{2, 6, 6}, // Incorrect dimensions
                new int[]{64, 32},
                1,
                ActivationFunctions.relu(),
                ActivationFunctions.tanh(),
                false
        );
        
        // Should throw an exception
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            new TensorNetworkWrapper(network, boardSize, channels);
        });
        
        assertTrue(exception.getMessage().contains("Network input shape"));
    }
    
    @Test
    public void testGetNetwork() {
        TensorNetwork network = wrapper.getNetwork();
        assertNotNull(network);
        assertArrayEquals(new int[]{channels, boardSize, boardSize}, network.getInputShape());
    }
}