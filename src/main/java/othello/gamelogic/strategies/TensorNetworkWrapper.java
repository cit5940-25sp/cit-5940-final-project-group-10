package othello.gamelogic.strategies;

import deeplearningjava.api.TensorNetwork;
import deeplearningjava.core.tensor.Tensor;
import deeplearningjava.network.DenseNetwork;
import othello.gamelogic.BoardSpace;

/**
 * Wrapper class for using tensor-based neural networks with the Othello game.
 * This class serves as an adapter between the TensorNetwork interface and
 * the existing Othello strategy interface.
 */
public class TensorNetworkWrapper {
    
    private final TensorNetwork network;
    private final int boardSize;
    private final int channels;
    
    /**
     * Creates a new wrapper for a tensor network.
     * 
     * @param network The tensor network to wrap
     * @param boardSize The size of the Othello board
     * @param channels The number of input channels for the network
     */
    public TensorNetworkWrapper(TensorNetwork network, int boardSize, int channels) {
        this.network = network;
        this.boardSize = boardSize;
        this.channels = channels;
        
        // Validate network dimensions
        int[] inputShape = network.getInputShape();
        if (inputShape.length != 3 || 
                inputShape[0] != channels || 
                inputShape[1] != boardSize || 
                inputShape[2] != boardSize) {
            throw new IllegalArgumentException(
                    "Network input shape must be [" + channels + ", " + 
                    boardSize + ", " + boardSize + "]");
        }
    }
    
    /**
     * Creates a new wrapper with a default tensor network for Othello.
     * 
     * @param boardSize The size of the Othello board
     * @param channels The number of input channels
     * @return A new tensor network wrapper
     */
    public static TensorNetworkWrapper createDefault(int boardSize, int channels) {
        DenseNetwork network = DenseNetwork.createForOthello(boardSize, channels);
        return new TensorNetworkWrapper(network, boardSize, channels);
    }
    
    /**
     * Evaluates a board state using the tensor network.
     * 
     * @param boardState The board state as a 2D array of BoardSpace objects
     * @return The evaluation score
     */
    public double evaluateBoard(BoardSpace[][] boardState) {
        // Convert board state to tensor
        Tensor input = boardToTensor(boardState);
        
        // Forward through network
        Tensor output = network.forward(input);
        
        // Return scalar output
        return output.get(0);
    }
    
    /**
     * Trains the network on a board state with a known evaluation.
     * 
     * @param boardState The board state as a 2D array of BoardSpace objects
     * @param evaluation The target evaluation score
     * @return The actual network output after training
     */
    public double trainOnBoard(BoardSpace[][] boardState, double evaluation) {
        // Convert board state to tensor
        Tensor input = boardToTensor(boardState);
        
        // Create target tensor
        Tensor target = new Tensor(new double[]{evaluation}, 1);
        
        // Train network
        Tensor output = network.train(input, target);
        
        // Return scalar output
        return output.get(0);
    }
    
    /**
     * Converts a 2D board state to a tensor suitable for the network.
     * 
     * @param boardState The board state as a 2D array of BoardSpace objects
     * @return A tensor representation of the board
     */
    private Tensor boardToTensor(BoardSpace[][] boardState) {
        if (boardState.length != boardSize || boardState[0].length != boardSize) {
            throw new IllegalArgumentException(
                    "Board state dimensions must be " + boardSize + "x" + boardSize);
        }
        
        // Initialize data array for the tensor
        double[] data = new double[channels * boardSize * boardSize];
        
        // For single-channel input (standard case)
        if (channels == 1) {
            int index = 0;
            for (int row = 0; row < boardSize; row++) {
                for (int col = 0; col < boardSize; col++) {
                    // Convert BoardSpace enum to integer value for the neural network
                    // EMPTY -> 0, BLACK -> 1, WHITE -> 2
                    int value = 0;
                    switch (boardState[row][col].getType()) {
                        case EMPTY -> value = 0;
                        case BLACK -> value = 1;
                        case WHITE -> value = 2;
                    }
                    data[index++] = value;
                }
            }
        } 
        // For multi-channel input (e.g., one channel per piece type)
        else {
            // Channel 0 for empty spaces, Channel 1 for BLACK, Channel 2 for WHITE
            for (int c = 0; c < channels; c++) {
                BoardSpace.SpaceType targetType;
                switch (c) {
                    case 0 -> targetType = BoardSpace.SpaceType.EMPTY;
                    case 1 -> targetType = BoardSpace.SpaceType.BLACK;
                    case 2 -> targetType = BoardSpace.SpaceType.WHITE;
                    default -> throw new IllegalStateException("Unexpected channel value: " + c);
                }
                
                for (int row = 0; row < boardSize; row++) {
                    for (int col = 0; col < boardSize; col++) {
                        int index = c * (boardSize * boardSize) + row * boardSize + col;
                        // Set 1.0 if this space contains the piece type for this channel, 0.0 otherwise
                        data[index] = (boardState[row][col].getType() == targetType) ? 1.0 : 0.0;
                    }
                }
            }
        }
        
        return new Tensor(data, channels, boardSize, boardSize);
    }
    
    /**
     * For testing purposes - allows passing a 2D integer array directly
     * 
     * @param boardState The board state as a 2D array of integers (0=empty, 1=black, 2=white)
     * @return The evaluation score
     */
    public double evaluateIntBoard(int[][] boardState) {
        if (boardState.length != boardSize || boardState[0].length != boardSize) {
            throw new IllegalArgumentException(
                    "Board state dimensions must be " + boardSize + "x" + boardSize);
        }
        
        // Initialize data array for the tensor
        double[] data = new double[channels * boardSize * boardSize];
        
        // For single-channel input (standard case)
        if (channels == 1) {
            int index = 0;
            for (int row = 0; row < boardSize; row++) {
                for (int col = 0; col < boardSize; col++) {
                    data[index++] = boardState[row][col];
                }
            }
        } 
        // For multi-channel input (e.g., one channel per piece type)
        else {
            for (int c = 0; c < channels; c++) {
                int pieceType = c; // Assuming piece types are 0, 1, 2, etc.
                
                for (int row = 0; row < boardSize; row++) {
                    for (int col = 0; col < boardSize; col++) {
                        int index = c * (boardSize * boardSize) + row * boardSize + col;
                        // Set 1.0 if this space contains the piece type for this channel, 0.0 otherwise
                        data[index] = (boardState[row][col] == pieceType) ? 1.0 : 0.0;
                    }
                }
            }
        }
        
        Tensor input = new Tensor(data, channels, boardSize, boardSize);
        Tensor output = network.forward(input);
        return output.get(0);
    }
    
    /**
     * Gets the underlying tensor network.
     * 
     * @return The tensor network
     */
    public TensorNetwork getNetwork() {
        return network;
    }
}