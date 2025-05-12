package othello.gamelogic.strategies;

import deeplearningjava.factory.NetworkFactory;
import deeplearningjava.factory.OnnxNetworkLoader;
import deeplearningjava.api.Network;
import deeplearningjava.api.TensorNetwork;
import othello.Constants;

import java.io.IOException;

/**
 * Factory for creating strategy instances.
 */
public class StrategyFactory {
    // Static field to store custom ONNX model path
    private static String customOnnxModelPath = null;
    
    /**
     * Creates a strategy based on the strategy name.
     * @param strategyName The name of the strategy
     * @return The corresponding strategy instance
     * @throws IllegalArgumentException if the strategy name is unknown
     */
    public static Strategy createStrategy(String strategyName) {
        return switch(strategyName) {
            case "minimax" -> new MinimaxStrategy();
            case "expectimax" -> new ExpectimaxStrategy();
            case "mcts" -> new MCTSStrategy();
            case "custom" -> new NeuralStrategy(createDefaultNetwork());
            case "tensor" -> createTensorNetworkStrategy();
            case "onnx" -> createOnnxNetworkStrategy();
            default -> throw new IllegalArgumentException("Unknown strategy: " + strategyName);
        };
    }
    
    /**
     * Sets a custom path for the ONNX model.
     * 
     * @param modelPath The path to the custom ONNX model
     */
    public static void setCustomOnnxModelPath(String modelPath) {
        customOnnxModelPath = modelPath;
        System.out.println("Custom ONNX model path set to: " + modelPath);
    }
    
    /**
     * Creates a default neural network for board evaluation.
     * This network is used when "custom" strategy is selected.
     * 
     * @return A neural network wrapper configured for Othello board evaluation
     */
    private static NetworkWrapper createDefaultNetwork() {
        // Create a neural network specifically designed for Othello evaluation
        return new NetworkWrapper(NetworkFactory.createOthelloNetwork());
    }
    
    /**
     * Creates a neural strategy using a tensor network for board evaluation.
     * This strategy uses 2D board input rather than flattened vector input.
     * 
     * @return A neural strategy using tensor network
     */
    private static NeuralStrategy createTensorNetworkStrategy() {
        int boardSize = 8; // Standard Othello board size
        int channels = 1; // Use 1 channel for simplicity, could be increased for more features
        
        return NeuralStrategy.createWithTensorNetwork(boardSize, channels);
    }
    
    /**
     * Creates a neural strategy using a pre-trained ONNX model.
     * This strategy loads the network architecture and weights from an ONNX file.
     * If a custom model path has been set, it will be used instead of the default.
     * 
     * @return A neural strategy using a pre-trained ONNX model
     */
    public static NeuralStrategy createOnnxNetworkStrategy() {
        try {
            // Get the path to the ONNX model (custom or default)
            String modelPath = (customOnnxModelPath != null) 
                ? customOnnxModelPath 
                : OnnxNetworkLoader.getDefaultOthelloModelPath();
            
            System.out.println("Creating ONNX neural strategy with model: " + modelPath);
            
            // Standard Othello parameters
            int boardSize = 8;
            int channels = 1; // Adjust based on the expected input for your model
            
            System.out.println("Loading ONNX network with boardSize=" + boardSize + ", channels=" + channels);
            
            // Load a tensor network from the ONNX model
            TensorNetwork network = OnnxNetworkLoader.loadOthelloNetwork(modelPath, boardSize, channels);
            
            System.out.println("Successfully loaded ONNX network: " + network.getClass().getSimpleName());
            System.out.println("Network layers: " + network.getLayerCount());
            System.out.println("Input shape: " + java.util.Arrays.toString(network.getInputShape()));
            System.out.println("Output shape: " + java.util.Arrays.toString(network.getOutputShape()));
            
            // Create a wrapper for the tensor network
            TensorNetworkWrapper wrapper = new TensorNetworkWrapper(network, boardSize, channels);
            
            // Create a neural strategy with the tensor network
            return new NeuralStrategy(wrapper);
        } catch (IOException e) {
            System.err.println("Failed to load ONNX model: " + e.getMessage());
            e.printStackTrace();
            
            // Fallback to the default tensor network strategy
            return createTensorNetworkStrategy();
        }
    }
    
    /**
     * Creates a neural strategy using a specific ONNX model file.
     * 
     * @param modelPath The path to the ONNX model file
     * @return A neural strategy using the specified ONNX model
     */
    public static NeuralStrategy createOnnxNetworkStrategy(String modelPath) {
        try {
            System.out.println("Creating ONNX neural strategy with specific model: " + modelPath);
            
            // Standard Othello parameters
            int boardSize = 8;
            int channels = 1; // Adjust based on the expected input for your model
            
            System.out.println("Loading ONNX network with boardSize=" + boardSize + ", channels=" + channels);
            
            // Load a tensor network from the ONNX model
            TensorNetwork network = OnnxNetworkLoader.loadOthelloNetwork(modelPath, boardSize, channels);
            
            System.out.println("Successfully loaded ONNX network: " + network.getClass().getSimpleName());
            System.out.println("Network layers: " + network.getLayerCount());
            System.out.println("Input shape: " + java.util.Arrays.toString(network.getInputShape()));
            System.out.println("Output shape: " + java.util.Arrays.toString(network.getOutputShape()));
            
            // Create a wrapper for the tensor network
            TensorNetworkWrapper wrapper = new TensorNetworkWrapper(network, boardSize, channels);
            
            // Create a neural strategy with the tensor network
            return new NeuralStrategy(wrapper);
        } catch (IOException e) {
            System.err.println("Failed to load ONNX model from path " + modelPath + ": " + e.getMessage());
            e.printStackTrace();
            
            // Fallback to the default tensor network strategy
            return createTensorNetworkStrategy();
        }
    }
}