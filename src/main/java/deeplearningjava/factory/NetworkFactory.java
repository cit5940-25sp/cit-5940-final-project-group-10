package deeplearningjava.factory;

import deeplearningjava.api.Network;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.network.DenseNetwork;
import deeplearningjava.network.ConvolutionalNeuralNetwork;

/**
 * Factory class for creating neural network instances.
 * Provides pre-configured networks for common use cases.
 */
public class NetworkFactory {
    
    /**
     * Creates a DenseNetwork with the specified layer sizes and activations.
     * 
     * @param layerSizes Array of layer sizes (including input and output)
     * @param hiddenActivation Name of activation for hidden layers ("relu", "tanh", "sigmoid")
     * @param outputActivation Name of activation for output layer
     * @param useSoftmax Whether to use softmax in the output layer
     * @return The created network
     */
    public static Network createDenseNetwork(
            int[] layerSizes, 
            String hiddenActivation, 
            String outputActivation, 
            boolean useSoftmax) {
        
        // Map activation function names to implementations
        var hiddenFunc = getActivationByName(hiddenActivation);
        var outputFunc = getActivationByName(outputActivation);
        
        return new DenseNetwork(layerSizes, hiddenFunc, outputFunc, useSoftmax);
    }
    
    /**
     * Creates a default neural network configuration for Othello board evaluation.
     * This network is designed for evaluating board positions in the Othello game.
     * 
     * @return A neural network configured for Othello
     */
    public static Network createOthelloNetwork() {
        int inputSize = 8 * 8 * 3; // 8x8 board with 3 channels (empty, player, opponent)
        int[] layerSizes = {inputSize, 128, 64, 32, 1};
        
        return new DenseNetwork(
                layerSizes,
                ActivationFunctions.relu(),
                ActivationFunctions.tanh(),
                false  // Don't use softmax for regression output
        );
    }
    
    /**
     * Creates a convolutional neural network for image-like board evaluation.
     * This is useful for games like Othello where the board can be represented as an image.
     * 
     * @param inputShape Input shape array [batchSize, channels, height, width]
     * @param outputSize Number of outputs (typically 1 for evaluation)
     * @return A convolutional neural network
     */
    public static deeplearningjava.api.ConvolutionalNetwork createOthelloCNN(int[] inputShape, int outputSize) {
        return ConvolutionalNeuralNetwork.createSimpleImageClassifier(inputShape, outputSize);
    }
    
    /**
     * Gets an activation function by name.
     * 
     * @param name The activation function name ("relu", "tanh", "sigmoid", etc.)
     * @return The corresponding activation function
     */
    private static deeplearningjava.core.activation.ActivationFunction getActivationByName(String name) {
        return switch (name.toLowerCase()) {
            case "relu" -> ActivationFunctions.relu();
            case "tanh" -> ActivationFunctions.tanh();
            case "sigmoid" -> ActivationFunctions.sigmoid();
            case "leaky_relu" -> ActivationFunctions.leakyRelu(0.01);
            case "linear" -> ActivationFunctions.linear();
            default -> throw new IllegalArgumentException("Unknown activation function: " + name);
        };
    }
}