package othello.gamelogic.strategies;

import deeplearningjava.Network;
import deeplearningjava.Node;
import deeplearningjava.Layer;

/**
 * Factory for creating strategy instances.
 */
public class StrategyFactory {
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
            default -> throw new IllegalArgumentException("Unknown strategy: " + strategyName);
        };
    }
    
    /**
     * Creates a default neural network for board evaluation.
     * This network is used when "custom" strategy is selected.
     * 
     * @return A neural network configured for Othello board evaluation
     */
    private static Network createDefaultNetwork() {
////        // Define network architecture for Othello
////        // Input: 8x8x3 (board dimensions + 3 channels for player pieces, opponent pieces, empty)
////        // Hidden layers: 128, 64, 32 neurons
////        // Output: 1 (evaluation score)
////
////        int inputSize = 8 * 8 * 3; // 8x8 board with 3 channels
////        int[] hiddenLayers = {128, 64, 32};
////        int outputSize = 1;
////
////        Network network = new Network();
////
////        // Input layer
////        Layer inputLayer = new Layer();
////        for (int i = 0; i < inputSize; i++) {
////            inputLayer.addNode(new Node(Node.RELU, Node.RELU_DERIVATIVE));
////        }
////        network.addLayer(inputLayer);
////
////        // Hidden layers
////        for (int size : hiddenLayers) {
////            Layer hiddenLayer = new Layer();
////            for (int i = 0; i < size; i++) {
////                hiddenLayer.addNode(new Node(Node.RELU, Node.RELU_DERIVATIVE));
////            }
////            network.addLayer(hiddenLayer);
////        }
////
////        // Output layer
////        Layer outputLayer = new Layer();
////        for (int i = 0; i < outputSize; i++) {
////            outputLayer.addNode(new Node(Node.TANH, Node.TANH_DERIVATIVE)); // Tanh for [-1, 1] range score
////        }
////        network.addLayer(outputLayer);
////
////        // Initialize connections
////        network.connectLayers();
////
//        return network;
    return null;
    }
}