package deeplearningjava.network;

import java.util.List;
import java.util.Objects;

import deeplearningjava.api.Layer;
import deeplearningjava.core.activation.ActivationFunction;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.layer.InputLayer;
import deeplearningjava.layer.OutputLayer;
import deeplearningjava.layer.StandardLayer;

/**
 * Implementation of a basic feedforward neural network.
 * A feedforward network passes information in one direction, from input to output.
 */
public class FeedForwardNetwork extends NeuralNetwork {
    
    /**
     * Creates an empty feedforward network.
     */
    public FeedForwardNetwork() {
        super();
    }
    
    /**
     * Creates a feedforward network with the specified layers.
     * @param layers The layers for this network
     */
    public FeedForwardNetwork(List<Layer> layers) {
        super(layers);
        validateLayerStructure();
    }
    
    /**
     * Creates a feedforward network with the specified architecture.
     * 
     * @param layerSizes Array of layer sizes (including input and output)
     * @param hiddenActivation Activation function for hidden layers
     * @param outputActivation Activation function for the output layer
     * @param useSoftmax Whether to use softmax in the output layer
     */
    public FeedForwardNetwork(int[] layerSizes, 
                             ActivationFunction hiddenActivation,
                             ActivationFunction outputActivation,
                             boolean useSoftmax) {
        super();
        
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("Network must have at least input and output layers");
        }
        
        Objects.requireNonNull(hiddenActivation, "hiddenActivation must not be null");
        Objects.requireNonNull(outputActivation, "outputActivation must not be null");
        
        // Create input layer
        addLayer(new InputLayer(layerSizes[0]));
        
        // Create hidden layers
        for (int i = 1; i < layerSizes.length - 1; i++) {
            addLayer(new StandardLayer(layerSizes[i], hiddenActivation));
        }
        
        // Create output layer
        addLayer(new OutputLayer(layerSizes[layerSizes.length - 1], 
                                outputActivation, useSoftmax));
    }
    
    /**
     * Creates a feedforward network with ReLU activation for hidden layers
     * and appropriate output activation based on the output size.
     * 
     * @param layerSizes Array of layer sizes (including input and output)
     * @return A configured feedforward network
     */
    public static FeedForwardNetwork createDefault(int[] layerSizes) {
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("Network must have at least input and output layers");
        }
        
        // Use appropriate output activation based on output size
        ActivationFunction outputActivation;
        boolean useSoftmax;
        
        if (layerSizes[layerSizes.length - 1] == 1) {
            // For regression or binary classification without softmax
            outputActivation = ActivationFunctions.tanh();
            useSoftmax = false;
        } else {
            // For multi-class classification
            outputActivation = ActivationFunctions.linear();
            useSoftmax = true;
        }
        
        return new FeedForwardNetwork(layerSizes, 
                                     ActivationFunctions.relu(),
                                     outputActivation, 
                                     useSoftmax);
    }
    
    /**
     * Validates that the layer structure is appropriate for a feedforward network.
     */
    private void validateLayerStructure() {
        if (layers.isEmpty()) {
            return; // Empty network is valid (for now)
        }
        
        // First layer should be an input layer
        if (layers.get(0).getType() != Layer.LayerType.INPUT) {
            throw new IllegalArgumentException("First layer must be an input layer");
        }
        
        // Last layer should be an output layer
        if (layers.get(layers.size() - 1).getType() != Layer.LayerType.OUTPUT) {
            throw new IllegalArgumentException("Last layer must be an output layer");
        }
    }
    
    @Override
    public double[] forward(double[] inputs) {
        if (layers.isEmpty()) {
            throw new IllegalStateException("Network has no layers");
        }
        
        if (inputs.length != layers.get(0).getSize()) {
            throw new IllegalArgumentException(
                String.format("Input size (%d) must match input layer size (%d)",
                             inputs.length, layers.get(0).getSize()));
        }
        
        // Forward through input layer
        double[] currentOutput = layers.get(0).forward(inputs);
        
        // Forward through remaining layers
        for (int i = 1; i < layers.size(); i++) {
            currentOutput = layers.get(i).forward(null); // Pass null to use connections
        }
        
        return currentOutput;
    }
    
    @Override
    public double[] train(double[] inputs, double[] targets) {
        // Forward pass
        double[] outputs = forward(inputs);
        
        // Backward pass (starting from output layer)
        double[] gradients = targets;
        for (int i = layers.size() - 1; i >= 0; i--) {
            gradients = layers.get(i).backward(gradients);
        }
        
        return outputs;
    }
    
    @Override
    public int getInputSize() {
        if (layers.isEmpty()) {
            throw new IllegalStateException("Network has no layers");
        }
        return layers.get(0).getSize();
    }
    
    @Override
    public int getOutputSize() {
        if (layers.isEmpty()) {
            throw new IllegalStateException("Network has no layers");
        }
        return layers.get(layers.size() - 1).getSize();
    }
}