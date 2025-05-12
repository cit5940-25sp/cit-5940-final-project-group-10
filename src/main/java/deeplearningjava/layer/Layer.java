package deeplearningjava.layer;

import deeplearningjava.core.Node;
import java.util.List;

/**
 * Interface representing a layer in a neural network.
 * Each layer consists of nodes (neurons) and implements forward and backward propagation.
 */
public interface Layer extends deeplearningjava.api.Layer {
    /**
     * Get all nodes in this layer.
     * @return List of nodes that make up this layer
     */
    @Override
    List<Node> getNodes();
    
    /**
     * Get the size of the layer (number of nodes).
     * @return The number of nodes
     */
    @Override
    int getSize();
    
    /**
     * Connect this layer to the next layer, creating appropriate edges.
     * @param nextLayer The next layer to connect to
     */
    @Override
    void connectTo(deeplearningjava.api.Layer nextLayer);
    
    /**
     * Initialize weights for connections to the next layer.
     * @param nextLayerSize The size of the next layer
     */
    @Override
    void initializeWeights(int nextLayerSize);
    
    /**
     * Calculate outputs for all nodes in this layer.
     * Should use inputs from the previous layer and apply activations.
     */
    void calculateOutputs();
    
    /**
     * Set output values directly (typically used for input layer).
     * @param inputs The values to set as outputs
     */
    void setOutputs(double[] inputs);
    
    /**
     * Calculate output gradients for the layer (typically for output layer).
     * @param targetOutputs The expected/target outputs
     */
    void calculateOutputGradients(double[] targetOutputs);
    
    /**
     * Calculate hidden gradients for this layer during backpropagation.
     */
    void calculateHiddenGradients();
    
    /**
     * Update weights for this layer during backpropagation.
     * @param learningRate The learning rate to apply
     */
    void updateParameters(double learningRate);
    
    /**
     * Handle forward pass through this layer.
     * @param inputs Inputs to this layer (if applicable)
     * @return Outputs from this layer
     */
    @Override
    double[] forward(double[] inputs);
    
    /**
     * Handle backward pass through this layer.
     * @param nextLayerGradients Gradients from the next layer
     * @return Gradients to propagate to the previous layer
     */
    @Override
    double[] backward(double[] nextLayerGradients);
    
    /**
     * Checks if this layer is a specific type of layer.
     * @param layerType The layer type to check
     * @return True if this layer is of the specified type
     */
    boolean isLayerType(deeplearningjava.api.Layer.LayerType layerType);
}