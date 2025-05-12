package deeplearningjava.api;

import deeplearningjava.core.tensor.Tensor;

/**
 * Interface for neural network layers that operate on tensor data.
 * This is the foundation for convolutional neural network layers.
 */
public interface TensorLayer {
    
    /**
     * Gets the shape of the input tensor.
     * 
     * @return The input shape as an array of dimensions
     */
    int[] getInputShape();
    
    /**
     * Gets the shape of the output tensor.
     * 
     * @return The output shape as an array of dimensions
     */
    int[] getOutputShape();
    
    /**
     * Performs the forward pass through this layer.
     * 
     * @param input The input tensor
     * @return The output tensor
     */
    Tensor forward(Tensor input);
    
    /**
     * Performs the backward pass through this layer.
     * 
     * @param gradients The gradients from the next layer
     * @return The gradients to propagate to the previous layer
     */
    Tensor backward(Tensor gradients);
    
    /**
     * Connects this layer to the next layer.
     * 
     * @param nextLayer The next layer to connect to
     */
    void connectTo(TensorLayer nextLayer);
    
    /**
     * Initializes the parameters of this layer.
     */
    void initializeParameters();
    
    /**
     * Updates the parameters of this layer using the accumulated gradients.
     * 
     * @param learningRate The learning rate to use for the update
     */
    void updateParameters(double learningRate);
    
    /**
     * Gets the type of this layer.
     * 
     * @return The layer type
     */
    LayerType getType();
    
    /**
     * Enum for the layer types used in CNNs.
     */
    enum LayerType {
        CONVOLUTIONAL,
        POOLING,
        FLATTENING,
        FULLY_CONNECTED,
        BATCH_NORM,
        DROPOUT
    }
}