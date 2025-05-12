package deeplearningjava.api;

import java.util.List;
import deeplearningjava.core.tensor.Tensor;

/**
 * Interface for convolutional neural networks that operate on tensor data.
 * A convolutional neural network processes multi-dimensional data using
 * spatial operations like convolution and pooling.
 */
public interface ConvolutionalNetwork extends BaseNetwork {
    
    /**
     * Performs forward propagation through the network.
     * 
     * @param input The input tensor
     * @return The output tensor
     */
    Tensor forward(Tensor input);
    
    /**
     * Gets all tensor layers in the network.
     * 
     * @return A list of the network's tensor layers
     */
    List<TensorLayer> getTensorLayers();
    
    /**
     * Adds a tensor layer to the network.
     * 
     * @param layer The tensor layer to add
     */
    void addTensorLayer(TensorLayer layer);
    
    /**
     * Gets the input shape expected by the network.
     * 
     * @return The input shape as an array of dimensions
     */
    int[] getInputShape();
    
    /**
     * Gets the output shape produced by the network.
     * 
     * @return The output shape as an array of dimensions
     */
    int[] getOutputShape();
    
    /**
     * Trains the network with the given input and target tensors.
     * 
     * @param input The input tensor
     * @param target The target tensor
     * @return The output tensor after forward propagation
     */
    Tensor train(Tensor input, Tensor target);
    
    /**
     * Performs batch training on multiple samples.
     * 
     * @param inputs Array of input tensors
     * @param targets Array of target tensors
     * @param epochs Number of complete passes through the data
     * @return The average loss after training
     */
    double trainBatch(Tensor[] inputs, Tensor[] targets, int epochs);
    
    @Override
    default int getLayerCount() {
        return getTensorLayers().size();
    }
    
    @Override
    default boolean isInitialized() {
        return !getTensorLayers().isEmpty();
    }
    
    @Override
    default NetworkType getType() {
        return NetworkType.CONVOLUTIONAL;
    }
}