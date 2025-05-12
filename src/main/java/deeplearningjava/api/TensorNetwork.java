package deeplearningjava.api;

import java.util.List;
import deeplearningjava.core.tensor.Tensor;

/**
 * Interface representing a neural network that operates on tensor data.
 * This interface extends BaseNetwork but uses Tensor objects instead of
 * arrays, allowing for multi-dimensional inputs like images.
 */
public interface TensorNetwork extends Trainable, BaseNetwork {
    
    /**
     * Performs forward propagation through the network using tensor input.
     * @param input Tensor input to the network
     * @return The network's output as a tensor
     */
    Tensor forward(Tensor input);
    
    /**
     * Gets all tensor layers in the network.
     * @return An unmodifiable list of the network's tensor layers
     */
    List<TensorLayer> getTensorLayers();
    
    /**
     * Adds a tensor layer to the network.
     * @param layer The tensor layer to add
     * @throws IllegalArgumentException if the layer is incompatible
     */
    void addTensorLayer(TensorLayer layer);
    
    /**
     * Gets the input shape of the network.
     * @return The shape of the input tensor
     */
    int[] getInputShape();
    
    /**
     * Gets the output shape of the network.
     * @return The shape of the output tensor
     */
    int[] getOutputShape();
    
    /**
     * Trains the network on tensor data.
     * 
     * @param input Input tensor
     * @param target Target output tensor
     * @return The actual output tensor after this training iteration
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
}