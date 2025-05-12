package deeplearningjava.api;

import java.util.List;

/**
 * Interface representing a traditional dense neural network.
 * A dense neural network consists of fully connected layers that process
 * input vectors to produce output vectors.
 */
public interface Network extends Trainable, BaseNetwork {
    
    /**
     * Performs forward propagation through the network.
     * @param inputs Input values to the network
     * @return The network's output values
     */
    double[] forward(double[] inputs);
    
    /**
     * Gets all layers in the network.
     * @return An unmodifiable list of the network's layers
     */
    List<Layer> getLayers();
    
    /**
     * Adds a layer to the network.
     * @param layer The layer to add
     * @throws IllegalArgumentException if the layer is incompatible
     */
    void addLayer(Layer layer);
    
    /**
     * Gets the size of the input layer.
     * @return The number of input neurons
     */
    int getInputSize();
    
    /**
     * Gets the size of the output layer.
     * @return The number of output neurons
     */
    int getOutputSize();
    
    @Override
    default int getLayerCount() {
        return getLayers().size();
    }
    
    @Override
    default boolean isInitialized() {
        return !getLayers().isEmpty();
    }
    
    @Override
    default NetworkType getType() {
        return NetworkType.DENSE;
    }
}