package deeplearningjava.api;

import java.util.List;
import deeplearningjava.core.Node;

/**
 * Interface representing a layer in a neural network.
 * A layer is a collection of nodes (neurons) that process inputs and produce outputs.
 */
public interface Layer {
    
    /**
     * Gets all nodes in this layer.
     * @return List of nodes that make up this layer
     */
    List<Node> getNodes();
    
    /**
     * Gets the size of the layer (number of nodes).
     * @return The number of nodes
     */
    int getSize();
    
    /**
     * Connect this layer to the next layer, creating appropriate edges.
     * @param nextLayer The next layer to connect to
     */
    void connectTo(Layer nextLayer);
    
    /**
     * Initialize weights for connections to the next layer.
     * @param nextLayerSize The size of the next layer
     */
    void initializeWeights(int nextLayerSize);
    
    /**
     * Performs forward propagation through this layer.
     * @param inputs Input values (null if taking inputs from connected layers)
     * @return Output values from this layer
     */
    double[] forward(double[] inputs);
    
    /**
     * Performs backward propagation through this layer.
     * @param gradients Gradients from the next layer (for output layer, this is the target values)
     * @return Gradients to propagate to the previous layer
     */
    double[] backward(double[] gradients);
    
    /**
     * Gets the type of this layer.
     * @return The layer type
     */
    LayerType getType();
    
    /**
     * For backward compatibility with the legacy layer type.
     * @return The API layer type
     */
    default LayerType getAPIType() {
        return getType();
    }
    
    /**
     * Enum defining the possible layer types.
     */
    enum LayerType {
        INPUT,
        HIDDEN,
        OUTPUT,
        BATCH_NORM,
        CONVOLUTIONAL,
        POOLING
    }
}