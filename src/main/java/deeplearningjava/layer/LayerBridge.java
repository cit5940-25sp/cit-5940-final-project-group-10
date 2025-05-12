package deeplearningjava.layer;

import deeplearningjava.api.Layer;
import deeplearningjava.core.activation.ActivationFunction;
import deeplearningjava.core.activation.ActivationFunctions;
import java.util.function.DoubleUnaryOperator;

/**
 * Bridge class for creating layers that are compatible with the new API.
 */
public class LayerBridge {

    /**
     * Creates an InputLayer.
     * 
     * @param size Number of input nodes
     * @return An InputLayer instance
     */
    public static Layer createInputLayer(int size) {
        return new InputLayer(size);
    }
    
    /**
     * Creates a StandardLayer.
     * 
     * @param size Number of nodes
     * @param activation Activation function
     * @return A StandardLayer instance
     */
    public static Layer createStandardLayer(
            int size, 
            ActivationFunction activation) {
        return new StandardLayer(size, activation);
    }
    
    /**
     * Creates an OutputLayer.
     * 
     * @param size Number of output nodes
     * @param activation Activation function
     * @param useSoftmax Whether to use softmax activation
     * @return An OutputLayer instance
     */
    public static Layer createOutputLayer(
            int size, 
            ActivationFunction activation,
            boolean useSoftmax) {
        return new OutputLayer(size, activation, useSoftmax);
    }
    
    /**
     * Creates a BatchNormLayer.
     * 
     * @param size Number of nodes
     * @param activation Activation function
     * @return A BatchNormLayer instance
     */
    public static Layer createBatchNormLayer(
            int size, 
            ActivationFunction activation) {
        return new BatchNormLayer(size, activation);
    }
}