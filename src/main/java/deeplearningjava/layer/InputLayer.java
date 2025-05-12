package deeplearningjava.layer;

import deeplearningjava.core.activation.ActivationFunction;
import deeplearningjava.core.activation.ActivationFunctions;

/**
 * Input layer for a neural network.
 * This layer doesn't compute anything; it simply passes inputs to the next layer.
 */
public class InputLayer extends AbstractLayer {
    
    /**
     * Creates an input layer with the specified size.
     * @param size Number of input nodes
     */
    public InputLayer(int size) {
        super(size, ActivationFunctions.linear(), LayerType.INPUT);
    }
    
    @Override
    public double[] forward(double[] inputs) {
        // Input layer just passes through its inputs
        setOutputs(inputs);
        return getOutputs();
    }
    
    @Override
    public double[] backward(double[] gradients) {
        // Input layer doesn't have parameters to update
        return new double[size]; // No gradients to pass back
    }
}