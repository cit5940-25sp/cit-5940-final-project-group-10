package deeplearningjava.layer;

import deeplearningjava.core.Node;
import deeplearningjava.core.activation.ActivationFunction;

/**
 * Output layer for a neural network.
 * Generates final predictions and calculates error gradients.
 */
public class OutputLayer extends AbstractLayer {
    
    private final boolean useSoftmax;
    
    /**
     * Creates an output layer with the specified size and activation function.
     * @param size Number of output nodes
     * @param activationFunction Activation function for the nodes
     * @param useSoftmax Whether to apply softmax activation
     */
    public OutputLayer(int size, ActivationFunction activationFunction, boolean useSoftmax) {
        super(size, activationFunction, LayerType.OUTPUT);
        this.useSoftmax = useSoftmax;
    }
    
    /**
     * Checks if this layer uses softmax activation.
     * @return True if softmax is used, false otherwise
     */
    public boolean usesSoftmax() {
        return useSoftmax;
    }
    
    @Override
    public double[] forward(double[] inputs) {
        // Calculate outputs
        if (inputs != null) {
            setOutputs(inputs);
        } else {
            calculateOutputs();
        }
        
        // Apply softmax if necessary
        if (useSoftmax) {
            applySoftmax();
        }
        
        return getOutputs();
    }
    
    /**
     * Applies softmax activation to the outputs.
     * Softmax: exp(x_i) / sum(exp(x_j))
     */
    private void applySoftmax() {
        // Get node values
        double[] values = new double[size];
        for (int i = 0; i < size; i++) {
            values[i] = nodes.get(i).getValue();
        }
        
        // Find max for numerical stability
        double max = Double.NEGATIVE_INFINITY;
        for (double value : values) {
            if (value > max) {
                max = value;
            }
        }
        
        // Calculate exp(x_i - max) for each value
        double[] exps = new double[size];
        double sum = 0;
        for (int i = 0; i < size; i++) {
            exps[i] = Math.exp(values[i] - max);
            sum += exps[i];
        }
        
        // Normalize by sum
        for (int i = 0; i < size; i++) {
            nodes.get(i).setValue(exps[i] / sum);
        }
    }
    
    @Override
    public double[] backward(double[] targets) {
        // For output layer, the input gradient is the target values
        // Calculate output gradients based on loss function
        for (int i = 0; i < size; i++) {
            Node node = nodes.get(i);
            
            if (useSoftmax) {
                // For softmax with cross-entropy loss, gradient = (output - target)
                node.setGradient(node.getValue() - targets[i]);
            } else {
                // For MSE loss, gradient = (output - target) * activation_derivative
                node.calculateOutputGradient(targets[i]);
            }
        }
        
        // Update weights and biases
        for (Node node : nodes) {
            // Update bias
            double bias = node.getBias();
            double updatedBias = bias - 0.01 * node.getGradient(); // TODO: Use optimizer
            node.setBias(updatedBias);
            
            // Update weights
            node.updateIncomingWeights(0.01); // TODO: Use optimizer
        }
        
        // Calculate gradients for previous layer
        double[] previousGradients = new double[nodes.get(0).getIncomingConnections().size()];
        // TODO: Implement gradient propagation to previous layer
        
        return previousGradients;
    }
}