package deeplearningjava.layer;

import deeplearningjava.core.Node;
import deeplearningjava.core.activation.ActivationFunction;

/**
 * Standard fully-connected hidden layer for a neural network.
 * Applies activation functions to weighted sums of inputs.
 */
public class StandardLayer extends AbstractLayer {
    
    /**
     * Creates a standard layer with the specified size and activation function.
     * @param size Number of nodes
     * @param activationFunction Activation function for all nodes
     */
    public StandardLayer(int size, ActivationFunction activationFunction) {
        super(size, activationFunction, LayerType.HIDDEN);
    }
    
    @Override
    public double[] forward(double[] inputs) {
        if (inputs != null) {
            // Direct inputs provided
            setOutputs(inputs);
        } else {
            // Calculate outputs from incoming connections
            calculateOutputs();
        }
        
        return getOutputs();
    }
    
    @Override
    public double[] backward(double[] gradients) {
        // Calculate gradients for each node
        for (int i = 0; i < size; i++) {
            Node node = nodes.get(i);
            node.setGradient(gradients[i]);
            node.calculateHiddenGradient();
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
        
        // Propagate gradients to previous layer
        double[] previousGradients = new double[nodes.get(0).getIncomingConnections().size()];
        // TODO: Implement gradient propagation to previous layer
        
        return previousGradients;
    }
}