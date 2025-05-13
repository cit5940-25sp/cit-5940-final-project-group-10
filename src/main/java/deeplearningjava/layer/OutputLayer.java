package deeplearningjava.layer;

import deeplearningjava.core.Node;
import deeplearningjava.core.Edge;
import deeplearningjava.core.activation.ActivationFunction;
import java.util.List;

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
        if (nodes.isEmpty() || nodes.get(0).getIncomingConnections().isEmpty()) {
            return new double[0]; // No previous layer
        }
        
        // Determine the size of the previous layer by checking
        // how many unique source nodes connect to this layer
        Node firstNode = nodes.get(0);
        List<Edge> incomingConnections = firstNode.getIncomingConnections();
        
        if (incomingConnections.isEmpty()) {
            return new double[0];
        }
        
        // Assuming a fully connected network, each output node
        // is connected to every node in the previous layer.
        // So we can count the number of incoming connections
        // to the first output node to get the previous layer size.
        int previousLayerSize = incomingConnections.size();
        double[] previousGradients = new double[previousLayerSize];
        
        // Map each gradient to the corresponding previous layer node
        for (int i = 0; i < previousLayerSize; i++) {
            // Get the source node for this connection
            Node sourceNode = incomingConnections.get(i).getSourceNode();
            double gradientSum = 0.0;
            
            // Sum gradients from all output nodes connected to this source node
            for (Node outputNode : nodes) {
                for (Edge edge : outputNode.getIncomingConnections()) {
                    if (edge.getSourceNode() == sourceNode) {
                        gradientSum += edge.getWeight() * outputNode.getGradient();
                    }
                }
            }
            
            previousGradients[i] = gradientSum;
        }
        
        return previousGradients;
    }
}