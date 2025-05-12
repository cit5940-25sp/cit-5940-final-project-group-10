package deeplearningjava.core;

import deeplearningjava.core.activation.ActivationFunction;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Represents a node (neuron) in a neural network.
 */
public class Node {
    private double value = 0;
    private double bias;
    private double gradient = 0;
    private double netInput = 0;

    private final List<Edge> outgoingConnections = new ArrayList<>();
    private final List<Edge> incomingConnections = new ArrayList<>();

    private final ActivationFunction activationFunction;
    
    /**
     * Creates a node with the specified activation function.
     * @param activationFunction The activation function to use
     */
    public Node(ActivationFunction activationFunction) {
        this.activationFunction = Objects.requireNonNull(activationFunction, 
                "activationFunction must not be null");
        this.bias = (Math.random() - 0.5) * 0.2; // Initialize with small random value
    }
    
    /**
     * Calculates the net input to this node from incoming connections.
     * @return The weighted sum of inputs plus bias
     */
    public double calculateNetInput() {
        this.netInput = 0.0;

        if (!incomingConnections.isEmpty()) {
            for (Edge edge : incomingConnections) {
                this.netInput += edge.getSourceNode().getValue() * edge.getWeight();
            }
        }
        this.netInput += this.bias;

        return this.netInput;
    }
    
    /**
     * Applies the activation function to the net input.
     */
    public void applyActivation() {
        this.value = this.activationFunction.apply(this.netInput);
    }
    
    /**
     * Calculates the gradient for this node as an output node.
     * @param targetValue The target value for this node
     */
    public void calculateOutputGradient(double targetValue) {
        double errorDerivative = this.value - targetValue;
        double activationDerivative = this.activationFunction.derivative(this.netInput);
        this.gradient = errorDerivative * activationDerivative;
    }
    
    /**
     * Calculates the gradient for this node as a hidden node.
     */
    public void calculateHiddenGradient() {
        if (outgoingConnections.isEmpty()) {
            this.gradient = 0;
            return;
        }
        
        double downstreamWeightedSum = 0;
        for (Edge edge : outgoingConnections) {
            downstreamWeightedSum += edge.getWeight() * edge.getTargetNode().getGradient();
        }
        
        double activationDerivative = this.activationFunction.derivative(this.netInput);
        this.gradient = downstreamWeightedSum * activationDerivative;
    }
    
    /**
     * Updates the bias based on the gradient and learning rate.
     * @param learningRate The learning rate
     */
    public void updateBias(double learningRate) {
        this.bias -= learningRate * this.gradient;
    }
    
    /**
     * Updates the weights of incoming connections.
     * @param learningRate The learning rate
     */
    public void updateIncomingWeights(double learningRate) {
        for (Edge edge : incomingConnections) {
            edge.updateWeight(learningRate);
        }
    }
    
    // Getters and setters
    
    public double getValue() {
        return this.value;
    }
    
    public double getBias() {
        return this.bias;
    }
    
    public double getGradient() {
        return this.gradient;
    }
    
    public double getNetInput() {
        return this.netInput;
    }
    
    public List<Edge> getOutgoingConnections() {
        return this.outgoingConnections;
    }
    
    public List<Edge> getIncomingConnections() {
        return this.incomingConnections;
    }
    
    public ActivationFunction getActivationFunction() {
        return this.activationFunction;
    }
    
    public void setValue(double value) {
        this.value = value;
        this.netInput = value;
    }
    
    public void setGradient(double gradient) {
        this.gradient = gradient;
    }
    
    public void setBias(double bias) {
        this.bias = bias;
    }
    
    public void setInputLayerValue(double value) {
        this.value = value;
        this.netInput = value;
    }
}