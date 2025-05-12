package deeplearningjava.network;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import deeplearningjava.api.BaseNetwork;

/**
 * Abstract base class providing common functionality for all network types.
 * This class implements common behaviors and properties shared by all networks.
 */
public abstract class AbstractNetwork implements BaseNetwork {
    
    protected double learningRate;
    protected final NetworkType networkType;
    
    /**
     * Creates a new abstract network with the specified type.
     * 
     * @param networkType The type of network
     */
    protected AbstractNetwork(NetworkType networkType) {
        this.networkType = Objects.requireNonNull(networkType, "networkType must not be null");
        this.learningRate = 0.01; // Default learning rate
    }
    
    @Override
    public void setLearningRate(double learningRate) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        this.learningRate = learningRate;
    }
    
    @Override
    public double getLearningRate() {
        return learningRate;
    }
    
    @Override
    public NetworkType getType() {
        return networkType;
    }
    
    /**
     * Creates a string representation describing this network.
     * 
     * @return A detailed description of the network structure
     */
    public String getSummary() {
        StringBuilder summary = new StringBuilder();
        summary.append(getType()).append(" Network Summary:\n");
        summary.append("Layer Count: ").append(getLayerCount()).append("\n");
        summary.append("Learning Rate: ").append(getLearningRate()).append("\n");
        
        // Add network-specific details (implemented by subclasses)
        appendDetails(summary);
        
        return summary.toString();
    }
    
    /**
     * Appends network-specific details to the summary.
     * To be implemented by subclasses.
     * 
     * @param summary The StringBuilder to append details to
     */
    protected abstract void appendDetails(StringBuilder summary);
    
    /**
     * Verifies that the network structure is valid before performing operations.
     * 
     * @throws IllegalStateException if the network is not properly configured
     */
    protected void validateNetwork() {
        if (!isInitialized()) {
            throw new IllegalStateException("Network has not been initialized or has no layers");
        }
    }
    
    /**
     * Validates parameters for batch training.
     * 
     * @param inputs Array of input data
     * @param targets Array of target data
     * @param epochs Number of training epochs
     * @throws IllegalArgumentException if the parameters are invalid
     */
    protected void validateBatchParameters(Object[] inputs, Object[] targets, int epochs) {
        Objects.requireNonNull(inputs, "inputs must not be null");
        Objects.requireNonNull(targets, "targets must not be null");
        
        if (inputs.length == 0) {
            throw new IllegalArgumentException("inputs cannot be empty");
        }
        
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException(
                    "Number of inputs (" + inputs.length + 
                    ") must match number of targets (" + targets.length + ")");
        }
        
        if (epochs <= 0) {
            throw new IllegalArgumentException("epochs must be positive");
        }
    }
}