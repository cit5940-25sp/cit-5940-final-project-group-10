package deeplearningjava.core;

import java.util.Objects;

/**
 * Represents a connection between two nodes in a neural network.
 */
public class Edge {
    private final Node sourceNode;
    private final Node targetNode;
    private double weight;
    
    /**
     * Creates a connection between two nodes.
     * @param sourceNode The source node
     * @param targetNode The target node
     */
    public Edge(Node sourceNode, Node targetNode) {
        this.sourceNode = Objects.requireNonNull(sourceNode, "sourceNode must not be null");
        this.targetNode = Objects.requireNonNull(targetNode, "targetNode must not be null");
        
        // Initialize with small random weight
        this.weight = (Math.random() - 0.5) * 0.2;
    }
    
    /**
     * Gets the source node of this connection.
     * @return The source node
     */
    public Node getSourceNode() {
        return sourceNode;
    }
    
    /**
     * Gets the target node of this connection.
     * @return The target node
     */
    public Node getTargetNode() {
        return targetNode;
    }
    
    /**
     * Gets the weight of this connection.
     * @return The weight
     */
    public double getWeight() {
        return weight;
    }
    
    /**
     * Sets the weight of this connection.
     * @param weight The new weight
     */
    public void setWeight(double weight) {
        this.weight = weight;
    }
    
    /**
     * Initializes the weight using He/Xavier initialization.
     * @param fanIn Number of input connections to the layer
     * @param fanOut Number of output connections from the layer
     */
    public void initializeWeight(int fanIn, int fanOut) {
        // Use Xavier/Glorot initialization for tanh/sigmoid
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        this.weight = (Math.random() * 2.0 - 1.0) * limit;
    }
    
    /**
     * Updates the weight based on gradient descent.
     * @param learningRate The learning rate
     */
    public void updateWeight(double learningRate) {
        double inputValue = sourceNode.getValue();
        double gradient = targetNode.getGradient();
        
        // Update weight using gradient descent
        this.weight -= learningRate * gradient * inputValue;
    }
}