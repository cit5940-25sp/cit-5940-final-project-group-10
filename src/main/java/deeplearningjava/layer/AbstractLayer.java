package deeplearningjava.layer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import deeplearningjava.api.Layer;
import deeplearningjava.core.Edge;
import deeplearningjava.core.Node;
import deeplearningjava.core.activation.ActivationFunction;

/**
 * Abstract base implementation of the Layer interface.
 * Provides common functionality for all layer types.
 */
public abstract class AbstractLayer implements Layer {
    
    protected final List<Node> nodes;
    protected final int size;
    protected final ActivationFunction activationFunction;
    protected final LayerType type;
    
    /**
     * Creates a layer with the specified size and activation function.
     * 
     * @param size Number of nodes in this layer
     * @param activationFunction Activation function for the nodes
     * @param type Type of this layer
     */
    protected AbstractLayer(int size, ActivationFunction activationFunction, LayerType type) {
        if (size <= 0) {
            throw new IllegalArgumentException("Layer size must be positive");
        }
        
        this.size = size;
        this.activationFunction = Objects.requireNonNull(activationFunction, 
                "activationFunction must not be null");
        this.type = Objects.requireNonNull(type, "type must not be null");
        
        // Create nodes
        this.nodes = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            this.nodes.add(new Node(activationFunction));
        }
    }
    
    @Override
    public List<Node> getNodes() {
        return this.nodes;
    }
    
    @Override
    public int getSize() {
        return this.size;
    }
    
    @Override
    public deeplearningjava.api.Layer.LayerType getType() {
        switch (this.type) {
            case INPUT:
                return deeplearningjava.api.Layer.LayerType.INPUT;
            case HIDDEN:
                return deeplearningjava.api.Layer.LayerType.HIDDEN;
            case OUTPUT:
                return deeplearningjava.api.Layer.LayerType.OUTPUT;
            case BATCH_NORM:
                return deeplearningjava.api.Layer.LayerType.BATCH_NORM;
            default:
                throw new IllegalStateException("Unknown layer type: " + this.type);
        }
    }
    
    public boolean isLayerType(deeplearningjava.api.Layer.LayerType layerType) {
        return getType() == layerType;
    }
    
    @Override
    public void connectTo(Layer nextLayer) {
        Objects.requireNonNull(nextLayer, "nextLayer must not be null");
        
        for (Node sourceNode : this.nodes) {
            for (Node targetNode : nextLayer.getNodes()) {
                Edge edge = new Edge(sourceNode, targetNode);
                sourceNode.getOutgoingConnections().add(edge);
                targetNode.getIncomingConnections().add(edge);
            }
        }
    }
    
    @Override
    public void initializeWeights(int nextLayerSize) {
        for(Node node : this.nodes) {
            for (Edge edge : node.getOutgoingConnections()) {
                edge.initializeWeight(this.size, nextLayerSize);
            }
        }
    }
    
    /**
     * Gets the weights for all nodes in this layer.
     * For a standard fully-connected layer, returns a 2D array where:
     * - rows represent this layer's nodes
     * - columns represent the next layer's nodes
     * - weights[i][j] is the weight from node i in this layer to node j in the next layer
     * 
     * @return 2D array of weights (or null if no connections exist)
     */
    public double[][] getWeights() {
        if (nodes.isEmpty() || nodes.get(0).getOutgoingConnections().isEmpty()) {
            return null;
        }
        
        int nextLayerSize = nodes.get(0).getOutgoingConnections().size();
        double[][] weights = new double[size][nextLayerSize];
        
        for (int i = 0; i < size; i++) {
            Node node = nodes.get(i);
            List<Edge> connections = node.getOutgoingConnections();
            
            for (int j = 0; j < connections.size(); j++) {
                weights[i][j] = connections.get(j).getWeight();
            }
        }
        
        return weights;
    }
    
    /**
     * Sets the weights for all nodes in this layer.
     * The provided weights must match the layer connectivity.
     * 
     * @param weights 2D array of weights where weights[i][j] is the weight
     *                from node i in this layer to node j in the next layer
     * @throws IllegalArgumentException if weights dimensions don't match the network structure
     */
    public void setWeights(double[][] weights) {
        if (weights == null) {
            throw new IllegalArgumentException("Weights cannot be null");
        }
        
        if (weights.length != size) {
            throw new IllegalArgumentException(String.format(
                    "Weights rows (%d) must match layer size (%d)",
                    weights.length, size));
        }
        
        for (int i = 0; i < size; i++) {
            Node node = nodes.get(i);
            List<Edge> connections = node.getOutgoingConnections();
            
            if (connections.isEmpty()) {
                continue; // No connections to set weights for
            }
            
            if (weights[i].length != connections.size()) {
                throw new IllegalArgumentException(String.format(
                        "Weights columns (%d) must match next layer size (%d)",
                        weights[i].length, connections.size()));
            }
            
            for (int j = 0; j < connections.size(); j++) {
                connections.get(j).setWeight(weights[i][j]);
            }
        }
    }
    
    /**
     * Gets the bias values for all nodes in this layer.
     * 
     * @return Array of bias values
     */
    public double[] getBiases() {
        double[] biases = new double[size];
        for (int i = 0; i < size; i++) {
            biases[i] = nodes.get(i).getBias();
        }
        return biases;
    }
    
    /**
     * Sets the bias values for all nodes in this layer.
     * 
     * @param biases Array of bias values (must match layer size)
     * @throws IllegalArgumentException if biases array length doesn't match layer size
     */
    public void setBiases(double[] biases) {
        if (biases == null) {
            throw new IllegalArgumentException("Biases cannot be null");
        }
        
        if (biases.length != size) {
            throw new IllegalArgumentException(String.format(
                    "Biases length (%d) must match layer size (%d)",
                    biases.length, size));
        }
        
        for (int i = 0; i < size; i++) {
            nodes.get(i).setBias(biases[i]);
        }
    }
    
    /**
     * Sets output values for this layer's nodes (typically used for input layers).
     * @param inputs Input values (size must match layer size)
     */
    protected void setOutputs(double[] inputs) {
        if (inputs.length != this.size) {
            throw new IllegalArgumentException(String.format(
                    "Input size (%d) must match layer size (%d)",
                    inputs.length, this.size));
        }
        
        for (int i = 0; i < this.size; i++) {
            nodes.get(i).setValue(inputs[i]);
        }
    }
    
    /**
     * Calculates outputs for all nodes in this layer by applying
     * their activation functions to their net inputs.
     */
    protected void calculateOutputs() {
        for (Node node : this.nodes) {
            node.calculateNetInput();
            node.applyActivation();
        }
    }
    
    /**
     * Gets the current output values of this layer.
     * @return Array of output values
     */
    protected double[] getOutputs() {
        double[] outputs = new double[size];
        for (int i = 0; i < size; i++) {
            outputs[i] = nodes.get(i).getValue();
        }
        return outputs;
    }
}