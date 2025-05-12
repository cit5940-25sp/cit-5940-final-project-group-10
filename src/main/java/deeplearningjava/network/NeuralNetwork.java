package deeplearningjava.network;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import deeplearningjava.api.Layer;
import deeplearningjava.api.Network;
import deeplearningjava.network.optimizer.Optimizer;
import deeplearningjava.network.optimizer.SGD;

/**
 * Abstract base class for neural network implementations.
 * Provides common functionality for all neural networks.
 */
public abstract class NeuralNetwork implements Network {
    
    protected final List<Layer> layers;
    protected Optimizer optimizer;
    
    /**
     * Creates a neural network with no layers.
     */
    public NeuralNetwork() {
        this.layers = new ArrayList<>();
        this.optimizer = new SGD(0.01); // Default optimizer
    }
    
    /**
     * Creates a neural network with the specified layers.
     * @param layers Initial layers for the network
     */
    public NeuralNetwork(List<Layer> layers) {
        this.layers = new ArrayList<>(Objects.requireNonNull(layers, "layers must not be null"));
        this.optimizer = new SGD(0.01); // Default optimizer
    }
    
    @Override
    public void addLayer(Layer layer) {
        Objects.requireNonNull(layer, "layer must not be null");
        
        if (!layers.isEmpty()) {
            Layer previousLayer = layers.get(layers.size() - 1);
            previousLayer.connectTo(layer);
            previousLayer.initializeWeights(layer.getSize());
        }
        
        layers.add(layer);
    }
    
    @Override
    public List<Layer> getLayers() {
        return Collections.unmodifiableList(layers);
    }
    
    @Override
    public void setLearningRate(double learningRate) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        optimizer.setLearningRate(learningRate);
    }
    
    @Override
    public double getLearningRate() {
        return optimizer.getLearningRate();
    }
    
    /**
     * Sets the optimizer for this network.
     * @param optimizer The optimizer to use
     */
    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer must not be null");
    }
    
    /**
     * Gets the current optimizer.
     * @return The optimizer
     */
    public Optimizer getOptimizer() {
        return optimizer;
    }
    
    @Override
    public double trainBatch(double[][] inputs, double[][] targets, int epochs) {
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException("Number of inputs must match number of targets");
        }
        
        double totalLoss = 0;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochLoss = 0;
            
            for (int i = 0; i < inputs.length; i++) {
                double[] output = train(inputs[i], targets[i]);
                epochLoss += calculateLoss(output, targets[i]);
            }
            
            totalLoss = epochLoss / inputs.length;
            System.out.printf("Epoch %d/%d - Loss: %.6f%n", epoch + 1, epochs, totalLoss);
        }
        
        return totalLoss;
    }
    
    /**
     * Calculates the loss between predicted and target outputs.
     * Default implementation uses Mean Squared Error (MSE).
     * 
     * @param predicted Predicted values
     * @param target Target values
     * @return The loss value
     */
    protected double calculateLoss(double[] predicted, double[] target) {
        if (predicted.length != target.length) {
            throw new IllegalArgumentException("Predicted and target arrays must have the same length");
        }
        
        double sum = 0;
        for (int i = 0; i < predicted.length; i++) {
            double diff = predicted[i] - target[i];
            sum += diff * diff;
        }
        
        return sum / predicted.length; // Mean Squared Error
    }
}