package deeplearningjava.layer;

import deeplearningjava.core.Node;
import deeplearningjava.core.activation.ActivationFunction;

/**
 * Batch normalization layer for a neural network.
 * Normalizes activations using running statistics for improved training stability.
 */
public class BatchNormLayer extends AbstractLayer {
    
    // Batch normalization parameters
    private final double[] runningMean;        // Moving average of means
    private final double[] runningVariance;    // Moving average of variances
    private final double[] scale;              // Learnable scale parameter (gamma)
    private final double[] bias;               // Learnable bias parameter (beta)
    private final double epsilon;              // Small constant for numerical stability
    private final double momentum;             // Momentum for running statistics updates
    
    // Cache for backpropagation
    private double[] inputCache;               // Input values
    private double[] normalizedCache;          // Normalized values before scale and shift
    
    /**
     * Creates a batch normalization layer with default parameters.
     * 
     * @param size Size of the layer
     * @param activationFunction Activation function to apply after normalization
     */
    public BatchNormLayer(int size, ActivationFunction activationFunction) {
        this(size, activationFunction, null, null, null, null, 1e-5, 0.9);
    }
    
    /**
     * Creates a batch normalization layer with specified parameters.
     * 
     * @param size Size of the layer
     * @param activationFunction Activation function to apply after normalization
     * @param mean Initial running mean values (null for zeros)
     * @param variance Initial running variance values (null for ones)
     * @param scale Initial scale/gamma parameters (null for ones)
     * @param bias Initial bias/beta parameters (null for zeros)
     * @param epsilon Small constant for numerical stability
     * @param momentum Momentum for running statistics updates
     */
    public BatchNormLayer(int size, ActivationFunction activationFunction,
                        double[] mean, double[] variance, 
                        double[] scale, double[] bias,
                        double epsilon, double momentum) {
        super(size, activationFunction, LayerType.BATCH_NORM);
        
        this.runningMean = new double[size];
        this.runningVariance = new double[size];
        this.scale = new double[size];
        this.bias = new double[size];
        this.epsilon = epsilon;
        this.momentum = momentum;
        
        // Initialize parameters
        for (int i = 0; i < size; i++) {
            this.runningMean[i] = (mean != null && i < mean.length) ? mean[i] : 0.0;
            this.runningVariance[i] = (variance != null && i < variance.length) ? variance[i] : 1.0;
            this.scale[i] = (scale != null && i < scale.length) ? scale[i] : 1.0;
            this.bias[i] = (bias != null && i < bias.length) ? bias[i] : 0.0;
        }
        
        // Initialize caches
        this.inputCache = new double[size];
        this.normalizedCache = new double[size];
    }
    
    @Override
    public double[] forward(double[] inputs) {
        if (inputs != null) {
            // Direct input provided
            applyBatchNorm(inputs);
        } else {
            // Calculate inputs from previous layer
            for (int i = 0; i < size; i++) {
                Node node = nodes.get(i);
                inputCache[i] = node.calculateNetInput();
            }
            applyBatchNorm(inputCache);
        }
        
        return getOutputs();
    }
    
    /**
     * Applies batch normalization to the inputs.
     * 
     * @param inputs Input values
     */
    private void applyBatchNorm(double[] inputs) {
        // Store inputs for backpropagation
        System.arraycopy(inputs, 0, inputCache, 0, size);
        
        // Apply normalization to each node
        for (int i = 0; i < size; i++) {
            // Normalize: (x - mean) / sqrt(variance + epsilon)
            double normalized = (inputs[i] - runningMean[i]) / 
                               Math.sqrt(runningVariance[i] + epsilon);
            
            // Store normalized value for backpropagation
            normalizedCache[i] = normalized;
            
            // Scale and shift: gamma * normalized + beta
            double output = scale[i] * normalized + bias[i];
            
            // Apply activation function
            output = activationFunction.apply(output);
            
            // Set the node value
            nodes.get(i).setValue(output);
        }
    }
    
    @Override
    public double[] backward(double[] gradients) {
        // Here we would implement the backward pass for batch normalization
        // This is a complex operation involving the chain rule for the
        // normalization, scale, and shift operations
        
        // For a simple implementation, we'll just pass the gradients through
        // TODO: Implement proper batch normalization backward pass
        
        return gradients;
    }
    
    /**
     * Gets the running mean values.
     * @return Array of means
     */
    public double[] getRunningMean() {
        double[] copy = new double[size];
        System.arraycopy(runningMean, 0, copy, 0, size);
        return copy;
    }
    
    /**
     * Gets the running variance values.
     * @return Array of variances
     */
    public double[] getRunningVariance() {
        double[] copy = new double[size];
        System.arraycopy(runningVariance, 0, copy, 0, size);
        return copy;
    }
    
    /**
     * Gets the scale (gamma) parameters.
     * @return Array of scale parameters
     */
    public double[] getScale() {
        double[] copy = new double[size];
        System.arraycopy(scale, 0, copy, 0, size);
        return copy;
    }
    
    /**
     * Gets the bias (beta) parameters.
     * @return Array of bias parameters
     */
    public double[] getBias() {
        double[] copy = new double[size];
        System.arraycopy(bias, 0, copy, 0, size);
        return copy;
    }
}