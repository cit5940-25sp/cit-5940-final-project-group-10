package deeplearningjava.layer.tensor;

import java.util.Arrays;
import java.util.Objects;
import deeplearningjava.api.TensorLayer;
import deeplearningjava.core.activation.ActivationFunction;
import deeplearningjava.core.tensor.Tensor;

/**
 * Implementation of a fully connected (dense) layer for neural networks.
 * This layer connects every input neuron to every output neuron.
 */
public class FullyConnectedLayer extends AbstractTensorLayer {
    
    private final int inputSize;
    private final int outputSize;
    private final boolean useSoftmax;
    
    // Weights and biases
    private Tensor weights; // Shape: [outputSize, inputSize]
    private Tensor bias;    // Shape: [outputSize]
    
    // For backpropagation
    private Tensor lastInput;
    private Tensor lastOutput;
    private Tensor preActivation;
    private Tensor weightGradients;
    private Tensor biasGradients;
    
    /**
     * Creates a new fully connected layer.
     * 
     * @param inputShape The shape of the input (must be 1D)
     * @param outputSize The number of output neurons
     * @param useSoftmax Whether to use softmax activation for the output
     * @param activationFunction The activation function to use
     */
    public FullyConnectedLayer(int[] inputShape, int outputSize, boolean useSoftmax,
                             ActivationFunction activationFunction) {
        super(inputShape, new int[]{outputSize}, activationFunction, LayerType.FULLY_CONNECTED);
        
        if (inputShape.length != 1) {
            throw new IllegalArgumentException("Input shape must be 1D, got " + formatShape(inputShape));
        }
        
        this.inputSize = inputShape[0];
        this.outputSize = outputSize;
        this.useSoftmax = useSoftmax;
        
        initializeParameters();
    }
    
    @Override
    public void initializeParameters() {
        // Initialize weights and bias
        weights = new Tensor(outputSize, inputSize);
        bias = new Tensor(outputSize);
        
        // Xavier initialization for weights
        xavierInitialization(weights, inputSize, outputSize);
        
        // Initialize bias to small values
        double[] biasData = bias.getData();
        Arrays.fill(biasData, 0.01);
        
        // Initialize gradients
        weightGradients = new Tensor(outputSize, inputSize);
        biasGradients = new Tensor(outputSize);
    }
    
    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "Input tensor must not be null");
        
        int[] shape = input.getShape();
        if (shape.length != 1 || shape[0] != inputSize) {
            throw new IllegalArgumentException(
                    "Expected input shape [" + inputSize + "], got " + formatShape(shape));
        }
        
        // Save input for backward pass
        lastInput = input.copy();
        
        // Perform matrix multiplication: output = weights * input + bias
        double[] inputData = input.getData();
        double[] outputData = new double[outputSize];
        
        for (int o = 0; o < outputSize; o++) {
            double sum = bias.get(o);
            
            for (int i = 0; i < inputSize; i++) {
                sum += weights.get(o, i) * inputData[i];
            }
            
            outputData[o] = sum;
        }
        
        // Save pre-activation for backward pass
        preActivation = new Tensor(outputData, outputSize);
        
        // Apply activation function
        Tensor output;
        if (useSoftmax) {
            output = applySoftmax(preActivation);
        } else {
            output = applyActivation(preActivation);
        }
        
        // Save output for backward pass
        lastOutput = output.copy();
        
        return output;
    }
    
    /**
     * Applies the softmax function to the input tensor.
     * 
     * @param input The input tensor
     * @return The softmax output tensor
     */
    private Tensor applySoftmax(Tensor input) {
        double[] inputData = input.getData();
        double[] outputData = new double[inputData.length];
        
        // Find max for numerical stability
        double max = Double.NEGATIVE_INFINITY;
        for (double value : inputData) {
            max = Math.max(max, value);
        }
        
        // Compute exp(x - max) for each element
        double sum = 0.0;
        for (int i = 0; i < inputData.length; i++) {
            outputData[i] = Math.exp(inputData[i] - max);
            sum += outputData[i];
        }
        
        // Normalize
        for (int i = 0; i < outputData.length; i++) {
            outputData[i] /= sum;
        }
        
        return new Tensor(outputData, input.getShape());
    }
    
    @Override
    public Tensor backward(Tensor gradients) {
        Objects.requireNonNull(gradients, "Gradient tensor must not be null");
        
        int[] shape = gradients.getShape();
        if (shape.length != 1 || shape[0] != outputSize) {
            throw new IllegalArgumentException(
                    "Expected gradient shape [" + outputSize + "], got " + formatShape(shape));
        }
        
        // Compute gradients
        Tensor outputGradients;
        if (useSoftmax) {
            outputGradients = computeSoftmaxGradients(gradients);
        } else {
            // Apply activation derivative
            Tensor activationDerivatives = applyActivationDerivative(preActivation);
            
            // Element-wise multiplication with incoming gradients
            double[] activationDerivData = activationDerivatives.getData();
            double[] gradientData = gradients.getData();
            double[] outputGradData = new double[outputSize];
            
            for (int i = 0; i < outputSize; i++) {
                outputGradData[i] = gradientData[i] * activationDerivData[i];
            }
            
            outputGradients = new Tensor(outputGradData, outputSize);
        }
        
        // Compute weight gradients
        updateWeightGradients(outputGradients);
        
        // Compute bias gradients
        updateBiasGradients(outputGradients);
        
        // Compute input gradients: input_grad = weights^T * output_grad
        return computeInputGradients(outputGradients);
    }
    
    /**
     * Computes the softmax gradients.
     * 
     * @param gradients The incoming gradients
     * @return The softmax gradients
     */
    private Tensor computeSoftmaxGradients(Tensor gradients) {
        double[] outputData = lastOutput.getData();
        double[] gradientData = gradients.getData();
        double[] softmaxGradData = new double[outputSize];
        
        // For each output unit
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                // Softmax derivative: softmax_i * (delta_ij - softmax_j)
                double derivative = outputData[i] * ((i == j ? 1.0 : 0.0) - outputData[j]);
                softmaxGradData[i] += gradientData[j] * derivative;
            }
        }
        
        return new Tensor(softmaxGradData, outputSize);
    }
    
    /**
     * Updates the weight gradients.
     * 
     * @param outputGradients The output gradients
     */
    private void updateWeightGradients(Tensor outputGradients) {
        double[] outputGradData = outputGradients.getData();
        double[] inputData = lastInput.getData();
        
        for (int o = 0; o < outputSize; o++) {
            for (int i = 0; i < inputSize; i++) {
                double gradient = outputGradData[o] * inputData[i];
                double currentGradient = weightGradients.get(o, i);
                weightGradients.set(currentGradient + gradient, o, i);
            }
        }
    }
    
    /**
     * Updates the bias gradients.
     * 
     * @param outputGradients The output gradients
     */
    private void updateBiasGradients(Tensor outputGradients) {
        double[] outputGradData = outputGradients.getData();
        
        for (int o = 0; o < outputSize; o++) {
            double currentGradient = biasGradients.get(o);
            biasGradients.set(currentGradient + outputGradData[o], o);
        }
    }
    
    /**
     * Computes the input gradients.
     * 
     * @param outputGradients The output gradients
     * @return The input gradients
     */
    private Tensor computeInputGradients(Tensor outputGradients) {
        double[] outputGradData = outputGradients.getData();
        double[] inputGradData = new double[inputSize];
        
        for (int i = 0; i < inputSize; i++) {
            double sum = 0.0;
            for (int o = 0; o < outputSize; o++) {
                sum += weights.get(o, i) * outputGradData[o];
            }
            inputGradData[i] = sum;
        }
        
        return new Tensor(inputGradData, inputSize);
    }
    
    @Override
    public void updateParameters(double learningRate) {
        // Update weights
        double[] weightsData = weights.getData();
        double[] weightGradData = weightGradients.getData();
        
        for (int i = 0; i < weightsData.length; i++) {
            weightsData[i] -= learningRate * weightGradData[i];
            // Reset gradient
            weightGradData[i] = 0.0;
        }
        
        // Update bias
        double[] biasData = bias.getData();
        double[] biasGradData = biasGradients.getData();
        
        for (int i = 0; i < biasData.length; i++) {
            biasData[i] -= learningRate * biasGradData[i];
            // Reset gradient
            biasGradData[i] = 0.0;
        }
    }
    
    /**
     * Gets the weights used by this layer.
     * 
     * @return The weights tensor
     */
    public Tensor getWeights() {
        return weights.copy();
    }
    
    /**
     * Sets the weights for this layer.
     * 
     * @param weights The weights tensor (must match expected dimensions)
     * @throws IllegalArgumentException if weights dimensions don't match the layer structure
     */
    public void setWeights(Tensor weights) {
        if (weights == null) {
            throw new IllegalArgumentException("Weights tensor cannot be null");
        }
        
        int[] expectedShape = this.weights.getShape();
        int[] providedShape = weights.getShape();
        
        if (!Arrays.equals(expectedShape, providedShape)) {
            throw new IllegalArgumentException(String.format(
                    "Weights shape mismatch: expected %s, got %s",
                    Arrays.toString(expectedShape), Arrays.toString(providedShape)));
        }
        
        this.weights = weights.copy();
    }
    
    /**
     * Gets the bias values used by this layer.
     * 
     * @return The bias tensor
     */
    public Tensor getBias() {
        return bias.copy();
    }
    
    /**
     * Sets the bias values for this layer.
     * 
     * @param bias The bias tensor (must match expected dimensions)
     * @throws IllegalArgumentException if bias dimensions don't match the layer structure
     */
    public void setBias(Tensor bias) {
        if (bias == null) {
            throw new IllegalArgumentException("Bias tensor cannot be null");
        }
        
        int[] expectedShape = this.bias.getShape();
        int[] providedShape = bias.getShape();
        
        if (!Arrays.equals(expectedShape, providedShape)) {
            throw new IllegalArgumentException(String.format(
                    "Bias shape mismatch: expected %s, got %s",
                    Arrays.toString(expectedShape), Arrays.toString(providedShape)));
        }
        
        this.bias = bias.copy();
    }
    
    /**
     * Checks if this layer uses softmax activation.
     * 
     * @return true if softmax is used, false otherwise
     */
    public boolean usesSoftmax() {
        return useSoftmax;
    }
}