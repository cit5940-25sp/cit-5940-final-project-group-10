package deeplearningjava.layer.tensor;

import java.util.Arrays;
import java.util.Objects;
import deeplearningjava.api.TensorLayer;
import deeplearningjava.core.activation.ActivationFunction;
import deeplearningjava.core.tensor.Tensor;
import deeplearningjava.core.tensor.TensorOperations;

/**
 * Implementation of a convolutional layer for CNNs.
 * This layer applies convolution operations to input tensors.
 */
public class ConvolutionalLayer extends AbstractTensorLayer {
    
    private final int kernelHeight;
    private final int kernelWidth;
    private final int[] stride;
    private final boolean padding;
    private final int inChannels;
    private final int outChannels;
    
    // Weights and biases
    private Tensor kernels; // Shape: [outChannels, inChannels, kernelHeight, kernelWidth]
    private Tensor bias;    // Shape: [outChannels]
    
    // For backpropagation
    private Tensor lastInput;
    private Tensor lastOutput;
    private Tensor kernelGradients;
    private Tensor biasGradients;
    
    /**
     * Creates a new convolutional layer.
     * 
     * @param inputShape The shape of the input tensor [batch, channels, height, width]
     * @param kernelSize The size of the convolution kernel [height, width]
     * @param outChannels The number of output channels (filters)
     * @param stride The stride of the convolution [vertical, horizontal]
     * @param padding Whether to use same padding
     * @param activationFunction The activation function to use
     */
    public ConvolutionalLayer(int[] inputShape, int[] kernelSize, int outChannels, 
                            int[] stride, boolean padding, ActivationFunction activationFunction) {
        super(inputShape, calculateOutputShape(inputShape, kernelSize, outChannels, stride, padding), 
              activationFunction, LayerType.CONVOLUTIONAL);
        
        if (inputShape.length != 4) {
            throw new IllegalArgumentException("Input shape must be 4D [batch, channels, height, width]");
        }
        
        if (kernelSize.length != 2) {
            throw new IllegalArgumentException("Kernel size must be 2D [height, width]");
        }
        
        if (stride.length != 2) {
            throw new IllegalArgumentException("Stride must be 2D [vertical, horizontal]");
        }
        
        this.inChannels = inputShape[1];
        this.outChannels = outChannels;
        this.kernelHeight = kernelSize[0];
        this.kernelWidth = kernelSize[1];
        this.stride = stride.clone();
        this.padding = padding;
        
        initializeParameters();
    }
    
    /**
     * Calculates the output shape for a convolutional layer.
     * 
     * @param inputShape The input shape
     * @param kernelSize The kernel size
     * @param outChannels The number of output channels
     * @param stride The stride values
     * @param padding Whether to use padding
     * @return The output shape
     */
    private static int[] calculateOutputShape(int[] inputShape, int[] kernelSize, 
                                           int outChannels, int[] stride, boolean padding) {
        int batchSize = inputShape[0];
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];
        
        int kernelHeight = kernelSize[0];
        int kernelWidth = kernelSize[1];
        
        int strideY = stride[0];
        int strideX = stride[1];
        
        int paddingY = padding ? kernelHeight / 2 : 0;
        int paddingX = padding ? kernelWidth / 2 : 0;
        
        int outputHeight = (inputHeight - kernelHeight + 2 * paddingY) / strideY + 1;
        int outputWidth = (inputWidth - kernelWidth + 2 * paddingX) / strideX + 1;
        
        return new int[] {batchSize, outChannels, outputHeight, outputWidth};
    }
    
    @Override
    public void initializeParameters() {
        // Initialize kernels with shape [outChannels, inChannels, kernelHeight, kernelWidth]
        kernels = new Tensor(outChannels, inChannels, kernelHeight, kernelWidth);
        
        // Initialize bias with shape [outChannels]
        bias = new Tensor(outChannels);
        
        // He initialization for kernels (used for ReLU-like activations)
        int fanIn = inChannels * kernelHeight * kernelWidth;
        heInitialization(kernels, fanIn);
        
        // Initialize bias to small values
        double[] biasData = bias.getData();
        Arrays.fill(biasData, 0.01);
        
        // Initialize gradients
        kernelGradients = new Tensor(outChannels, inChannels, kernelHeight, kernelWidth);
        biasGradients = new Tensor(outChannels);
    }
    
    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "Input tensor must not be null");
        
        int[] shape = input.getShape();
        if (shape.length != 4 || shape[1] != inChannels) {
            throw new IllegalArgumentException(
                    "Expected input shape [batch, " + inChannels + ", height, width], got " + 
                    formatShape(shape));
        }
        
        // Save input for backward pass
        lastInput = input.copy();
        
        // Apply convolution
        Tensor preActivation = TensorOperations.convolve(input, kernels, stride, padding);
        
        // Add bias to each output channel
        double[] preActivationData = preActivation.getData();
        int[] preActivationShape = preActivation.getShape();
        int batchSize = preActivationShape[0];
        int outputHeight = preActivationShape[2];
        int outputWidth = preActivationShape[3];
        
        // Add bias to each output channel
        for (int b = 0; b < batchSize; b++) {
            for (int oc = 0; oc < outChannels; oc++) {
                double biasValue = bias.get(oc);
                
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        double value = preActivation.get(b, oc, oh, ow);
                        preActivation.set(value + biasValue, b, oc, oh, ow);
                    }
                }
            }
        }
        
        // Apply activation function
        lastOutput = applyActivation(preActivation);
        
        return lastOutput;
    }
    
    @Override
    public Tensor backward(Tensor gradients) {
        Objects.requireNonNull(gradients, "Gradient tensor must not be null");
        
        int[] shape = gradients.getShape();
        if (!Arrays.equals(shape, outputShape)) {
            throw new IllegalArgumentException(
                    "Expected gradient shape " + formatShape(outputShape) + 
                    ", got " + formatShape(shape));
        }
        
        // Calculate the gradient of the activation function
        activationGradients = applyActivationDerivative(lastOutput);
        
        // Element-wise multiplication with incoming gradients
        Tensor outputGradients = new Tensor(gradients.getShape());
        double[] outputGradientsData = outputGradients.getData();
        double[] gradientsData = gradients.getData();
        double[] activationGradientsData = activationGradients.getData();
        
        for (int i = 0; i < outputGradientsData.length; i++) {
            outputGradientsData[i] = gradientsData[i] * activationGradientsData[i];
        }
        
        // Calculate gradients for kernels and bias
        updateKernelGradients(outputGradients);
        updateBiasGradients(outputGradients);
        
        // Calculate gradients for the input tensor
        // This requires a transposed convolution operation
        Tensor inputGradients = calculateInputGradients(outputGradients);
        
        return inputGradients;
    }
    
    private void updateKernelGradients(Tensor outputGradients) {
        // Reset kernel gradients
        kernelGradients.fill(0.0);
        
        int[] outputShape = outputGradients.getShape();
        int batchSize = outputShape[0];
        int outputHeight = outputShape[2];
        int outputWidth = outputShape[3];
        
        // For each element in the output
        for (int b = 0; b < batchSize; b++) {
            for (int oc = 0; oc < outChannels; oc++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        double gradient = outputGradients.get(b, oc, oh, ow);
                        
                        // Update gradients for all connections in the kernel
                        for (int ic = 0; ic < inChannels; ic++) {
                            for (int kh = 0; kh < kernelHeight; kh++) {
                                for (int kw = 0; kw < kernelWidth; kw++) {
                                    // Calculate corresponding input position
                                    int ih = oh * stride[0] + kh - (padding ? kernelHeight / 2 : 0);
                                    int iw = ow * stride[1] + kw - (padding ? kernelWidth / 2 : 0);
                                    
                                    // Check if input position is valid
                                    if (ih >= 0 && ih < lastInput.getShape()[2] && 
                                        iw >= 0 && iw < lastInput.getShape()[3]) {
                                        
                                        double inputValue = lastInput.get(b, ic, ih, iw);
                                        double currentGradient = kernelGradients.get(oc, ic, kh, kw);
                                        
                                        kernelGradients.set(currentGradient + inputValue * gradient, 
                                                          oc, ic, kh, kw);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    private void updateBiasGradients(Tensor outputGradients) {
        // Reset bias gradients
        biasGradients.fill(0.0);
        
        int[] outputShape = outputGradients.getShape();
        int batchSize = outputShape[0];
        int outputHeight = outputShape[2];
        int outputWidth = outputShape[3];
        
        // For each output channel
        for (int oc = 0; oc < outChannels; oc++) {
            double sum = 0.0;
            
            // Sum gradients across batch, height, and width
            for (int b = 0; b < batchSize; b++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        sum += outputGradients.get(b, oc, oh, ow);
                    }
                }
            }
            
            biasGradients.set(sum, oc);
        }
    }
    
    private Tensor calculateInputGradients(Tensor outputGradients) {
        // Create tensor for input gradients
        Tensor inputGradients = new Tensor(inputShape);
        
        int[] outputShape = outputGradients.getShape();
        int batchSize = outputShape[0];
        int outputHeight = outputShape[2];
        int outputWidth = outputShape[3];
        
        // For each element in the output gradients
        for (int b = 0; b < batchSize; b++) {
            for (int oc = 0; oc < outChannels; oc++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        double gradient = outputGradients.get(b, oc, oh, ow);
                        
                        // Distribute gradient to all connected inputs
                        for (int ic = 0; ic < inChannels; ic++) {
                            for (int kh = 0; kh < kernelHeight; kh++) {
                                for (int kw = 0; kw < kernelWidth; kw++) {
                                    // Calculate corresponding input position
                                    int ih = oh * stride[0] + kh - (padding ? kernelHeight / 2 : 0);
                                    int iw = ow * stride[1] + kw - (padding ? kernelWidth / 2 : 0);
                                    
                                    // Check if input position is valid
                                    if (ih >= 0 && ih < inputShape[2] && 
                                        iw >= 0 && iw < inputShape[3]) {
                                        
                                        double kernelValue = kernels.get(oc, ic, kh, kw);
                                        double currentGradient = inputGradients.get(b, ic, ih, iw);
                                        
                                        inputGradients.set(currentGradient + kernelValue * gradient, 
                                                          b, ic, ih, iw);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return inputGradients;
    }
    
    @Override
    public void updateParameters(double learningRate) {
        // Update kernels
        double[] kernelsData = kernels.getData();
        double[] kernelGradientsData = kernelGradients.getData();
        
        for (int i = 0; i < kernelsData.length; i++) {
            kernelsData[i] -= learningRate * kernelGradientsData[i];
        }
        
        // Update bias
        double[] biasData = bias.getData();
        double[] biasGradientsData = biasGradients.getData();
        
        for (int i = 0; i < biasData.length; i++) {
            biasData[i] -= learningRate * biasGradientsData[i];
        }
    }
    
    /**
     * Gets the kernels used by this layer.
     * 
     * @return The kernel tensor
     */
    public Tensor getKernels() {
        return kernels.copy();
    }
    
    /**
     * Sets the kernels for this layer.
     * 
     * @param kernels The kernel tensor (must match expected dimensions)
     * @throws IllegalArgumentException if kernel dimensions don't match the layer structure
     */
    public void setKernels(Tensor kernels) {
        if (kernels == null) {
            throw new IllegalArgumentException("Kernels tensor cannot be null");
        }
        
        int[] expectedShape = this.kernels.getShape();
        int[] providedShape = kernels.getShape();
        
        if (!Arrays.equals(expectedShape, providedShape)) {
            throw new IllegalArgumentException(String.format(
                    "Kernels shape mismatch: expected %s, got %s",
                    Arrays.toString(expectedShape), Arrays.toString(providedShape)));
        }
        
        this.kernels = kernels.copy();
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
     * Gets the stride values used for convolution.
     * 
     * @return The stride values
     */
    public int[] getStride() {
        return stride.clone();
    }
    
    /**
     * Checks if this layer uses padding.
     * 
     * @return true if padding is used, false otherwise
     */
    public boolean usesPadding() {
        return padding;
    }
}