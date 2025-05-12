package deeplearningjava.layer.tensor;

import java.util.Arrays;
import java.util.Objects;
import deeplearningjava.api.TensorLayer;
import deeplearningjava.core.activation.ActivationFunction;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.core.tensor.Tensor;
import deeplearningjava.core.tensor.TensorOperations;

/**
 * Implementation of a pooling layer for CNNs.
 * This layer performs down-sampling operations (max pooling or average pooling).
 */
public class PoolingLayer extends AbstractTensorLayer {
    
    /**
     * Enum defining the available pooling types.
     */
    public enum PoolingType {
        MAX,
        AVERAGE
    }
    
    private final int[] poolSize;
    private final int[] stride;
    private final PoolingType poolingType;
    
    // For backpropagation
    private Tensor lastInput;
    private Tensor lastOutput;
    private int[][] maxIndices; // For max pooling (store indices of max values)
    
    /**
     * Creates a new pooling layer.
     * 
     * @param inputShape The shape of input tensors [batch, channels, height, width]
     * @param poolSize The size of the pooling window [height, width]
     * @param stride The stride for pooling [vertical, horizontal]
     * @param poolingType The type of pooling to perform
     */
    public PoolingLayer(int[] inputShape, int[] poolSize, int[] stride, PoolingType poolingType) {
        super(inputShape, calculateOutputShape(inputShape, poolSize, stride), 
              ActivationFunctions.linear(), LayerType.POOLING);
        
        if (inputShape.length != 4) {
            throw new IllegalArgumentException("Input shape must be 4D [batch, channels, height, width]");
        }
        
        if (poolSize.length != 2) {
            throw new IllegalArgumentException("Pool size must be 2D [height, width]");
        }
        
        if (stride.length != 2) {
            throw new IllegalArgumentException("Stride must be 2D [vertical, horizontal]");
        }
        
        this.poolSize = poolSize.clone();
        this.stride = stride.clone();
        this.poolingType = Objects.requireNonNull(poolingType, "poolingType must not be null");
        
        // No parameters to initialize for pooling layers
    }
    
    /**
     * Calculates the output shape for a pooling layer.
     * 
     * @param inputShape The input shape
     * @param poolSize The pooling window size
     * @param stride The stride values
     * @return The output shape
     */
    private static int[] calculateOutputShape(int[] inputShape, int[] poolSize, int[] stride) {
        int batchSize = inputShape[0];
        int channels = inputShape[1];
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];
        
        int poolHeight = poolSize[0];
        int poolWidth = poolSize[1];
        
        int strideY = stride[0];
        int strideX = stride[1];
        
        int outputHeight = (inputHeight - poolHeight) / strideY + 1;
        int outputWidth = (inputWidth - poolWidth) / strideX + 1;
        
        return new int[] {batchSize, channels, outputHeight, outputWidth};
    }
    
    @Override
    public void initializeParameters() {
        // Pooling layers have no trainable parameters
    }
    
    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "Input tensor must not be null");
        
        int[] shape = input.getShape();
        if (!Arrays.equals(shape, inputShape)) {
            throw new IllegalArgumentException(
                    "Expected input shape " + formatShape(inputShape) + 
                    ", got " + formatShape(shape));
        }
        
        // Save input for backward pass
        lastInput = input.copy();
        
        // Perform pooling
        Tensor output;
        if (poolingType == PoolingType.MAX) {
            output = performMaxPooling(input);
        } else {
            output = performAveragePooling(input);
        }
        
        // Save output for backward pass
        lastOutput = output.copy();
        
        return output;
    }
    
    private Tensor performMaxPooling(Tensor input) {
        int[] inputShape = input.getShape();
        int batchSize = inputShape[0];
        int channels = inputShape[1];
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];
        
        int poolHeight = poolSize[0];
        int poolWidth = poolSize[1];
        
        int strideY = stride[0];
        int strideX = stride[1];
        
        int outputHeight = outputShape[2];
        int outputWidth = outputShape[3];
        
        Tensor output = new Tensor(outputShape);
        
        // Initialize max indices storage
        int maxIndicesSize = batchSize * channels * outputHeight * outputWidth * 2;
        maxIndices = new int[maxIndicesSize][2];
        
        // Perform max pooling
        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        double maxVal = Double.NEGATIVE_INFINITY;
                        int maxH = -1, maxW = -1;
                        
                        for (int ph = 0; ph < poolHeight; ph++) {
                            for (int pw = 0; pw < poolWidth; pw++) {
                                int ih = oh * strideY + ph;
                                int iw = ow * strideX + pw;
                                
                                if (ih < inputHeight && iw < inputWidth) {
                                    double val = input.get(b, c, ih, iw);
                                    if (val > maxVal) {
                                        maxVal = val;
                                        maxH = ih;
                                        maxW = iw;
                                    }
                                }
                            }
                        }
                        
                        output.set(maxVal, b, c, oh, ow);
                        
                        // Store the indices of the max value for backpropagation
                        int idx = ((b * channels + c) * outputHeight + oh) * outputWidth + ow;
                        maxIndices[idx][0] = maxH;
                        maxIndices[idx][1] = maxW;
                    }
                }
            }
        }
        
        return output;
    }
    
    private Tensor performAveragePooling(Tensor input) {
        return TensorOperations.avgPool(input, poolSize, stride);
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
        
        // Create tensor for input gradients
        Tensor inputGradients = new Tensor(inputShape);
        
        int[] gradientShape = gradients.getShape();
        int batchSize = gradientShape[0];
        int channels = gradientShape[1];
        int outputHeight = gradientShape[2];
        int outputWidth = gradientShape[3];
        
        if (poolingType == PoolingType.MAX) {
            // For max pooling, pass gradient only to the max value
            for (int b = 0; b < batchSize; b++) {
                for (int c = 0; c < channels; c++) {
                    for (int oh = 0; oh < outputHeight; oh++) {
                        for (int ow = 0; ow < outputWidth; ow++) {
                            double gradient = gradients.get(b, c, oh, ow);
                            
                            // Get the indices of the max value
                            int idx = ((b * channels + c) * outputHeight + oh) * outputWidth + ow;
                            int maxH = maxIndices[idx][0];
                            int maxW = maxIndices[idx][1];
                            
                            // Pass the gradient to the max value
                            double currentGradient = inputGradients.get(b, c, maxH, maxW);
                            inputGradients.set(currentGradient + gradient, b, c, maxH, maxW);
                        }
                    }
                }
            }
        } else {
            // For average pooling, distribute gradient evenly
            int poolHeight = poolSize[0];
            int poolWidth = poolSize[1];
            int strideY = stride[0];
            int strideX = stride[1];
            
            for (int b = 0; b < batchSize; b++) {
                for (int c = 0; c < channels; c++) {
                    for (int oh = 0; oh < outputHeight; oh++) {
                        for (int ow = 0; ow < outputWidth; ow++) {
                            double gradient = gradients.get(b, c, oh, ow);
                            
                            // Count valid positions
                            int count = 0;
                            for (int ph = 0; ph < poolHeight; ph++) {
                                for (int pw = 0; pw < poolWidth; pw++) {
                                    int ih = oh * strideY + ph;
                                    int iw = ow * strideX + pw;
                                    
                                    if (ih < inputShape[2] && iw < inputShape[3]) {
                                        count++;
                                    }
                                }
                            }
                            
                            // Distribute gradient
                            double distributedGradient = gradient / count;
                            for (int ph = 0; ph < poolHeight; ph++) {
                                for (int pw = 0; pw < poolWidth; pw++) {
                                    int ih = oh * strideY + ph;
                                    int iw = ow * strideX + pw;
                                    
                                    if (ih < inputShape[2] && iw < inputShape[3]) {
                                        double currentGradient = inputGradients.get(b, c, ih, iw);
                                        inputGradients.set(currentGradient + distributedGradient, 
                                                         b, c, ih, iw);
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
        // Pooling layers have no trainable parameters
    }
    
    /**
     * Gets the pool size used by this layer.
     * 
     * @return The pool size
     */
    public int[] getPoolSize() {
        return poolSize.clone();
    }
    
    /**
     * Gets the stride values used for pooling.
     * 
     * @return The stride values
     */
    public int[] getStride() {
        return stride.clone();
    }
    
    /**
     * Gets the pooling type used by this layer.
     * 
     * @return The pooling type
     */
    public PoolingType getPoolingType() {
        return poolingType;
    }
}