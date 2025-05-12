package deeplearningjava.core.tensor;

import java.util.Arrays;

/**
 * Utility class providing operations for tensors, particularly focused on
 * operations needed for convolutional neural networks.
 */
public class TensorOperations {
    
    /**
     * Performs convolution operation between an input tensor and a kernel.
     * 
     * @param input The input tensor (e.g., image) [batch, channels, height, width]
     * @param kernel The convolution kernel [outChannels, inChannels, kernelHeight, kernelWidth]
     * @param stride The stride of the convolution (vertical and horizontal)
     * @param padding The padding to apply (same padding if true, no padding if false)
     * @return The convolved output
     */
    public static Tensor convolve(Tensor input, Tensor kernel, int[] stride, boolean padding) {
        int[] inputShape = input.getShape();
        int[] kernelShape = kernel.getShape();
        
        if (inputShape.length != 4) {
            throw new IllegalArgumentException("Input must be a 4D tensor [batch, channels, height, width]");
        }
        
        if (kernelShape.length != 4) {
            throw new IllegalArgumentException(
                    "Kernel must be a 4D tensor [outChannels, inChannels, kernelHeight, kernelWidth]");
        }
        
        if (inputShape[1] != kernelShape[1]) {
            throw new IllegalArgumentException(
                    "Input channels (" + inputShape[1] + 
                    ") must match kernel input channels (" + kernelShape[1] + ")");
        }
        
        int batchSize = inputShape[0];
        int inputChannels = inputShape[1];
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];
        
        int outputChannels = kernelShape[0];
        int kernelHeight = kernelShape[2];
        int kernelWidth = kernelShape[3];
        
        int strideY = stride[0];
        int strideX = stride[1];
        
        int paddingY = padding ? kernelHeight / 2 : 0;
        int paddingX = padding ? kernelWidth / 2 : 0;
        
        int outputHeight = (inputHeight - kernelHeight + 2 * paddingY) / strideY + 1;
        int outputWidth = (inputWidth - kernelWidth + 2 * paddingX) / strideX + 1;
        
        Tensor output = new Tensor(batchSize, outputChannels, outputHeight, outputWidth);
        
        // Perform convolution
        for (int b = 0; b < batchSize; b++) {
            for (int oc = 0; oc < outputChannels; oc++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        double sum = 0.0;
                        
                        for (int ic = 0; ic < inputChannels; ic++) {
                            for (int kh = 0; kh < kernelHeight; kh++) {
                                for (int kw = 0; kw < kernelWidth; kw++) {
                                    int ih = oh * strideY + kh - paddingY;
                                    int iw = ow * strideX + kw - paddingX;
                                    
                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                        sum += input.get(b, ic, ih, iw) * kernel.get(oc, ic, kh, kw);
                                    }
                                }
                            }
                        }
                        
                        output.set(sum, b, oc, oh, ow);
                    }
                }
            }
        }
        
        return output;
    }
    
    /**
     * Performs max pooling operation on an input tensor.
     * 
     * @param input The input tensor [batch, channels, height, width]
     * @param poolSize The size of the pooling window [height, width]
     * @param stride The stride of the pooling [vertical, horizontal]
     * @return The pooled output
     */
    public static Tensor maxPool(Tensor input, int[] poolSize, int[] stride) {
        int[] inputShape = input.getShape();
        
        if (inputShape.length != 4) {
            throw new IllegalArgumentException("Input must be a 4D tensor [batch, channels, height, width]");
        }
        
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
        
        Tensor output = new Tensor(batchSize, channels, outputHeight, outputWidth);
        
        // Perform max pooling
        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        double maxVal = Double.NEGATIVE_INFINITY;
                        
                        for (int ph = 0; ph < poolHeight; ph++) {
                            for (int pw = 0; pw < poolWidth; pw++) {
                                int ih = oh * strideY + ph;
                                int iw = ow * strideX + pw;
                                
                                if (ih < inputHeight && iw < inputWidth) {
                                    maxVal = Math.max(maxVal, input.get(b, c, ih, iw));
                                }
                            }
                        }
                        
                        output.set(maxVal, b, c, oh, ow);
                    }
                }
            }
        }
        
        return output;
    }
    
    /**
     * Performs average pooling operation on an input tensor.
     * 
     * @param input The input tensor [batch, channels, height, width]
     * @param poolSize The size of the pooling window [height, width]
     * @param stride The stride of the pooling [vertical, horizontal]
     * @return The pooled output
     */
    public static Tensor avgPool(Tensor input, int[] poolSize, int[] stride) {
        int[] inputShape = input.getShape();
        
        if (inputShape.length != 4) {
            throw new IllegalArgumentException("Input must be a 4D tensor [batch, channels, height, width]");
        }
        
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
        
        Tensor output = new Tensor(batchSize, channels, outputHeight, outputWidth);
        
        // Perform average pooling
        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        double sum = 0.0;
                        int count = 0;
                        
                        for (int ph = 0; ph < poolHeight; ph++) {
                            for (int pw = 0; pw < poolWidth; pw++) {
                                int ih = oh * strideY + ph;
                                int iw = ow * strideX + pw;
                                
                                if (ih < inputHeight && iw < inputWidth) {
                                    sum += input.get(b, c, ih, iw);
                                    count++;
                                }
                            }
                        }
                        
                        output.set(sum / count, b, c, oh, ow);
                    }
                }
            }
        }
        
        return output;
    }
    
    /**
     * Converts a tensor to a 1D array by flattening it.
     * This is useful for connecting convolutional layers to fully connected layers.
     * 
     * @param input The input tensor
     * @return A 1D array containing the flattened tensor
     */
    public static double[] flatten(Tensor input) {
        return input.getData();
    }
    
    /**
     * Reshapes a tensor to a new shape.
     * The total size of the tensor must remain the same.
     * 
     * @param input The input tensor
     * @param newShape The new shape
     * @return A new tensor with the specified shape
     * @throws IllegalArgumentException if the new shape has a different total size
     */
    public static Tensor reshape(Tensor input, int... newShape) {
        int oldSize = input.getSize();
        int newSize = 1;
        
        for (int dim : newShape) {
            newSize *= dim;
        }
        
        if (oldSize != newSize) {
            throw new IllegalArgumentException(
                    "Cannot reshape tensor of size " + oldSize + 
                    " to new shape with size " + newSize);
        }
        
        return new Tensor(input.getData(), newShape);
    }
    
    /**
     * Adds a channel dimension to a tensor.
     * 
     * @param input The input tensor
     * @param position The position to add the dimension (0 for batch, 1 for channel)
     * @return A new tensor with an added dimension
     */
    public static Tensor expandDims(Tensor input, int position) {
        int[] oldShape = input.getShape();
        int[] newShape = new int[oldShape.length + 1];
        
        for (int i = 0; i < position; i++) {
            newShape[i] = oldShape[i];
        }
        
        newShape[position] = 1;
        
        for (int i = position; i < oldShape.length; i++) {
            newShape[i + 1] = oldShape[i];
        }
        
        return new Tensor(input.getData(), newShape);
    }
    
    /**
     * Performs element-wise addition of two tensors.
     * 
     * @param a The first tensor
     * @param b The second tensor
     * @return A new tensor containing the sum
     * @throws IllegalArgumentException if the shapes don't match
     */
    public static Tensor add(Tensor a, Tensor b) {
        int[] shapeA = a.getShape();
        int[] shapeB = b.getShape();
        
        if (!Arrays.equals(shapeA, shapeB)) {
            throw new IllegalArgumentException(
                    "Tensor shapes must match for addition: " + 
                    Arrays.toString(shapeA) + " vs " + Arrays.toString(shapeB));
        }
        
        double[] dataA = a.getData();
        double[] dataB = b.getData();
        double[] result = new double[dataA.length];
        
        for (int i = 0; i < dataA.length; i++) {
            result[i] = dataA[i] + dataB[i];
        }
        
        return new Tensor(result, shapeA);
    }
    
    /**
     * Transposes specific dimensions of a tensor.
     * 
     * @param input The input tensor
     * @param dims The new order of dimensions
     * @return A new tensor with transposed dimensions
     * @throws IllegalArgumentException if the dimensions array is invalid
     */
    public static Tensor transpose(Tensor input, int... dims) {
        int[] shape = input.getShape();
        
        if (dims.length != shape.length) {
            throw new IllegalArgumentException(
                    "Dimensions array must have the same length as tensor rank");
        }
        
        // Check if dims contains all dimensions exactly once
        boolean[] used = new boolean[shape.length];
        for (int d : dims) {
            if (d < 0 || d >= shape.length || used[d]) {
                throw new IllegalArgumentException(
                        "Invalid dimensions array: " + Arrays.toString(dims));
            }
            used[d] = true;
        }
        
        // Create new shape
        int[] newShape = new int[shape.length];
        for (int i = 0; i < shape.length; i++) {
            newShape[i] = shape[dims[i]];
        }
        
        Tensor result = new Tensor(newShape);
        
        // Transpose the data
        int[] indices = new int[shape.length];
        int[] newIndices = new int[shape.length];
        
        // Recursive function to traverse all elements
        transposeHelper(input, result, shape, dims, indices, newIndices, 0);
        
        return result;
    }
    
    private static void transposeHelper(Tensor input, Tensor output, int[] shape,
            int[] dims, int[] indices, int[] newIndices, int dim) {
        
        if (dim == shape.length) {
            // We've determined all indices, copy the value
            double value = input.get(indices);
            output.set(value, newIndices);
            return;
        }
        
        for (int i = 0; i < shape[dim]; i++) {
            indices[dim] = i;
            newIndices[dims[dim]] = i;
            transposeHelper(input, output, shape, dims, indices, newIndices, dim + 1);
        }
    }
}