package deeplearningjava.core.tensor;

import java.util.Arrays;
import java.util.Objects;

/**
 * Represents a multi-dimensional array (tensor) for neural network operations.
 * This class is essential for convolutional neural networks which operate on
 * multi-dimensional data such as images (3D tensors).
 */
public class Tensor {
    private final double[] data;
    private final int[] shape;
    private final int[] strides;
    
    /**
     * Creates a new tensor with the given shape and initializes all elements to zero.
     * 
     * @param shape The shape of the tensor (dimensions)
     */
    public Tensor(int... shape) {
        if (shape.length == 0) {
            throw new IllegalArgumentException("Tensor must have at least one dimension");
        }
        
        for (int dim : shape) {
            if (dim <= 0) {
                throw new IllegalArgumentException("All dimensions must be positive");
            }
        }
        
        this.shape = shape.clone();
        this.strides = computeStrides(shape);
        int size = computeSize(shape);
        this.data = new double[size];
    }
    
    /**
     * Creates a new tensor with the given shape and data.
     * 
     * @param data The data to fill the tensor with
     * @param shape The shape of the tensor (dimensions)
     * @throws IllegalArgumentException if the data length doesn't match the product of shape dimensions
     */
    public Tensor(double[] data, int... shape) {
        if (shape.length == 0) {
            throw new IllegalArgumentException("Tensor must have at least one dimension");
        }
        
        for (int dim : shape) {
            if (dim <= 0) {
                throw new IllegalArgumentException("All dimensions must be positive");
            }
        }
        
        this.shape = shape.clone();
        this.strides = computeStrides(shape);
        int size = computeSize(shape);
        
        if (data.length != size) {
            throw new IllegalArgumentException(
                    "Data length " + data.length + 
                    " doesn't match the shape size " + size);
        }
        
        this.data = data.clone();
    }
    
    /**
     * Computes the strides for each dimension of the tensor.
     * 
     * @param shape The shape of the tensor
     * @return The strides for each dimension
     */
    private int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        
        return strides;
    }
    
    /**
     * Computes the total size (number of elements) of the tensor.
     * 
     * @param shape The shape of the tensor
     * @return The total number of elements
     */
    private int computeSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }
    
    /**
     * Gets the value at the specified indices.
     * 
     * @param indices The indices in each dimension
     * @return The value at the specified indices
     * @throws IllegalArgumentException if the number of indices doesn't match the tensor's dimensionality
     * @throws IndexOutOfBoundsException if any index is out of bounds
     */
    public double get(int... indices) {
        Objects.requireNonNull(indices, "Indices must not be null");
        
        if (indices.length != shape.length) {
            throw new IllegalArgumentException(
                    "Number of indices (" + indices.length + 
                    ") doesn't match tensor dimensionality (" + shape.length + ")");
        }
        
        int index = 0;
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException(
                        "Index " + indices[i] + " is out of bounds for dimension " + 
                        i + " with size " + shape[i]);
            }
            index += indices[i] * strides[i];
        }
        
        return data[index];
    }
    
    /**
     * Sets the value at the specified indices.
     * 
     * @param value The value to set
     * @param indices The indices in each dimension
     * @throws IllegalArgumentException if the number of indices doesn't match the tensor's dimensionality
     * @throws IndexOutOfBoundsException if any index is out of bounds
     */
    public void set(double value, int... indices) {
        Objects.requireNonNull(indices, "Indices must not be null");
        
        if (indices.length != shape.length) {
            throw new IllegalArgumentException(
                    "Number of indices (" + indices.length + 
                    ") doesn't match tensor dimensionality (" + shape.length + ")");
        }
        
        int index = 0;
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException(
                        "Index " + indices[i] + " is out of bounds for dimension " + 
                        i + " with size " + shape[i]);
            }
            index += indices[i] * strides[i];
        }
        
        data[index] = value;
    }
    
    /**
     * Gets the raw data array of the tensor.
     * 
     * @return The raw data array
     */
    public double[] getData() {
        return data.clone();
    }
    
    /**
     * Gets the shape of the tensor.
     * 
     * @return The shape array
     */
    public int[] getShape() {
        return shape.clone();
    }
    
    /**
     * Gets the strides of the tensor.
     * 
     * @return The strides array
     */
    public int[] getStrides() {
        return strides.clone();
    }
    
    /**
     * Gets the number of dimensions of the tensor.
     * 
     * @return The number of dimensions
     */
    public int getRank() {
        return shape.length;
    }
    
    /**
     * Gets the total number of elements in the tensor.
     * 
     * @return The total number of elements
     */
    public int getSize() {
        return data.length;
    }
    
    /**
     * Creates a copy of this tensor.
     * 
     * @return A new tensor with the same data and shape
     */
    public Tensor copy() {
        return new Tensor(data.clone(), shape.clone());
    }
    
    /**
     * Fills the tensor with the specified value.
     * 
     * @param value The value to fill the tensor with
     * @return This tensor for method chaining
     */
    public Tensor fill(double value) {
        Arrays.fill(data, value);
        return this;
    }
    
    /**
     * Applies a function to each element of the tensor.
     * 
     * @param function The function to apply
     * @return A new tensor with the function applied to each element
     */
    public Tensor map(TensorFunction function) {
        double[] newData = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            newData[i] = function.apply(data[i]);
        }
        return new Tensor(newData, shape.clone());
    }
    
    /**
     * Creates a string representation of the tensor.
     * 
     * @return A string representation of the tensor
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor(shape=").append(Arrays.toString(shape));
        sb.append(", data=").append(Arrays.toString(data)).append(")");
        return sb.toString();
    }
    
    /**
     * Functional interface for tensor operations.
     */
    @FunctionalInterface
    public interface TensorFunction {
        /**
         * Applies a function to a single value.
         * 
         * @param value The input value
         * @return The output value
         */
        double apply(double value);
    }
}