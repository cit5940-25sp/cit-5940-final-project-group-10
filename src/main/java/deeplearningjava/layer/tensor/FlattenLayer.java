package deeplearningjava.layer.tensor;

import java.util.Arrays;
import java.util.Objects;
import deeplearningjava.api.TensorLayer;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.core.tensor.Tensor;
import deeplearningjava.core.tensor.TensorOperations;

/**
 * A layer that flattens an input tensor to a 1D tensor.
 * This is typically used to connect convolutional layers to fully connected layers.
 */
public class FlattenLayer extends AbstractTensorLayer {
    
    private int[] inputDimensions;
    private int outputSize;
    
    // For backpropagation
    private Tensor lastInput;
    
    /**
     * Creates a new flatten layer.
     * 
     * @param inputShape The shape of the input tensor
     */
    public FlattenLayer(int[] inputShape) {
        super(inputShape, calculateOutputShape(inputShape), 
              ActivationFunctions.linear(), LayerType.FLATTENING);
        
        this.inputDimensions = inputShape.clone();
        
        // Calculate the total size of the flattened tensor
        int size = 1;
        for (int dim : inputShape) {
            size *= dim;
        }
        this.outputSize = size;
    }
    
    /**
     * Calculates the output shape for a flatten layer.
     * 
     * @param inputShape The input shape
     * @return The output shape (a 1D tensor)
     */
    private static int[] calculateOutputShape(int[] inputShape) {
        int size = 1;
        for (int dim : inputShape) {
            size *= dim;
        }
        return new int[] {size};
    }
    
    @Override
    public void initializeParameters() {
        // Flatten layers have no trainable parameters
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
        
        // Flatten the tensor
        double[] flatData = input.getData();
        return new Tensor(flatData, outputSize);
    }
    
    @Override
    public Tensor backward(Tensor gradients) {
        Objects.requireNonNull(gradients, "Gradient tensor must not be null");
        
        int[] shape = gradients.getShape();
        if (shape.length != 1 || shape[0] != outputSize) {
            throw new IllegalArgumentException(
                    "Expected gradient shape [" + outputSize + "], got " + formatShape(shape));
        }
        
        // Reshape the gradients to match the input shape
        return TensorOperations.reshape(gradients, inputDimensions);
    }
    
    @Override
    public void updateParameters(double learningRate) {
        // Flatten layers have no trainable parameters
    }
    
    /**
     * Gets the input dimensions before flattening.
     * 
     * @return The input dimensions
     */
    public int[] getInputDimensions() {
        return inputDimensions.clone();
    }
    
    /**
     * Gets the output size after flattening.
     * 
     * @return The output size
     */
    public int getOutputSize() {
        return outputSize;
    }
    
    /**
     * Checks if the shapes are compatible for connection.
     * Flatten layer can accept any input shape, as long as the total size matches.
     */
    @Override
    protected boolean isShapeCompatible(int[] outputShape, int[] inputShape) {
        // For flatten layer, we just need to make sure the total size matches
        int outputSize = 1;
        for (int dim : outputShape) {
            outputSize *= dim;
        }
        
        if (inputShape.length != 1) {
            return false;
        }
        
        return inputShape[0] == outputSize;
    }
}