package deeplearningjava.layer.tensor;

import java.util.Objects;
import java.util.Random;
import deeplearningjava.api.TensorLayer;
import deeplearningjava.core.activation.ActivationFunction;
import deeplearningjava.core.tensor.Tensor;

/**
 * Abstract base class for layers that operate on tensor data.
 * Provides common functionality for different tensor layer implementations.
 */
public abstract class AbstractTensorLayer implements TensorLayer {
    
    protected int[] inputShape;
    protected int[] outputShape;
    protected TensorLayer nextLayer;
    protected ActivationFunction activationFunction;
    protected Tensor activationGradients;
    protected final LayerType type;
    protected final Random random;
    protected static final double WEIGHT_INIT_RANGE = 0.1;
    
    /**
     * Creates a new tensor layer with the specified input and output shapes.
     * 
     * @param inputShape The shape of input tensors
     * @param outputShape The shape of output tensors
     * @param activationFunction The activation function to use
     * @param type The layer type
     */
    protected AbstractTensorLayer(int[] inputShape, int[] outputShape, 
                                 ActivationFunction activationFunction, LayerType type) {
        this.inputShape = Objects.requireNonNull(inputShape, "inputShape must not be null").clone();
        this.outputShape = Objects.requireNonNull(outputShape, "outputShape must not be null").clone();
        this.activationFunction = Objects.requireNonNull(activationFunction, 
                                                       "activationFunction must not be null");
        this.type = Objects.requireNonNull(type, "type must not be null");
        this.random = new Random();
    }
    
    @Override
    public int[] getInputShape() {
        return inputShape.clone();
    }
    
    @Override
    public int[] getOutputShape() {
        return outputShape.clone();
    }
    
    @Override
    public LayerType getType() {
        return type;
    }
    
    @Override
    public void connectTo(TensorLayer nextLayer) {
        this.nextLayer = Objects.requireNonNull(nextLayer, "nextLayer must not be null");
        
        // Verify that shapes are compatible
        if (!isShapeCompatible(this.outputShape, nextLayer.getInputShape())) {
            throw new IllegalArgumentException(
                    "Output shape " + formatShape(this.outputShape) + 
                    " is not compatible with next layer's input shape " + 
                    formatShape(nextLayer.getInputShape()));
        }
    }
    
    /**
     * Checks if two shapes are compatible for connection.
     * 
     * @param outputShape The output shape of this layer
     * @param inputShape The input shape of the next layer
     * @return true if shapes are compatible, false otherwise
     */
    protected boolean isShapeCompatible(int[] outputShape, int[] inputShape) {
        // Default implementation checks if shapes are exactly the same
        if (outputShape.length != inputShape.length) {
            return false;
        }
        
        for (int i = 0; i < outputShape.length; i++) {
            if (outputShape[i] != inputShape[i]) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Formats a shape array as a string.
     * 
     * @param shape The shape array
     * @return A string representation of the shape
     */
    protected String formatShape(int[] shape) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) {
            sb.append(shape[i]);
            if (i < shape.length - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }
    
    /**
     * Initializes weights using He initialization.
     * 
     * @param tensor The tensor to initialize
     * @param fanIn The number of input connections
     */
    protected void heInitialization(Tensor tensor, int fanIn) {
        double[] data = tensor.getData();
        double stdDev = Math.sqrt(2.0 / fanIn);
        
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextGaussian() * stdDev;
        }
    }
    
    /**
     * Initializes weights using Xavier initialization.
     * 
     * @param tensor The tensor to initialize
     * @param fanIn The number of input connections
     * @param fanOut The number of output connections
     */
    protected void xavierInitialization(Tensor tensor, int fanIn, int fanOut) {
        double[] data = tensor.getData();
        double stdDev = Math.sqrt(2.0 / (fanIn + fanOut));
        
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextGaussian() * stdDev;
        }
    }
    
    /**
     * Applies the activation function to a tensor.
     * 
     * @param input The input tensor
     * @return The activated output tensor
     */
    protected Tensor applyActivation(Tensor input) {
        return input.map(activationFunction::apply);
    }
    
    /**
     * Applies the derivative of the activation function to a tensor.
     * 
     * @param input The input tensor
     * @return The tensor with derivatives
     */
    protected Tensor applyActivationDerivative(Tensor input) {
        return input.map(activationFunction::derivative);
    }
}