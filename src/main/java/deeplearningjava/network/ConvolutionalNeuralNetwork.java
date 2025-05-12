package deeplearningjava.network;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import deeplearningjava.api.ConvolutionalNetwork;
import deeplearningjava.api.TensorLayer;
import deeplearningjava.api.TensorLayer.LayerType;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.core.tensor.Tensor;
import deeplearningjava.layer.tensor.ConvolutionalLayer;
import deeplearningjava.layer.tensor.FlattenLayer;
import deeplearningjava.layer.tensor.FullyConnectedLayer;
import deeplearningjava.layer.tensor.PoolingLayer;
import deeplearningjava.layer.tensor.PoolingLayer.PoolingType;

/**
 * Implementation of a convolutional neural network.
 * This class manages the layers of a CNN and handles forward and backward propagation.
 */
public class ConvolutionalNeuralNetwork extends AbstractNetwork implements ConvolutionalNetwork {
    
    protected final List<TensorLayer> layers;
    
    /**
     * Creates an empty convolutional neural network.
     */
    public ConvolutionalNeuralNetwork() {
        super(NetworkType.CONVOLUTIONAL);
        this.layers = new ArrayList<>();
    }
    
    /**
     * Creates a convolutional neural network with the given layers.
     * 
     * @param layers The layers for this network
     */
    public ConvolutionalNeuralNetwork(List<TensorLayer> layers) {
        super(NetworkType.CONVOLUTIONAL);
        this.layers = new ArrayList<>(Objects.requireNonNull(layers, "layers must not be null"));
        
        // Connect the layers
        for (int i = 0; i < layers.size() - 1; i++) {
            layers.get(i).connectTo(layers.get(i + 1));
        }
    }
    
    @Override
    public void addTensorLayer(TensorLayer layer) {
        Objects.requireNonNull(layer, "layer must not be null");
        
        if (!layers.isEmpty()) {
            TensorLayer lastLayer = layers.get(layers.size() - 1);
            lastLayer.connectTo(layer);
        }
        
        layers.add(layer);
    }
    
    @Override
    public List<TensorLayer> getTensorLayers() {
        return Collections.unmodifiableList(layers);
    }
    
    @Override
    public int[] getInputShape() {
        validateNetwork();
        return layers.get(0).getInputShape();
    }
    
    @Override
    public int[] getOutputShape() {
        validateNetwork();
        return layers.get(layers.size() - 1).getOutputShape();
    }
    
    @Override
    public Tensor forward(Tensor input) {
        validateNetwork();
        
        Tensor output = input;
        for (TensorLayer layer : layers) {
            output = layer.forward(output);
        }
        
        return output;
    }
    
    @Override
    public Tensor train(Tensor input, Tensor target) {
        // Forward pass
        Tensor output = forward(input);
        
        // Calculate output gradients
        Tensor gradients = calculateOutputGradients(output, target);
        
        // Backward pass
        for (int i = layers.size() - 1; i >= 0; i--) {
            gradients = layers.get(i).backward(gradients);
        }
        
        // Update parameters
        for (TensorLayer layer : layers) {
            layer.updateParameters(learningRate);
        }
        
        return output;
    }
    
    @Override
    public double trainBatch(Tensor[] inputs, Tensor[] targets, int epochs) {
        validateBatchParameters(inputs, targets, epochs);
        
        double totalLoss = 0;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochLoss = 0;
            
            for (int i = 0; i < inputs.length; i++) {
                Tensor output = train(inputs[i], targets[i]);
                epochLoss += calculateMSELoss(output, targets[i]);
            }
            
            totalLoss = epochLoss / inputs.length;
            System.out.printf("Epoch %d/%d - Loss: %.6f%n", epoch + 1, epochs, totalLoss);
        }
        
        return totalLoss;
    }
    
    /**
     * Calculates the Mean Squared Error loss between output and target tensors.
     * 
     * @param output The output tensor
     * @param target The target tensor
     * @return The MSE loss value
     */
    protected double calculateMSELoss(Tensor output, Tensor target) {
        int[] outputShape = output.getShape();
        int[] targetShape = target.getShape();
        
        if (!Arrays.equals(outputShape, targetShape)) {
            throw new IllegalArgumentException(
                    "Output shape " + formatShape(outputShape) + 
                    " doesn't match target shape " + formatShape(targetShape));
        }
        
        double[] outputData = output.getData();
        double[] targetData = target.getData();
        double sumSquaredError = 0.0;
        
        for (int i = 0; i < outputData.length; i++) {
            double error = outputData[i] - targetData[i];
            sumSquaredError += error * error;
        }
        
        return sumSquaredError / outputData.length;
    }
    
    /**
     * Calculates the gradients for the output layer.
     * 
     * @param output The output tensor
     * @param target The target tensor
     * @return The gradients for the output layer
     */
    protected Tensor calculateOutputGradients(Tensor output, Tensor target) {
        int[] outputShape = output.getShape();
        int[] targetShape = target.getShape();
        
        if (!Arrays.equals(outputShape, targetShape)) {
            throw new IllegalArgumentException(
                    "Output shape " + formatShape(outputShape) + 
                    " doesn't match target shape " + formatShape(targetShape));
        }
        
        // Calculate MSE loss gradient: 2 * (output - target) / batchSize
        double[] outputData = output.getData();
        double[] targetData = target.getData();
        double[] gradients = new double[outputData.length];
        
        int batchSize = determineBatchSize(outputShape);
        
        for (int i = 0; i < outputData.length; i++) {
            gradients[i] = 2 * (outputData[i] - targetData[i]) / batchSize;
        }
        
        return new Tensor(gradients, outputShape);
    }
    
    /**
     * Determines the batch size from the output shape.
     * 
     * @param shape The shape array
     * @return The batch size
     */
    private int determineBatchSize(int[] shape) {
        if (shape.length > 0 && shape[0] > 0) {
            return shape[0];
        }
        return 1; // Default batch size
    }
    
    /**
     * Formats a shape array as a string.
     * 
     * @param shape The shape array
     * @return A string representation of the shape
     */
    private String formatShape(int[] shape) {
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
    
    @Override
    protected void appendDetails(StringBuilder summary) {
        if (!isInitialized()) {
            summary.append("Network is empty");
            return;
        }
        
        int[] inputShape = getInputShape();
        int[] outputShape = getOutputShape();
        
        summary.append("Input Shape: ").append(formatShape(inputShape)).append("\n");
        summary.append("Output Shape: ").append(formatShape(outputShape)).append("\n");
        summary.append("Layer Types: ");
        
        for (int i = 0; i < layers.size(); i++) {
            summary.append(layers.get(i).getType());
            if (i < layers.size() - 1) {
                summary.append(" -> ");
            }
        }
    }
    
    /**
     * Creates a simple CNN for image classification.
     * 
     * @param inputShape The input shape [batch, channels, height, width]
     * @param numClasses The number of output classes
     * @return A convolutional neural network
     */
    public static ConvolutionalNeuralNetwork createSimpleImageClassifier(
            int[] inputShape, int numClasses) {
        
        if (inputShape.length != 4) {
            throw new IllegalArgumentException(
                    "Input shape must be 4D [batch, channels, height, width]");
        }
        
        ConvolutionalNeuralNetwork network = new ConvolutionalNeuralNetwork();
        
        // First convolutional layer
        network.addTensorLayer(new ConvolutionalLayer(
                inputShape,
                new int[]{3, 3},   // 3x3 kernel
                16,                // 16 output channels
                new int[]{1, 1},   // stride of 1
                true,              // use padding
                ActivationFunctions.relu()
        ));
        
        // Max pooling layer
        network.addTensorLayer(new PoolingLayer(
                network.getOutputShape(),
                new int[]{2, 2},   // 2x2 pool
                new int[]{2, 2},   // stride of 2
                PoolingType.MAX
        ));
        
        // Second convolutional layer
        network.addTensorLayer(new ConvolutionalLayer(
                network.getOutputShape(),
                new int[]{3, 3},   // 3x3 kernel
                32,                // 32 output channels
                new int[]{1, 1},   // stride of 1
                true,              // use padding
                ActivationFunctions.relu()
        ));
        
        // Max pooling layer
        network.addTensorLayer(new PoolingLayer(
                network.getOutputShape(),
                new int[]{2, 2},   // 2x2 pool
                new int[]{2, 2},   // stride of 2
                PoolingType.MAX
        ));
        
        // Flatten layer
        network.addTensorLayer(new FlattenLayer(network.getOutputShape()));
        
        // Fully connected output layer
        int flattenSize = network.getOutputShape()[0];
        network.addTensorLayer(new FullyConnectedLayer(
                new int[]{flattenSize},
                numClasses,
                true,              // use softmax for multi-class classification
                ActivationFunctions.linear()
        ));
        
        return network;
    }
}