package deeplearningjava.network;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import deeplearningjava.api.Layer;
import deeplearningjava.api.Network;
import deeplearningjava.api.TensorLayer;
import deeplearningjava.api.TensorNetwork;
import deeplearningjava.core.activation.ActivationFunction;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.core.tensor.Tensor;
import deeplearningjava.layer.InputLayer;
import deeplearningjava.layer.OutputLayer;
import deeplearningjava.layer.StandardLayer;
import deeplearningjava.layer.tensor.FlattenLayer;
import deeplearningjava.layer.tensor.FullyConnectedLayer;
import deeplearningjava.network.optimizer.Optimizer;
import deeplearningjava.network.optimizer.SGD;

/**
 * Implementation of a versatile neural network that handles both standard vector inputs
 * and tensor (multidimensional) inputs.
 * 
 * This network can be used as either:
 * 1. A traditional dense network with vector inputs
 * 2. A tensor-based network with multidimensional inputs like game boards
 */
public class DenseNetwork extends AbstractNetwork implements Network, TensorNetwork {
    
    // For traditional vector-based network
    protected final List<Layer> layers;
    
    // For tensor-based network
    protected final List<TensorLayer> tensorLayers;
    
    protected Optimizer optimizer;
    protected boolean useTensorMode;
    
    /**
     * Creates an empty network in standard vector mode.
     */
    public DenseNetwork() {
        super(NetworkType.DENSE);
        this.layers = new ArrayList<>();
        this.tensorLayers = new ArrayList<>();
        this.optimizer = new SGD(learningRate);
        this.useTensorMode = false;
    }
    
    /**
     * Creates a standard vector-based dense network with the specified layers.
     * 
     * @param layers The layers for this network
     */
    public DenseNetwork(List<Layer> layers) {
        super(NetworkType.DENSE);
        this.layers = new ArrayList<>(Objects.requireNonNull(layers, "layers must not be null"));
        this.tensorLayers = new ArrayList<>();
        this.optimizer = new SGD(learningRate);
        this.useTensorMode = false;
        validateLayerStructure();
    }
    
    /**
     * Creates a tensor-based network with the specified layers.
     * 
     * @param tensorLayers The tensor layers for this network
     */
    public DenseNetwork(List<TensorLayer> tensorLayers, boolean useTensor) {
        super(useTensor ? NetworkType.TENSOR : NetworkType.DENSE);
        this.layers = new ArrayList<>();
        this.tensorLayers = new ArrayList<>(Objects.requireNonNull(tensorLayers, "tensorLayers must not be null"));
        this.optimizer = new SGD(learningRate);
        this.useTensorMode = true;
        validateTensorLayerStructure();
    }
    
    /**
     * Creates a standard vector-based dense network with the specified architecture.
     * 
     * @param layerSizes Array of layer sizes (including input and output)
     * @param hiddenActivation Activation function for hidden layers
     * @param outputActivation Activation function for the output layer
     * @param useSoftmax Whether to use softmax in the output layer
     */
    public DenseNetwork(int[] layerSizes, 
                      ActivationFunction hiddenActivation,
                      ActivationFunction outputActivation,
                      boolean useSoftmax) {
        super(NetworkType.DENSE);
        this.layers = new ArrayList<>();
        this.tensorLayers = new ArrayList<>();
        this.optimizer = new SGD(learningRate);
        this.useTensorMode = false;
        
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("Network must have at least input and output layers");
        }
        
        Objects.requireNonNull(hiddenActivation, "hiddenActivation must not be null");
        Objects.requireNonNull(outputActivation, "outputActivation must not be null");
        
        // Create input layer
        addLayer(new InputLayer(layerSizes[0]));
        
        // Create hidden layers
        for (int i = 1; i < layerSizes.length - 1; i++) {
            addLayer(new StandardLayer(layerSizes[i], hiddenActivation));
        }
        
        // Create output layer
        addLayer(new OutputLayer(layerSizes[layerSizes.length - 1], 
                               outputActivation, useSoftmax));
    }
    
    /**
     * Creates a standard dense network with ReLU activation for hidden layers
     * and appropriate output activation based on the output size.
     * 
     * @param layerSizes Array of layer sizes (including input and output)
     * @return A configured dense network
     */
    public static DenseNetwork createDefault(int[] layerSizes) {
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("Network must have at least input and output layers");
        }
        
        // Use appropriate output activation based on output size
        ActivationFunction outputActivation;
        boolean useSoftmax;
        
        if (layerSizes[layerSizes.length - 1] == 1) {
            // For regression or binary classification without softmax
            outputActivation = ActivationFunctions.tanh();
            useSoftmax = false;
        } else {
            // For multi-class classification
            outputActivation = ActivationFunctions.linear();
            useSoftmax = true;
        }
        
        return new DenseNetwork(layerSizes, 
                              ActivationFunctions.relu(),
                              outputActivation, 
                              useSoftmax);
    }
    
    /**
     * Creates a tensor-based network for processing 2D board states with one or more channels.
     * 
     * @param inputShape The input shape [channels, height, width]
     * @param hiddenLayerSizes Sizes of hidden fully connected layers
     * @param outputSize Size of output layer
     * @param hiddenActivation Activation function for hidden layers
     * @param outputActivation Activation function for output layer
     * @param useSoftmax Whether to use softmax in the output layer
     * @return A configured tensor network
     */
    public static DenseNetwork createForBoardGame(
            int[] inputShape, 
            int[] hiddenLayerSizes,
            int outputSize,
            ActivationFunction hiddenActivation,
            ActivationFunction outputActivation,
            boolean useSoftmax) {
        
        Objects.requireNonNull(inputShape, "inputShape must not be null");
        Objects.requireNonNull(hiddenLayerSizes, "hiddenLayerSizes must not be null");
        Objects.requireNonNull(hiddenActivation, "hiddenActivation must not be null");
        Objects.requireNonNull(outputActivation, "outputActivation must not be null");
        
        if (inputShape.length != 3) {
            throw new IllegalArgumentException("Input shape must be [channels, height, width]");
        }
        
        DenseNetwork network = new DenseNetwork();
        network.useTensorMode = true;
        
        // Add flatten layer to convert 3D input to 1D
        FlattenLayer flattenLayer = new FlattenLayer(inputShape);
        network.addTensorLayer(flattenLayer);
        
        int[] flattenedShape = flattenLayer.getOutputShape();
        
        // Add hidden fully connected layers
        int[] currentInputShape = flattenedShape;
        for (int size : hiddenLayerSizes) {
            FullyConnectedLayer hiddenLayer = new FullyConnectedLayer(
                    currentInputShape,
                    size,
                    false,  // No softmax for hidden layers
                    hiddenActivation
            );
            network.addTensorLayer(hiddenLayer);
            currentInputShape = hiddenLayer.getOutputShape();
        }
        
        // Add output layer
        FullyConnectedLayer outputLayer = new FullyConnectedLayer(
                currentInputShape,
                outputSize,
                useSoftmax,
                outputActivation
        );
        network.addTensorLayer(outputLayer);
        
        return network;
    }
    
    /**
     * Creates a network for Othello with appropriate defaults.
     * 
     * @param boardSize The size of the Othello board (typically 8)
     * @param channels Number of input channels (typically 1 or more for different piece types)
     * @return A configured network for Othello
     */
    public static DenseNetwork createForOthello(int boardSize, int channels) {
        int[] inputShape = {channels, boardSize, boardSize};
        int[] hiddenLayerSizes = {128, 64, 32}; // Example architecture
        
        return createForBoardGame(
                inputShape,
                hiddenLayerSizes,
                1,  // Single output for board evaluation
                ActivationFunctions.relu(),
                ActivationFunctions.tanh(),  // Tanh for [-1, 1] range evaluation
                false  // No softmax for regression
        );
    }
    
    // Standard vector-based layer methods
    
    @Override
    public void addLayer(Layer layer) {
        if (useTensorMode) {
            throw new IllegalStateException("Cannot add standard layers in tensor mode. Use addTensorLayer instead.");
        }
        
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
        if (useTensorMode) {
            throw new IllegalStateException("Network is in tensor mode. Use getTensorLayers instead.");
        }
        return Collections.unmodifiableList(layers);
    }
    
    // Tensor-based layer methods
    
    @Override
    public void addTensorLayer(TensorLayer layer) {
        if (!useTensorMode) {
            // If this is the first tensor layer, switch to tensor mode
            if (tensorLayers.isEmpty() && layers.isEmpty()) {
                useTensorMode = true;
            } else {
                throw new IllegalStateException("Cannot add tensor layers in standard mode. Use addLayer instead.");
            }
        }
        
        Objects.requireNonNull(layer, "layer must not be null");
        
        if (!tensorLayers.isEmpty()) {
            TensorLayer previousLayer = tensorLayers.get(tensorLayers.size() - 1);
            
            // Validate that output shape of previous layer matches input shape of new layer
            if (!isShapeCompatible(previousLayer.getOutputShape(), layer.getInputShape())) {
                throw new IllegalArgumentException(
                        "Output shape " + formatShape(previousLayer.getOutputShape()) + 
                        " of previous layer is not compatible with input shape " + 
                        formatShape(layer.getInputShape()) + " of new layer");
            }
            
            previousLayer.connectTo(layer);
        }
        
        tensorLayers.add(layer);
    }
    
    @Override
    public List<TensorLayer> getTensorLayers() {
        if (!useTensorMode) {
            throw new IllegalStateException("Network is in standard mode. Use getLayers instead.");
        }
        return Collections.unmodifiableList(tensorLayers);
    }
    
    /**
     * Checks if two shapes are compatible for connection.
     * 
     * @param outputShape The output shape of a layer
     * @param inputShape The input shape of the next layer
     * @return true if shapes are compatible, false otherwise
     */
    private boolean isShapeCompatible(int[] outputShape, int[] inputShape) {
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
    
    // Standard vector-based forward and training methods
    
    @Override
    public double[] forward(double[] inputs) {
        if (useTensorMode) {
            // Convert to tensor and use tensor forward method
            Tensor inputTensor = new Tensor(inputs, inputs.length);
            Tensor outputTensor = forward(inputTensor);
            return outputTensor.getData();
        }
        
        validateNetwork();
        
        if (inputs.length != getInputSize()) {
            throw new IllegalArgumentException(
                String.format("Input size (%d) must match input layer size (%d)",
                             inputs.length, getInputSize()));
        }
        
        // Forward through input layer
        double[] currentOutput = layers.get(0).forward(inputs);
        
        // Forward through remaining layers
        for (int i = 1; i < layers.size(); i++) {
            currentOutput = layers.get(i).forward(null); // Pass null to use connections
        }
        
        return currentOutput;
    }
    
    @Override
    public double[] train(double[] inputs, double[] targets) {
        if (useTensorMode) {
            // Convert to tensors and use tensor train method
            Tensor inputTensor = new Tensor(inputs, inputs.length);
            Tensor targetTensor = new Tensor(targets, targets.length);
            Tensor outputTensor = train(inputTensor, targetTensor);
            return outputTensor.getData();
        }
        
        // Standard vector-based forward pass
        double[] outputs = forward(inputs);
        
        // Backward pass (starting from output layer)
        double[] gradients = targets;
        for (int i = layers.size() - 1; i >= 0; i--) {
            gradients = layers.get(i).backward(gradients);
        }
        
        return outputs;
    }
    
    @Override
    public double trainBatch(double[][] inputs, double[][] targets, int epochs) {
        if (useTensorMode) {
            // Convert to tensors and use tensor trainBatch method
            Tensor[] inputTensors = new Tensor[inputs.length];
            Tensor[] targetTensors = new Tensor[targets.length];
            
            for (int i = 0; i < inputs.length; i++) {
                inputTensors[i] = new Tensor(inputs[i], inputs[i].length);
                targetTensors[i] = new Tensor(targets[i], targets[i].length);
            }
            
            return trainBatch(inputTensors, targetTensors, epochs);
        }
        
        validateBatchParameters(inputs, targets, epochs);
        
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
    
    // Tensor-based forward and training methods
    
    @Override
    public Tensor forward(Tensor input) {
        if (!useTensorMode) {
            throw new IllegalStateException("Network is in standard mode, not tensor mode");
        }
        
        validateTensorNetwork();
        
        Objects.requireNonNull(input, "input must not be null");
        int[] inputShape = input.getShape();
        int[] expectedShape = getInputShape();
        
        if (!Arrays.equals(inputShape, expectedShape)) {
            throw new IllegalArgumentException(
                    "Input shape " + formatShape(inputShape) + 
                    " doesn't match expected shape " + formatShape(expectedShape));
        }
        
        Tensor output = input;
        for (TensorLayer layer : tensorLayers) {
            output = layer.forward(output);
        }
        
        return output;
    }
    
    // Both Network and TensorNetwork interfaces use train methods
    public Tensor train(Tensor input, Tensor target) {
        if (!useTensorMode) {
            throw new IllegalStateException("Network is in standard mode, not tensor mode");
        }
        
        // Forward pass
        Tensor output = forward(input);
        
        // Backward pass (compute gradients)
        Tensor gradients = calculateLossGradients(output, target);
        
        // Backpropagate through layers (in reverse order)
        for (int i = tensorLayers.size() - 1; i >= 0; i--) {
            gradients = tensorLayers.get(i).backward(gradients);
        }
        
        // Update parameters
        for (TensorLayer layer : tensorLayers) {
            layer.updateParameters(learningRate);
        }
        
        return output;
    }
    
    @Override
    public double trainBatch(Tensor[] inputs, Tensor[] targets, int epochs) {
        if (!useTensorMode) {
            throw new IllegalStateException("Network is in standard mode, not tensor mode");
        }
        
        if (inputs == null || targets == null) {
            throw new IllegalArgumentException("Inputs and targets must not be null");
        }
        
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException(
                    "Number of inputs (" + inputs.length + 
                    ") doesn't match number of targets (" + targets.length + ")");
        }
        
        if (epochs <= 0) {
            throw new IllegalArgumentException("Epochs must be positive");
        }
        
        double finalLoss = 0;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochLoss = 0;
            
            for (int i = 0; i < inputs.length; i++) {
                Tensor output = train(inputs[i], targets[i]);
                epochLoss += calculateLoss(output, targets[i]);
            }
            
            finalLoss = epochLoss / inputs.length;
            System.out.printf("Epoch %d/%d - Loss: %.6f%n", epoch + 1, epochs, finalLoss);
        }
        
        return finalLoss;
    }
    
    /**
     * Calculates the loss between predicted and target tensors.
     * Uses Mean Squared Error (MSE).
     * 
     * @param predicted Predicted tensor
     * @param target Target tensor
     * @return The loss value
     */
    protected double calculateLoss(Tensor predicted, Tensor target) {
        int[] predictedShape = predicted.getShape();
        int[] targetShape = target.getShape();
        
        if (!Arrays.equals(predictedShape, targetShape)) {
            throw new IllegalArgumentException(
                    "Predicted shape " + formatShape(predictedShape) + 
                    " doesn't match target shape " + formatShape(targetShape));
        }
        
        double sum = 0;
        double[] predictedData = predicted.getData();
        double[] targetData = target.getData();
        
        for (int i = 0; i < predictedData.length; i++) {
            double diff = predictedData[i] - targetData[i];
            sum += diff * diff;
        }
        
        return sum / predictedData.length; // Mean Squared Error
    }
    
    /**
     * Calculates the gradients of the loss with respect to the outputs.
     * Uses Mean Squared Error (MSE) loss.
     * 
     * @param predicted Predicted tensor
     * @param target Target tensor
     * @return Gradients of the loss
     */
    protected Tensor calculateLossGradients(Tensor predicted, Tensor target) {
        int[] predictedShape = predicted.getShape();
        int[] targetShape = target.getShape();
        
        if (!Arrays.equals(predictedShape, targetShape)) {
            throw new IllegalArgumentException(
                    "Predicted shape " + formatShape(predictedShape) + 
                    " doesn't match target shape " + formatShape(targetShape));
        }
        
        // For MSE, gradient is 2 * (predicted - target) / n
        int size = predicted.getSize();
        double[] gradientData = new double[size];
        double[] predictedData = predicted.getData();
        double[] targetData = target.getData();
        
        for (int i = 0; i < size; i++) {
            gradientData[i] = 2.0 * (predictedData[i] - targetData[i]) / size;
        }
        
        return new Tensor(gradientData, targetShape);
    }
    
    /**
     * Calculates the loss between predicted and target outputs.
     * Uses Mean Squared Error (MSE).
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
    
    /**
     * Sets the optimizer for this network.
     * 
     * @param optimizer The optimizer to use
     */
    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer must not be null");
        this.optimizer.setLearningRate(this.learningRate);
    }
    
    /**
     * Gets the current optimizer.
     * 
     * @return The optimizer
     */
    public Optimizer getOptimizer() {
        return optimizer;
    }
    
    @Override
    public void setLearningRate(double learningRate) {
        super.setLearningRate(learningRate);
        optimizer.setLearningRate(learningRate);
    }
    
    @Override
    public int getInputSize() {
        if (useTensorMode) {
            throw new IllegalStateException("Network is in tensor mode. Use getInputShape instead.");
        }
        validateNetwork();
        return layers.get(0).getSize();
    }
    
    @Override
    public int getOutputSize() {
        if (useTensorMode) {
            throw new IllegalStateException("Network is in tensor mode. Use getOutputShape instead.");
        }
        validateNetwork();
        return layers.get(layers.size() - 1).getSize();
    }
    
    @Override
    public int[] getInputShape() {
        if (!useTensorMode) {
            int size = getInputSize();
            return new int[] { size };
        }
        validateTensorNetwork();
        return tensorLayers.get(0).getInputShape();
    }
    
    @Override
    public int[] getOutputShape() {
        if (!useTensorMode) {
            int size = getOutputSize();
            return new int[] { size };
        }
        validateTensorNetwork();
        return tensorLayers.get(tensorLayers.size() - 1).getOutputShape();
    }
    
    @Override
    public int getLayerCount() {
        return useTensorMode ? tensorLayers.size() : layers.size();
    }
    
    /**
     * Validates that the layer structure is appropriate for a standard dense network.
     * 
     * @throws IllegalArgumentException if the layer structure is invalid
     */
    private void validateLayerStructure() {
        if (layers.isEmpty()) {
            return; // Empty network is valid (for now)
        }
        
        // First layer should be an input layer
        if (layers.get(0).getType() != Layer.LayerType.INPUT) {
            throw new IllegalArgumentException("First layer must be an input layer");
        }
        
        // Last layer should be an output layer
        if (layers.get(layers.size() - 1).getType() != Layer.LayerType.OUTPUT) {
            throw new IllegalArgumentException("Last layer must be an output layer");
        }
    }
    
    /**
     * Validates that the tensor layer structure is appropriate.
     * 
     * @throws IllegalArgumentException if the layer structure is invalid
     */
    private void validateTensorLayerStructure() {
        if (tensorLayers.isEmpty()) {
            return; // Empty network is valid (for now)
        }
        
        // Verify layer connections
        for (int i = 1; i < tensorLayers.size(); i++) {
            TensorLayer prevLayer = tensorLayers.get(i - 1);
            TensorLayer currentLayer = tensorLayers.get(i);
            
            if (!isShapeCompatible(prevLayer.getOutputShape(), currentLayer.getInputShape())) {
                throw new IllegalArgumentException(
                        "Output shape " + formatShape(prevLayer.getOutputShape()) + 
                        " of layer " + (i - 1) + " is not compatible with input shape " + 
                        formatShape(currentLayer.getInputShape()) + " of layer " + i);
            }
        }
    }
    
    /**
     * Validates that the network has been initialized in standard mode.
     * 
     * @throws IllegalStateException if the network is empty
     */
    protected void validateNetwork() {
        if (!isInitialized() || layers.isEmpty()) {
            throw new IllegalStateException("Network has not been initialized");
        }
    }
    
    /**
     * Validates that the network has been initialized in tensor mode.
     * 
     * @throws IllegalStateException if the network is empty
     */
    protected void validateTensorNetwork() {
        if (!isInitialized() || tensorLayers.isEmpty()) {
            throw new IllegalStateException("Tensor network has not been initialized");
        }
    }
    
    @Override
    public boolean isInitialized() {
        return useTensorMode ? !tensorLayers.isEmpty() : !layers.isEmpty();
    }
    
    @Override
    public NetworkType getType() {
        return useTensorMode ? NetworkType.TENSOR : NetworkType.DENSE;
    }
    
    /**
     * Checks if this network is in tensor mode.
     * 
     * @return true if in tensor mode, false if in standard vector mode
     */
    public boolean isInTensorMode() {
        return useTensorMode;
    }
    
    @Override
    protected void appendDetails(StringBuilder summary) {
        if (!isInitialized()) {
            summary.append("Network is empty");
            return;
        }
        
        if (useTensorMode) {
            appendTensorDetails(summary);
        } else {
            appendVectorDetails(summary);
        }
        
        summary.append("\nOptimizer: ").append(optimizer.getClass().getSimpleName());
    }
    
    private void appendVectorDetails(StringBuilder summary) {
        summary.append("Input Size: ").append(getInputSize()).append("\n");
        summary.append("Output Size: ").append(getOutputSize()).append("\n");
        summary.append("Layer Sizes: ");
        
        for (int i = 0; i < layers.size(); i++) {
            summary.append(layers.get(i).getSize());
            if (i < layers.size() - 1) {
                summary.append(" -> ");
            }
        }
    }
    
    private void appendTensorDetails(StringBuilder summary) {
        summary.append("Input Shape: ").append(formatShape(getInputShape())).append("\n");
        summary.append("Output Shape: ").append(formatShape(getOutputShape())).append("\n");
        
        summary.append("Layer Structure:\n");
        for (int i = 0; i < tensorLayers.size(); i++) {
            TensorLayer layer = tensorLayers.get(i);
            summary.append("  ").append(i).append(": ")
                   .append(layer.getClass().getSimpleName())
                   .append(" - Input: ").append(formatShape(layer.getInputShape()))
                   .append(", Output: ").append(formatShape(layer.getOutputShape()));
            
            if (i < tensorLayers.size() - 1) {
                summary.append("\n");
            }
        }
    }
}