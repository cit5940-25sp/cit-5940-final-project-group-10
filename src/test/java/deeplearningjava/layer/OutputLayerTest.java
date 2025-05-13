package deeplearningjava.layer;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import deeplearningjava.api.Layer.LayerType;
import deeplearningjava.core.activation.ActivationFunction;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.core.Node;

import java.util.List;

/**
 * Tests for the OutputLayer class.
 */
public class OutputLayerTest {
    
    private OutputLayer withoutSoftmax;
    private OutputLayer withSoftmax;
    private final int layerSize = 3;
    private ActivationFunction activation;
    
    @BeforeEach
    public void setUp() {
        activation = ActivationFunctions.sigmoid();
        withoutSoftmax = new OutputLayer(layerSize, activation, false);
        withSoftmax = new OutputLayer(layerSize, activation, true);
    }
    
    @Test
    public void testConstructor() {
        // Verify the layers were created with correct size
        assertEquals(layerSize, withoutSoftmax.getSize());
        assertEquals(layerSize, withSoftmax.getSize());
        
        // Verify layer types
        assertEquals(LayerType.OUTPUT, withoutSoftmax.getType());
        assertEquals(LayerType.OUTPUT, withSoftmax.getType());
        
        assertTrue(withoutSoftmax.isLayerType(LayerType.OUTPUT));
        assertTrue(withSoftmax.isLayerType(LayerType.OUTPUT));
        
        // Verify activation function
        List<Node> nodes = withoutSoftmax.getNodes();
        assertEquals(layerSize, nodes.size());
        
        for (Node node : nodes) {
            assertSame(activation, node.getActivationFunction());
        }
        
        // Verify softmax flag
        assertFalse(withoutSoftmax.usesSoftmax());
        assertTrue(withSoftmax.usesSoftmax());
    }
    
    @Test
    public void testInvalidSize() {
        // Size must be positive
        assertThrows(IllegalArgumentException.class, () -> {
            new OutputLayer(0, activation, false);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new OutputLayer(-1, activation, true);
        });
    }
    
    @Test
    public void testNullActivationFunction() {
        // Activation function must not be null
        assertThrows(NullPointerException.class, () -> {
            new OutputLayer(layerSize, null, false);
        });
    }
    
    @Test
    public void testForwardWithDirectInputs() {
        // Test layer without softmax
        double[] inputs = {0.1, 0.2, 0.3};
        double[] outputs = withoutSoftmax.forward(inputs);
        
        // Without softmax, outputs should just pass through the activation function
        assertEquals(inputs.length, outputs.length);
        
        // Test that the internal node values were updated
        List<Node> nodes = withoutSoftmax.getNodes();
        for (int i = 0; i < layerSize; i++) {
            assertEquals(inputs[i], nodes.get(i).getValue());
        }
    }
    
    @Test
    public void testForwardWithSoftmax() {
        // Test layer with softmax
        double[] inputs = {1.0, 2.0, 3.0};
        double[] outputs = withSoftmax.forward(inputs);
        
        // With softmax, outputs should sum to 1
        double sum = 0;
        for (double output : outputs) {
            sum += output;
        }
        assertEquals(1.0, sum, 0.0001);
        
        // The largest input should have the largest output
        double maxOutputValue = outputs[0];
        int maxOutputIndex = 0;
        for (int i = 1; i < outputs.length; i++) {
            if (outputs[i] > maxOutputValue) {
                maxOutputValue = outputs[i];
                maxOutputIndex = i;
            }
        }
        assertEquals(2, maxOutputIndex, "Largest input should give largest softmax output");
        
        // The smallest input should have the smallest output
        double minOutputValue = outputs[0];
        int minOutputIndex = 0;
        for (int i = 1; i < outputs.length; i++) {
            if (outputs[i] < minOutputValue) {
                minOutputValue = outputs[i];
                minOutputIndex = i;
            }
        }
        assertEquals(0, minOutputIndex, "Smallest input should give smallest softmax output");
    }
    
    @Test
    public void testSoftmaxNumericalStability() {
        // Test softmax with large values that might cause numerical overflow
        double[] largeInputs = {1000.0, 1000.1, 1000.2};
        
        // This should not throw exceptions or produce NaN/infinity
        double[] outputs = withSoftmax.forward(largeInputs);
        
        // Check outputs are valid probabilities
        for (double output : outputs) {
            assertTrue(output >= 0 && output <= 1, "Output should be a valid probability");
            assertFalse(Double.isNaN(output), "Output should not be NaN");
            assertFalse(Double.isInfinite(output), "Output should not be infinite");
        }
        
        // Sum should still be 1
        double sum = 0;
        for (double output : outputs) {
            sum += output;
        }
        assertEquals(1.0, sum, 0.0001);
        
        // The largest input should still have the largest output
        double maxOutputValue = outputs[0];
        int maxOutputIndex = 0;
        for (int i = 1; i < outputs.length; i++) {
            if (outputs[i] > maxOutputValue) {
                maxOutputValue = outputs[i];
                maxOutputIndex = i;
            }
        }
        assertEquals(2, maxOutputIndex, "Largest input should give largest softmax output");
    }
    
    @Test
    public void testBackwardWithoutSoftmax() {
        // Create a network to test backward pass
        InputLayer inputLayer = new InputLayer(2);
        StandardLayer hiddenLayer = new StandardLayer(3, ActivationFunctions.relu());
        OutputLayer outputLayer = new OutputLayer(2, ActivationFunctions.sigmoid(), false);
        
        // Connect layers
        inputLayer.connectTo(hiddenLayer);
        hiddenLayer.connectTo(outputLayer);
        
        // Initialize weights
        inputLayer.initializeWeights(hiddenLayer.getSize());
        hiddenLayer.initializeWeights(outputLayer.getSize());
        
        // Forward pass
        double[] inputs = {1.0, 2.0};
        inputLayer.forward(inputs);
        hiddenLayer.forward(null);
        outputLayer.forward(null);
        
        // Save original weights and biases
        double[][] originalWeights = outputLayer.getWeights();
        double[] originalBiases = outputLayer.getBiases();
        
        // Perform backward pass with target values
        double[] targets = {0.0, 1.0};
        double[] hiddenGradients = outputLayer.backward(targets);
        
        // Verify gradients have correct dimension
        assertEquals(hiddenLayer.getSize(), hiddenGradients.length);
        
        // Verify weights and biases were updated
        double[][] updatedWeights = outputLayer.getWeights();
        double[] updatedBiases = outputLayer.getBiases();
        
        // Weights and biases should have changed
        boolean weightsChanged = false;
        for (int i = 0; i < originalWeights.length; i++) {
            for (int j = 0; j < originalWeights[i].length; j++) {
                if (Math.abs(originalWeights[i][j] - updatedWeights[i][j]) > 0.0001) {
                    weightsChanged = true;
                    break;
                }
            }
        }
        assertTrue(weightsChanged, "Weights should have been updated");
        
        boolean biasesChanged = false;
        for (int i = 0; i < originalBiases.length; i++) {
            if (Math.abs(originalBiases[i] - updatedBiases[i]) > 0.0001) {
                biasesChanged = true;
                break;
            }
        }
        assertTrue(biasesChanged, "Biases should have been updated");
    }
    
    @Test
    public void testBackwardWithSoftmax() {
        // Create a network with softmax output
        InputLayer inputLayer = new InputLayer(2);
        StandardLayer hiddenLayer = new StandardLayer(3, ActivationFunctions.relu());
        OutputLayer outputLayer = new OutputLayer(2, ActivationFunctions.sigmoid(), true);
        
        // Connect layers
        inputLayer.connectTo(hiddenLayer);
        hiddenLayer.connectTo(outputLayer);
        
        // Initialize weights
        inputLayer.initializeWeights(hiddenLayer.getSize());
        hiddenLayer.initializeWeights(outputLayer.getSize());
        
        // Forward pass
        double[] inputs = {1.0, 2.0};
        inputLayer.forward(inputs);
        hiddenLayer.forward(null);
        outputLayer.forward(null);
        
        // Save original weights and biases
        double[][] originalWeights = outputLayer.getWeights();
        double[] originalBiases = outputLayer.getBiases();
        
        // Perform backward pass with target values (one-hot encoding for class 1)
        double[] targets = {0.0, 1.0};
        double[] hiddenGradients = outputLayer.backward(targets);
        
        // Verify gradients have correct dimension
        assertEquals(hiddenLayer.getSize(), hiddenGradients.length);
        
        // Verify weights and biases were updated
        double[][] updatedWeights = outputLayer.getWeights();
        double[] updatedBiases = outputLayer.getBiases();
        
        // Weights and biases should have changed
        boolean weightsChanged = false;
        for (int i = 0; i < originalWeights.length; i++) {
            for (int j = 0; j < originalWeights[i].length; j++) {
                if (Math.abs(originalWeights[i][j] - updatedWeights[i][j]) > 0.0001) {
                    weightsChanged = true;
                    break;
                }
            }
        }
        assertTrue(weightsChanged, "Weights should have been updated");
        
        boolean biasesChanged = false;
        for (int i = 0; i < originalBiases.length; i++) {
            if (Math.abs(originalBiases[i] - updatedBiases[i]) > 0.0001) {
                biasesChanged = true;
                break;
            }
        }
        assertTrue(biasesChanged, "Biases should have been updated");
    }
    
    @ParameterizedTest
    @ValueSource(ints = {1, 2, 5, 10})
    public void testDifferentLayerSizes(int size) {
        OutputLayer layer = new OutputLayer(size, activation, true);
        assertEquals(size, layer.getSize());
        assertEquals(size, layer.getNodes().size());
        
        // Test forward pass with appropriate input size
        double[] inputs = new double[size];
        for (int i = 0; i < size; i++) {
            inputs[i] = i + 1.0; // 1.0, 2.0, 3.0, ...
        }
        
        double[] outputs = layer.forward(inputs);
        assertEquals(size, outputs.length);
        
        // Softmax outputs should sum to 1
        double sum = 0;
        for (double output : outputs) {
            sum += output;
        }
        assertEquals(1.0, sum, 0.0001);
    }
}