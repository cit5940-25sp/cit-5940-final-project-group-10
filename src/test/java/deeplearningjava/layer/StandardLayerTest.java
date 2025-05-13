package deeplearningjava.layer;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import deeplearningjava.api.Layer.LayerType;
import deeplearningjava.core.activation.ActivationFunction;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.core.Node;
import deeplearningjava.core.Edge;

import java.util.List;

/**
 * Tests for the StandardLayer class.
 */
public class StandardLayerTest {
    
    private StandardLayer standardLayer;
    private final int layerSize = 3;
    private ActivationFunction activation;
    
    @BeforeEach
    public void setUp() {
        activation = ActivationFunctions.relu();
        standardLayer = new StandardLayer(layerSize, activation);
    }
    
    @Test
    public void testConstructor() {
        // Verify the layer was created with correct size
        assertEquals(layerSize, standardLayer.getSize());
        
        // Verify layer type
        assertEquals(LayerType.HIDDEN, standardLayer.getType());
        assertTrue(standardLayer.isLayerType(LayerType.HIDDEN));
        
        // Verify activation function
        List<Node> nodes = standardLayer.getNodes();
        assertEquals(layerSize, nodes.size());
        
        for (Node node : nodes) {
            assertSame(activation, node.getActivationFunction());
        }
    }
    
    @Test
    public void testDifferentActivations() {
        // Test with different activation functions
        ActivationFunction[] activations = {
            ActivationFunctions.sigmoid(),
            ActivationFunctions.tanh(),
            ActivationFunctions.linear(),
            ActivationFunctions.leakyRelu(0.1)
        };
        
        for (ActivationFunction act : activations) {
            StandardLayer layer = new StandardLayer(layerSize, act);
            List<Node> nodes = layer.getNodes();
            
            for (Node node : nodes) {
                assertSame(act, node.getActivationFunction());
            }
        }
    }
    
    @Test
    public void testInvalidSize() {
        // Size must be positive
        assertThrows(IllegalArgumentException.class, () -> {
            new StandardLayer(0, activation);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new StandardLayer(-1, activation);
        });
    }
    
    @Test
    public void testNullActivationFunction() {
        // Activation function must not be null
        assertThrows(NullPointerException.class, () -> {
            new StandardLayer(layerSize, null);
        });
    }
    
    @Test
    public void testForwardWithDirectInputs() {
        // Test forward pass with direct inputs
        double[] inputs = {0.1, 0.2, 0.3};
        double[] outputs = standardLayer.forward(inputs);
        
        // For ReLU activation, output should equal input if input is positive
        assertArrayEquals(inputs, outputs);
        
        // Test that the internal node values were updated
        List<Node> nodes = standardLayer.getNodes();
        for (int i = 0; i < layerSize; i++) {
            assertEquals(inputs[i], nodes.get(i).getValue());
        }
    }
    
    @Test
    public void testForwardWithIncorrectInputSize() {
        // Test with incorrect input size
        double[] incorrectInputs = {0.1, 0.2}; // Size 2, should be 3
        
        assertThrows(IllegalArgumentException.class, () -> {
            standardLayer.forward(incorrectInputs);
        });
    }
    
    @Test
    public void testForwardWithCalculatedInputs() {
        // Create a network with input -> standard -> output
        InputLayer inputLayer = new InputLayer(2);
        StandardLayer hiddenLayer = new StandardLayer(3, ActivationFunctions.relu());
        outputLayer = new OutputLayer(1, ActivationFunctions.sigmoid(), false);
        
        // Connect layers
        inputLayer.connectTo(hiddenLayer);
        hiddenLayer.connectTo(outputLayer);
        
        // Initialize weights
        inputLayer.initializeWeights(hiddenLayer.getSize());
        hiddenLayer.initializeWeights(outputLayer.getSize());
        
        // Set custom weights for predictable results
        double[][] inputWeights = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};
        double[] hiddenBiases = {0.1, 0.2, 0.3};
        
        inputLayer.setWeights(inputWeights);
        hiddenLayer.setBiases(hiddenBiases);
        
        // Forward pass
        double[] inputs = {1.0, 2.0};
        inputLayer.forward(inputs);
        double[] hiddenOutputs = hiddenLayer.forward(null); // Calculate from incoming connections
        
        // Manually calculate expected outputs
        double[] expectedOutputs = new double[3];
        for (int i = 0; i < 3; i++) {
            double netInput = inputs[0] * inputWeights[0][i] + inputs[1] * inputWeights[1][i] + hiddenBiases[i];
            expectedOutputs[i] = Math.max(0, netInput); // ReLU activation
        }
        
        // Verify outputs
        assertArrayEquals(expectedOutputs, hiddenOutputs, 0.0001);
    }
    
    @Test
    public void testWeightsAndBiases() {
        // Connect to output layer
        OutputLayer outputLayer = new OutputLayer(2, ActivationFunctions.sigmoid(), false);
        standardLayer.connectTo(outputLayer);
        
        // Initialize weights
        standardLayer.initializeWeights(outputLayer.getSize());
        
        // Get initial weights and biases
        double[][] initialWeights = standardLayer.getWeights();
        double[] initialBiases = standardLayer.getBiases();
        
        // Verify dimensions
        assertEquals(layerSize, initialWeights.length);
        assertEquals(outputLayer.getSize(), initialWeights[0].length);
        assertEquals(layerSize, initialBiases.length);
        
        // Set custom weights and biases
        double[][] newWeights = {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}};
        double[] newBiases = {0.01, 0.02, 0.03};
        
        standardLayer.setWeights(newWeights);
        standardLayer.setBiases(newBiases);
        
        // Verify the weights and biases were set correctly
        assertArrayEquals(newWeights[0], standardLayer.getWeights()[0], 0.0001);
        assertArrayEquals(newWeights[1], standardLayer.getWeights()[1], 0.0001);
        assertArrayEquals(newWeights[2], standardLayer.getWeights()[2], 0.0001);
        assertArrayEquals(newBiases, standardLayer.getBiases(), 0.0001);
    }
    
    @ParameterizedTest
    @CsvSource({
        "sigmoid, 0.5, 0.25",  // sigmoid(0) = 0.5, derivative = 0.25
        "tanh, 0.0, 1.0",      // tanh(0) = 0, derivative = 1
        "relu, 0.0, 0.0",      // relu(0) = 0, derivative at 0 is typically 0
        "linear, 0.0, 1.0"     // linear(0) = 0, derivative = 1
    })
    public void testDifferentActivationFunctions(String activationName, double expectedOutput, double expectedDerivative) {
        // Get activation function by name
        ActivationFunction activation = ActivationFunctions.get(activationName);
        StandardLayer layer = new StandardLayer(1, activation);
        
        // Set input to 0
        layer.forward(new double[]{0.0});
        
        // Get output and verify it matches expected value for the activation function
        double output = layer.getNodes().get(0).getValue();
        assertEquals(expectedOutput, output, 0.0001);
        
        // Get derivative and verify
        double derivative = activation.derivative(0.0);
        assertEquals(expectedDerivative, derivative, 0.0001);
    }
    
    @Test
    public void testBackward() {
        // Create a small network to test backward pass
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
        
        // Set up gradients for backward pass
        double[] outputGradients = {0.1, -0.1};
        outputLayer.backward(new double[]{0.7, 0.3}); // Target values
        
        // Save original weights and biases
        double[][] originalWeights = hiddenLayer.getWeights();
        double[] originalBiases = hiddenLayer.getBiases();
        
        // Perform backward pass
        double[] inputGradients = hiddenLayer.backward(outputGradients);
        
        // Verify input gradients have correct dimension
        assertEquals(inputLayer.getSize(), inputGradients.length);
        
        // Verify weights and biases were updated
        double[][] updatedWeights = hiddenLayer.getWeights();
        double[] updatedBiases = hiddenLayer.getBiases();
        
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
    
    private OutputLayer outputLayer;
}