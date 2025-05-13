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
 * Tests for the InputLayer class.
 */
public class InputLayerTest {
    
    private InputLayer inputLayer;
    private final int layerSize = 4;
    
    @BeforeEach
    public void setUp() {
        inputLayer = new InputLayer(layerSize);
    }
    
    @Test
    public void testConstructor() {
        // Verify the layer was created with correct size
        assertEquals(layerSize, inputLayer.getSize());
        
        // Verify layer type
        assertEquals(LayerType.INPUT, inputLayer.getType());
        assertTrue(inputLayer.isLayerType(LayerType.INPUT));
        
        // Verify activation function is linear
        List<Node> nodes = inputLayer.getNodes();
        assertEquals(layerSize, nodes.size());
        
        // The activation function should be Linear for an input layer
        ActivationFunction activationFunction = nodes.get(0).getActivationFunction();
        assertEquals("Linear", activationFunction.getName());
    }
    
    @ParameterizedTest
    @ValueSource(ints = {1, 2, 10, 100})
    public void testLayerSizes(int size) {
        InputLayer layer = new InputLayer(size);
        assertEquals(size, layer.getSize());
        assertEquals(size, layer.getNodes().size());
    }
    
    @Test
    public void testInvalidSize() {
        // Size must be positive
        assertThrows(IllegalArgumentException.class, () -> {
            new InputLayer(0);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new InputLayer(-1);
        });
    }
    
    @Test
    public void testForward() {
        // Test forward pass with matching input size
        double[] inputs = {0.1, 0.2, 0.3, 0.4};
        double[] outputs = inputLayer.forward(inputs);
        
        // Input layer should pass through inputs unchanged
        assertArrayEquals(inputs, outputs);
        
        // Test that the internal node values were updated
        List<Node> nodes = inputLayer.getNodes();
        for (int i = 0; i < layerSize; i++) {
            assertEquals(inputs[i], nodes.get(i).getValue());
        }
    }
    
    @Test
    public void testForwardWithIncorrectInputSize() {
        // Test with incorrect input size
        double[] incorrectInputs = {0.1, 0.2, 0.3}; // Size 3, should be 4
        
        assertThrows(IllegalArgumentException.class, () -> {
            inputLayer.forward(incorrectInputs);
        });
    }
    
    @Test
    public void testBackward() {
        // Backward pass for input layer should return zeros
        double[] gradients = {0.1, 0.2, 0.3, 0.4};
        double[] backwardGradients = inputLayer.backward(gradients);
        
        // Should return array of zeros with size equal to layer size
        assertEquals(layerSize, backwardGradients.length);
        for (double grad : backwardGradients) {
            assertEquals(0.0, grad);
        }
    }
    
    @Test
    public void testConnectionToNextLayer() {
        // Create a standard layer to connect to
        StandardLayer nextLayer = new StandardLayer(2, ActivationFunctions.relu());
        
        // Connect the input layer to the standard layer
        inputLayer.connectTo(nextLayer);
        
        // Verify connections
        List<Node> inputNodes = inputLayer.getNodes();
        List<Node> nextNodes = nextLayer.getNodes();
        
        // Each input node should have connections to all nodes in the next layer
        for (Node inputNode : inputNodes) {
            assertEquals(nextNodes.size(), inputNode.getOutgoingConnections().size());
        }
        
        // Each node in the next layer should have connections from all input nodes
        for (Node nextNode : nextNodes) {
            assertEquals(inputNodes.size(), nextNode.getIncomingConnections().size());
        }
    }
    
    @Test
    public void testNoWeightsBeforeConnection() {
        // Input layer has no weights before connecting to next layer
        assertNull(inputLayer.getWeights());
    }
    
    @Test
    public void testWeightsAfterConnection() {
        // Create next layer
        StandardLayer nextLayer = new StandardLayer(2, ActivationFunctions.relu());
        
        // Connect and initialize weights
        inputLayer.connectTo(nextLayer);
        inputLayer.initializeWeights(nextLayer.getSize());
        
        // Get weights
        double[][] weights = inputLayer.getWeights();
        
        // Verify dimensions
        assertNotNull(weights);
        assertEquals(layerSize, weights.length);
        assertEquals(nextLayer.getSize(), weights[0].length);
    }
}