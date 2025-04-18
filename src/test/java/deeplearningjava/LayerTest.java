package deeplearningjava;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class LayerTest {

    @Test
    public void testLayerConstructor() {
        int size = 5;
        Layer layer = new Layer(size, Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        
        assertNotNull(layer);
        assertEquals(size, layer.getSize());
        assertEquals(size, layer.getNodes().size());
        assertFalse(layer.getNodes().contains(null));
    }

    @Test
    public void testConstructorWithInvalidSizeThrowsException() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            new Layer(0, Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        });
        
        assertTrue(exception.getMessage().contains("size must be positive"));
    }

    @Test
    public void testSoftmaxLayerConstructor() {
        int size = 3;
        Layer layer = new Layer(size, Node.SIGMOID, Node.SIGMOID_DERIVATIVE, true);
        
        assertNotNull(layer);
        assertEquals(size, layer.getSize());
        assertEquals(size, layer.getNodes().size());
        
        // Verify all nodes use LINEAR activation for softmax layer
        for (Node node : layer.getNodes()) {
            // We can't directly compare function references, but we can check behavior
            assertEquals(2.0, node.getActivationFunction().applyAsDouble(2.0), 1e-10);
            assertEquals(1.0, node.getActivationFunction().applyAsDouble(1.0), 1e-10);
        }
    }

    @Test
    public void testConnectToLayer() {
        Layer sourceLayer = new Layer(3, Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        Layer targetLayer = new Layer(2, Node.RELU, Node.RELU_DERIVATIVE);
        
        sourceLayer.connectTo(targetLayer);
        
        // Each source node should connect to each target node
        for (Node sourceNode : sourceLayer.getNodes()) {
            assertEquals(targetLayer.getSize(), sourceNode.getOutgoingConnections().size());
        }
        
        for (Node targetNode : targetLayer.getNodes()) {
            assertEquals(sourceLayer.getSize(), targetNode.getIncomingConnections().size());
        }
    }

    @Test
    public void testInitializeWeights() {
        Layer sourceLayer = new Layer(3, Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        Layer targetLayer = new Layer(2, Node.RELU, Node.RELU_DERIVATIVE);
        
        sourceLayer.connectTo(targetLayer);
        sourceLayer.initializeWeights(targetLayer.getSize());
        
        // Weights should be initialized, we can't check exact values due to randomness
        // but we can verify they're set to non-zero values
        for (Node sourceNode : sourceLayer.getNodes()) {
            sourceNode.getOutgoingConnections().forEach(edge -> 
                assertNotEquals(0.0, Math.abs(edge.getWeight()), 1e-10));
        }
    }

    @Test
    public void testSetOutputs() {
        int size = 3;
        Layer layer = new Layer(size, Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        
        double[] inputs = {0.1, 0.5, 0.9};
        layer.setOutputs(inputs);
        
        // Verify each node has the correct value
        for (int i = 0; i < size; i++) {
            assertEquals(inputs[i], layer.getNodes().get(i).getValue(), 1e-10);
        }
    }

    @Test
    public void testSetOutputsWithWrongSizeThrowsException() {
        Layer layer = new Layer(3, Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            layer.setOutputs(new double[] {0.1, 0.5});
        });
        
        assertTrue(exception.getMessage().contains("Number of inputs"));
    }

    @Test
    public void testCalculateOutputGradients() {
        Layer layer = new Layer(2, Node.LINEAR, Node.LINEAR_DERIVATIVE);
        
        // Set initial values
        layer.setOutputs(new double[] {0.7, 0.3});
        
        // Calculate gradients based on targets
        double[] targets = {1.0, 0.0};
        layer.calculateOutputGradients(targets);
        
        // For linear activation and MSE loss, gradient = (output - target)
        assertEquals(0.7 - 1.0, layer.getNodes().get(0).getGradient(), 1e-10);
        assertEquals(0.3 - 0.0, layer.getNodes().get(1).getGradient(), 1e-10);
    }

    @Test
    public void testCalculateHiddenGradients() {
        // Create a simple 2-layer network
        Layer hiddenLayer = new Layer(2, Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        Layer outputLayer = new Layer(1, Node.LINEAR, Node.LINEAR_DERIVATIVE);
        
        // Connect the layers
        hiddenLayer.connectTo(outputLayer);
        hiddenLayer.initializeWeights(outputLayer.getSize());
        
        // Set values and gradients
        hiddenLayer.setOutputs(new double[] {0.5, 0.5});
        outputLayer.getNodes().get(0).setGradient(1.0);
        
        // Calculate hidden gradients
        hiddenLayer.calculateHiddenGradients();
        
        // Each hidden node should have a non-zero gradient
        for (Node node : hiddenLayer.getNodes()) {
            assertNotEquals(0.0, node.getGradient());
        }
    }

    @Test
    public void testUpdateParameters() {
        // Create a simple 2-layer network
        Layer hiddenLayer = new Layer(2, Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        Layer outputLayer = new Layer(1, Node.LINEAR, Node.LINEAR_DERIVATIVE);
        
        // Connect the layers
        hiddenLayer.connectTo(outputLayer);
        
        // Store original bias values
        double[] originalBiases = outputLayer.getNodes().stream()
                                   .mapToDouble(Node::getBias)
                                   .toArray();
        
        // Set gradients to trigger updates
        for (Node node : outputLayer.getNodes()) {
            node.setGradient(0.1);
        }
        
        // Update parameters
        double learningRate = 0.5;
        outputLayer.updateParameters(learningRate);
        
        // Check bias updates
        for (int i = 0; i < outputLayer.getNodes().size(); i++) {
            double expectedBias = originalBiases[i] - learningRate * 0.1;
            assertEquals(expectedBias, outputLayer.getNodes().get(i).getBias(), 1e-10);
        }
    }
}