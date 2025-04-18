package deeplearningjava;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class NetworkTest {

    @Test
    public void testDefaultConstructor() {
        Network network = new Network();
        assertNotNull(network);
        assertTrue(network.getLayers().isEmpty());
    }

    @Test
    public void testParameterizedConstructor() {
        int[] layerSizes = {2, 3, 1};
        Network network = new Network(
            layerSizes,
            Node.SIGMOID,
            Node.SIGMOID_DERIVATIVE,
            Node.LINEAR,
            Node.LINEAR_DERIVATIVE,
            false
        );
        
        assertNotNull(network);
        assertEquals(layerSizes.length, network.getLayers().size());
        
        // Check layer sizes match what was specified
        List<Layer> layers = network.getLayers();
        for (int i = 0; i < layers.size(); i++) {
            assertEquals(layerSizes[i], layers.get(i).getSize());
        }
    }

    @Test
    public void testConstructorWithInvalidLayerSizesThrowsException() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            new Network(
                new int[] {0, 3, 1},
                Node.SIGMOID,
                Node.SIGMOID_DERIVATIVE,
                Node.LINEAR,
                Node.LINEAR_DERIVATIVE,
                false
            );
        });
        
        assertTrue(exception.getMessage().contains("Layer 1size must be positive"));
    }

    @Test
    public void testConstructorWithTooFewLayersThrowsException() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            new Network(
                new int[] {5},
                Node.SIGMOID,
                Node.SIGMOID_DERIVATIVE,
                Node.LINEAR,
                Node.LINEAR_DERIVATIVE,
                false
            );
        });
        
        assertTrue(exception.getMessage().contains("Network must have at least an input and output layer"));
    }

    @Test
    public void testSetLearningRate() {
        Network network = new Network();
        
        network.setLearningRate(0.05);
        
        // We can't check the learning rate directly as it's private, but we can test that
        // an invalid value is rejected
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            network.setLearningRate(0);
        });
        
        assertTrue(exception.getMessage().contains("Learning rate must be positive"));
    }

    @Test
    public void testAddLayer() {
        Network network = new Network();
        
        Layer inputLayer = new Layer(2, Node.LINEAR, Node.LINEAR_DERIVATIVE);
        Layer hiddenLayer = new Layer(3, Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        
        network.addLayer(inputLayer);
        assertEquals(1, network.getLayers().size());
        
        network.addLayer(hiddenLayer);
        assertEquals(2, network.getLayers().size());
        
        // Verify layers are connected
        assertEquals(hiddenLayer.getSize(), inputLayer.getNodes().get(0).getOutgoingConnections().size());
    }

    @Test
    public void testFeedForward() {
        Network network = new Network(
            new int[] {2, 3, 1},
            Node.SIGMOID,
            Node.SIGMOID_DERIVATIVE,
            Node.LINEAR,
            Node.LINEAR_DERIVATIVE,
            false
        );
        
        double[] inputs = {0.5, 0.7};
        double[] outputs = network.feedForward(inputs);
        
        // Check basic properties of output
        assertNotNull(outputs);
        assertEquals(1, outputs.length); // Should match size of output layer
    }

    @Test
    public void testFeedForwardWithWrongInputSizeThrowsException() {
        Network network = new Network(
            new int[] {2, 3, 1},
            Node.SIGMOID,
            Node.SIGMOID_DERIVATIVE,
            Node.LINEAR,
            Node.LINEAR_DERIVATIVE,
            false
        );
        
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            network.feedForward(new double[] {0.5, 0.7, 0.9});
        });
        
        assertTrue(exception.getMessage().contains("Input size"));
    }

    @Test
    public void testFeedForwardWithEmptyNetworkThrowsException() {
        Network network = new Network();
        
        Exception exception = assertThrows(IllegalStateException.class, () -> {
            network.feedForward(new double[] {0.5, 0.7});
        });
        
        assertTrue(exception.getMessage().contains("Network cannot be empty"));
    }

    @Test
    public void testBackPropagate() {
        Network network = new Network(
            new int[] {2, 3, 1},
            Node.SIGMOID,
            Node.SIGMOID_DERIVATIVE,
            Node.LINEAR,
            Node.LINEAR_DERIVATIVE,
            false
        );
        
        // First feed forward to set up the network state
        double[] inputs = {0.5, 0.7};
        network.feedForward(inputs);
        
        // Then backpropagate with target
        double[] targets = {1.0};
        network.backPropagate(targets);
        
        // We can't easily verify specific values, but we can ensure no exceptions were thrown
    }

    @Test
    public void testBackPropagateWithWrongTargetSizeThrowsException() {
        Network network = new Network(
            new int[] {2, 3, 1},
            Node.SIGMOID,
            Node.SIGMOID_DERIVATIVE,
            Node.LINEAR,
            Node.LINEAR_DERIVATIVE,
            false
        );
        
        double[] inputs = {0.5, 0.7};
        network.feedForward(inputs);
        
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            network.backPropagate(new double[] {1.0, 0.0});
        });
        
        assertTrue(exception.getMessage().contains("Target output size"));
    }

    @Test
    public void testTrainingIteration() {
        Network network = new Network(
            new int[] {2, 3, 1},
            Node.SIGMOID,
            Node.SIGMOID_DERIVATIVE,
            Node.LINEAR,
            Node.LINEAR_DERIVATIVE,
            false
        );
        
        double[] inputs = {0.5, 0.7};
        double[] targets = {1.0};
        
        double[] outputs = network.trainingIteration(inputs, targets);
        
        // Verify output properties
        assertNotNull(outputs);
        assertEquals(targets.length, outputs.length);
    }

    @Test
    public void testGetLayers() {
        Network network = new Network(
            new int[] {2, 3, 1},
            Node.SIGMOID,
            Node.SIGMOID_DERIVATIVE,
            Node.LINEAR,
            Node.LINEAR_DERIVATIVE,
            false
        );
        
        List<Layer> layers = network.getLayers();
        
        // Verify that getLayers returns an unmodifiable list
        Exception exception = assertThrows(UnsupportedOperationException.class, () -> {
            layers.add(new Layer(5, Node.SIGMOID, Node.SIGMOID_DERIVATIVE));
        });
    }
}