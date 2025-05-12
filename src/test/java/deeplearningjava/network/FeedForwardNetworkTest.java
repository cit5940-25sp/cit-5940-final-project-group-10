package deeplearningjava.network;

import org.junit.jupiter.api.Test;

import java.util.List;

import deeplearningjava.api.Layer;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.layer.InputLayer;
import deeplearningjava.layer.OutputLayer;
import deeplearningjava.layer.StandardLayer;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the FeedForwardNetwork implementation.
 */
public class FeedForwardNetworkTest {

    @Test
    public void testDefaultConstructor() {
        FeedForwardNetwork network = new FeedForwardNetwork();
        assertNotNull(network);
        assertTrue(network.getLayers().isEmpty());
    }

    @Test
    public void testParameterizedConstructor() {
        int[] layerSizes = {2, 3, 1};
        FeedForwardNetwork network = new FeedForwardNetwork(
            layerSizes,
            ActivationFunctions.sigmoid(),
            ActivationFunctions.linear(),
            false
        );
        
        assertNotNull(network);
        assertEquals(layerSizes.length, network.getLayers().size());
        
        // Check layer sizes match what was specified
        List<Layer> layers = network.getLayers();
        for (int i = 0; i < layers.size(); i++) {
            assertEquals(layerSizes[i], layers.get(i).getSize());
        }
        
        // Check layer types
        assertEquals(Layer.LayerType.INPUT, layers.get(0).getType());
        assertEquals(Layer.LayerType.HIDDEN, layers.get(1).getType());
        assertEquals(Layer.LayerType.OUTPUT, layers.get(2).getType());
    }
    
    @Test
    public void testCreateDefault() {
        int[] layerSizes = {2, 3, 1};
        FeedForwardNetwork network = FeedForwardNetwork.createDefault(layerSizes);
        
        assertNotNull(network);
        assertEquals(layerSizes.length, network.getLayers().size());
        
        // Check output layer configuration for regression (single output)
        Layer outputLayer = network.getLayers().get(network.getLayers().size() - 1);
        assertEquals(Layer.LayerType.OUTPUT, outputLayer.getType());
        assertEquals(1, outputLayer.getSize());
        
        // Check multi-class configuration
        int[] multiClassSizes = {2, 3, 4};
        FeedForwardNetwork multiClassNetwork = FeedForwardNetwork.createDefault(multiClassSizes);
        
        Layer multiClassOutputLayer = multiClassNetwork.getLayers().get(multiClassNetwork.getLayers().size() - 1);
        assertEquals(Layer.LayerType.OUTPUT, multiClassOutputLayer.getType());
        assertEquals(4, multiClassOutputLayer.getSize());
        
        // Verify actual output class
        assertTrue(multiClassOutputLayer instanceof OutputLayer);
    }

    @Test
    public void testConstructorWithInvalidLayerSizesThrowsException() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            new FeedForwardNetwork(
                new int[] {0, 3, 1},
                ActivationFunctions.sigmoid(),
                ActivationFunctions.linear(),
                false
            );
        });
        
        assertTrue(exception.getMessage().contains("must be positive"));
    }

    @Test
    public void testConstructorWithTooFewLayersThrowsException() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            new FeedForwardNetwork(
                new int[] {5},
                ActivationFunctions.sigmoid(),
                ActivationFunctions.linear(),
                false
            );
        });
        
        assertTrue(exception.getMessage().contains("at least input and output layers"));
    }

    @Test
    public void testAddLayer() {
        FeedForwardNetwork network = new FeedForwardNetwork();
        
        Layer inputLayer = new InputLayer(2);
        Layer hiddenLayer = new StandardLayer(3, ActivationFunctions.sigmoid());
        Layer outputLayer = new OutputLayer(1, ActivationFunctions.linear(), false);
        
        network.addLayer(inputLayer);
        assertEquals(1, network.getLayers().size());
        
        network.addLayer(hiddenLayer);
        assertEquals(2, network.getLayers().size());
        
        network.addLayer(outputLayer);
        assertEquals(3, network.getLayers().size());
        
        // Verify layer structure validation
        assertEquals(Layer.LayerType.INPUT, network.getLayers().get(0).getType());
        assertEquals(Layer.LayerType.HIDDEN, network.getLayers().get(1).getType());
        assertEquals(Layer.LayerType.OUTPUT, network.getLayers().get(2).getType());
    }

    @Test
    public void testForward() {
        FeedForwardNetwork network = new FeedForwardNetwork(
            new int[] {2, 3, 1},
            ActivationFunctions.sigmoid(),
            ActivationFunctions.linear(),
            false
        );
        
        double[] inputs = {0.5, 0.7};
        double[] outputs = network.forward(inputs);
        
        // Check basic properties of output
        assertNotNull(outputs);
        assertEquals(1, outputs.length); // Should match size of output layer
    }

    @Test
    public void testForwardWithWrongInputSizeThrowsException() {
        FeedForwardNetwork network = new FeedForwardNetwork(
            new int[] {2, 3, 1},
            ActivationFunctions.sigmoid(),
            ActivationFunctions.linear(),
            false
        );
        
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            network.forward(new double[] {0.5, 0.7, 0.9});
        });
        
        assertTrue(exception.getMessage().contains("Input size"));
    }

    @Test
    public void testForwardWithEmptyNetworkThrowsException() {
        FeedForwardNetwork network = new FeedForwardNetwork();
        
        Exception exception = assertThrows(IllegalStateException.class, () -> {
            network.forward(new double[] {0.5, 0.7});
        });
        
        assertTrue(exception.getMessage().contains("has no layers"));
    }

    @Test
    public void testTrain() {
        FeedForwardNetwork network = new FeedForwardNetwork(
            new int[] {2, 3, 1},
            ActivationFunctions.sigmoid(),
            ActivationFunctions.linear(),
            false
        );
        
        double[] inputs = {0.5, 0.7};
        double[] targets = {1.0};
        
        double[] outputs = network.train(inputs, targets);
        
        // Verify output properties
        assertNotNull(outputs);
        assertEquals(targets.length, outputs.length);
    }
    
    @Test
    public void testForwardPropagation() {
        // Create a simple network with fixed weights for predictable outputs
        FeedForwardNetwork network = new FeedForwardNetwork();
        
        network.addLayer(new InputLayer(2));
        network.addLayer(new StandardLayer(2, ActivationFunctions.linear()));
        network.addLayer(new OutputLayer(1, ActivationFunctions.linear(), false));
        
        // Use the network to process inputs
        double[] inputs = {1.0, 2.0};
        double[] outputs = network.forward(inputs);
        
        // Since output depends on random weights, we just check the output dimensions
        assertNotNull(outputs);
        assertEquals(1, outputs.length);
    }
}