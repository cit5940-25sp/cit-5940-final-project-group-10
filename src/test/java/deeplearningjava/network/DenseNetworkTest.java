package deeplearningjava.network;

import org.junit.jupiter.api.Test;

import java.util.List;

import deeplearningjava.api.Layer;
import deeplearningjava.api.BaseNetwork.NetworkType;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.layer.InputLayer;
import deeplearningjava.layer.OutputLayer;
import deeplearningjava.layer.StandardLayer;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the DenseNetwork implementation.
 */
public class DenseNetworkTest {

    @Test
    public void testDefaultConstructor() {
        DenseNetwork network = new DenseNetwork();
        assertNotNull(network);
        assertTrue(network.getLayers().isEmpty());
        assertEquals(NetworkType.DENSE, network.getType());
    }

    @Test
    public void testParameterizedConstructor() {
        int[] layerSizes = {2, 3, 1};
        DenseNetwork network = new DenseNetwork(
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
        DenseNetwork network = DenseNetwork.createDefault(layerSizes);
        
        assertNotNull(network);
        assertEquals(layerSizes.length, network.getLayers().size());
        
        // Check output layer configuration for regression (single output)
        Layer outputLayer = network.getLayers().get(network.getLayers().size() - 1);
        assertEquals(Layer.LayerType.OUTPUT, outputLayer.getType());
        assertEquals(1, outputLayer.getSize());
        
        // Check multi-class configuration
        int[] multiClassSizes = {2, 3, 4};
        DenseNetwork multiClassNetwork = DenseNetwork.createDefault(multiClassSizes);
        
        Layer multiClassOutputLayer = multiClassNetwork.getLayers().get(multiClassNetwork.getLayers().size() - 1);
        assertEquals(Layer.LayerType.OUTPUT, multiClassOutputLayer.getType());
        assertEquals(4, multiClassOutputLayer.getSize());
        
        // Verify actual output class
        assertTrue(multiClassOutputLayer instanceof OutputLayer);
        OutputLayer typedOutputLayer = (OutputLayer) multiClassOutputLayer;
        assertTrue(typedOutputLayer.usesSoftmax());
    }

    @Test
    public void testConstructorWithInvalidLayerSizesThrowsException() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            new DenseNetwork(
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
            new DenseNetwork(
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
        DenseNetwork network = new DenseNetwork();
        
        Layer inputLayer = new InputLayer(2);
        Layer hiddenLayer = new StandardLayer(3, ActivationFunctions.sigmoid());
        Layer outputLayer = new OutputLayer(1, ActivationFunctions.linear(), false);
        
        network.addLayer(inputLayer);
        assertEquals(1, network.getLayers().size());
        
        network.addLayer(hiddenLayer);
        assertEquals(2, network.getLayers().size());
        
        network.addLayer(outputLayer);
        assertEquals(3, network.getLayers().size());
        
        // Verify layer structure
        assertEquals(Layer.LayerType.INPUT, network.getLayers().get(0).getType());
        assertEquals(Layer.LayerType.HIDDEN, network.getLayers().get(1).getType());
        assertEquals(Layer.LayerType.OUTPUT, network.getLayers().get(2).getType());
    }

    @Test
    public void testForward() {
        DenseNetwork network = new DenseNetwork(
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
        DenseNetwork network = new DenseNetwork(
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
        DenseNetwork network = new DenseNetwork();
        
        Exception exception = assertThrows(IllegalStateException.class, () -> {
            network.forward(new double[] {0.5, 0.7});
        });
        
        assertTrue(exception.getMessage().contains("has not been initialized"));
    }

    @Test
    public void testTrain() {
        DenseNetwork network = new DenseNetwork(
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
    public void testGetSummary() {
        DenseNetwork network = new DenseNetwork(
            new int[] {2, 3, 1},
            ActivationFunctions.sigmoid(),
            ActivationFunctions.linear(),
            false
        );
        
        String summary = network.getSummary();
        
        // Verify summary information
        assertTrue(summary.contains("DENSE Network Summary"));
        assertTrue(summary.contains("Layer Count: 3"));
        assertTrue(summary.contains("Input Size: 2"));
        assertTrue(summary.contains("Output Size: 1"));
        assertTrue(summary.contains("Layer Sizes: 2 -> 3 -> 1"));
    }
    
    @Test
    public void testGetInputSize() {
        DenseNetwork network = new DenseNetwork(
            new int[] {5, 4, 3},
            ActivationFunctions.sigmoid(),
            ActivationFunctions.linear(),
            false
        );
        
        assertEquals(5, network.getInputSize());
    }
    
    @Test
    public void testGetOutputSize() {
        DenseNetwork network = new DenseNetwork(
            new int[] {5, 4, 3},
            ActivationFunctions.sigmoid(),
            ActivationFunctions.linear(),
            false
        );
        
        assertEquals(3, network.getOutputSize());
    }
    
    @Test
    public void testIsInitialized() {
        DenseNetwork emptyNetwork = new DenseNetwork();
        assertFalse(emptyNetwork.isInitialized());
        
        DenseNetwork network = new DenseNetwork(
            new int[] {2, 3, 1},
            ActivationFunctions.sigmoid(),
            ActivationFunctions.linear(),
            false
        );
        
        assertTrue(network.isInitialized());
    }
}