package deeplearningjava.network;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

import deeplearningjava.api.TensorLayer;
import deeplearningjava.api.BaseNetwork.NetworkType;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.core.tensor.Tensor;
import deeplearningjava.layer.tensor.FullyConnectedLayer;
import deeplearningjava.layer.tensor.FlattenLayer;

/**
 * Tests for the DenseNetwork implementation when used in tensor mode.
 */
public class DenseNetworkTensorTest {

    @Test
    public void testDefaultConstructorTensorMode() {
        DenseNetwork network = new DenseNetwork();
        // Set the network to tensor mode by adding a tensor layer
        FlattenLayer flattenLayer = new FlattenLayer(new int[]{1, 8, 8});
        network.addTensorLayer(flattenLayer);
        
        assertNotNull(network);
        assertEquals(1, network.getTensorLayers().size());
        assertEquals(NetworkType.TENSOR, network.getType());
        assertTrue(network.isInTensorMode());
    }

    @Test
    public void testCreateForBoardGame() {
        int[] inputShape = {1, 8, 8}; // 1 channel, 8x8 board
        int[] hiddenLayerSizes = {64, 32};
        int outputSize = 1;
        
        DenseNetwork network = DenseNetwork.createForBoardGame(
            inputShape,
            hiddenLayerSizes,
            outputSize,
            ActivationFunctions.relu(),
            ActivationFunctions.tanh(),
            false
        );
        
        assertNotNull(network);
        assertEquals(4, network.getTensorLayers().size()); // flatten + 2 hidden + output
        
        // Check layer types
        assertTrue(network.getTensorLayers().get(0) instanceof FlattenLayer);
        assertTrue(network.getTensorLayers().get(1) instanceof FullyConnectedLayer);
        assertTrue(network.getTensorLayers().get(2) instanceof FullyConnectedLayer);
        assertTrue(network.getTensorLayers().get(3) instanceof FullyConnectedLayer);
        
        // Check input and output shapes
        assertArrayEquals(inputShape, network.getInputShape());
        assertArrayEquals(new int[]{outputSize}, network.getOutputShape());
    }
    
    @Test
    public void testCreateForOthello() {
        DenseNetwork network = DenseNetwork.createForOthello(8, 1);
        
        assertNotNull(network);
        assertTrue(network.getTensorLayers().size() > 0);
        assertArrayEquals(new int[]{1, 8, 8}, network.getInputShape());
        assertArrayEquals(new int[]{1}, network.getOutputShape());
    }
    
    @Test
    public void testForward() {
        // Create a simple network for an 8x8 board
        DenseNetwork network = DenseNetwork.createForOthello(8, 1);
        
        // Create a test input tensor (1 channel, 8x8 board)
        double[] boardData = new double[64]; // 8x8 = 64 elements
        for (int i = 0; i < boardData.length; i++) {
            boardData[i] = (i % 3 == 0) ? 1.0 : (i % 3 == 1) ? -1.0 : 0.0; // Some arbitrary pattern
        }
        
        Tensor input = new Tensor(boardData, 1, 8, 8);
        
        // Forward pass
        Tensor output = network.forward(input);
        
        // Check output dimensions
        assertNotNull(output);
        assertArrayEquals(new int[]{1}, output.getShape());
    }
    
    @Test
    public void testTrain() {
        // Create a simple network for an 8x8 board
        DenseNetwork network = DenseNetwork.createForOthello(8, 1);
        
        // Create a test input tensor (1 channel, 8x8 board)
        double[] boardData = new double[64]; // 8x8 = 64 elements
        for (int i = 0; i < boardData.length; i++) {
            boardData[i] = (i % 3 == 0) ? 1.0 : (i % 3 == 1) ? -1.0 : 0.0; // Some arbitrary pattern
        }
        
        Tensor input = new Tensor(boardData, 1, 8, 8);
        Tensor target = new Tensor(new double[]{0.5}, 1); // Target evaluation
        
        // Training
        Tensor output = network.train(input, target);
        
        // Check output dimensions
        assertNotNull(output);
        assertArrayEquals(new int[]{1}, output.getShape());
    }
    
    @Test
    public void testTrainBatch() {
        // Create a simple network for an 8x8 board
        DenseNetwork network = DenseNetwork.createForOthello(8, 1);
        
        // Create a batch of input tensors
        Tensor[] inputs = new Tensor[2];
        Tensor[] targets = new Tensor[2];
        
        // First board
        double[] boardData1 = new double[64];
        for (int i = 0; i < boardData1.length; i++) {
            boardData1[i] = (i % 3 == 0) ? 1.0 : (i % 3 == 1) ? -1.0 : 0.0;
        }
        inputs[0] = new Tensor(boardData1, 1, 8, 8);
        targets[0] = new Tensor(new double[]{0.7}, 1);
        
        // Second board
        double[] boardData2 = new double[64];
        for (int i = 0; i < boardData2.length; i++) {
            boardData2[i] = (i % 3 == 1) ? 1.0 : (i % 3 == 2) ? -1.0 : 0.0;
        }
        inputs[1] = new Tensor(boardData2, 1, 8, 8);
        targets[1] = new Tensor(new double[]{-0.3}, 1);
        
        // Train for a single epoch (for test speed)
        double finalLoss = network.trainBatch(inputs, targets, 1);
        
        // Just verify training completes without error
        assertTrue(finalLoss >= 0.0); // Loss should be non-negative
    }
    
    @Test
    public void testForwardWithWrongInputShapeThrowsException() {
        DenseNetwork network = DenseNetwork.createForOthello(8, 1);
        
        // Create a tensor with wrong shape (2 channels instead of 1)
        Tensor input = new Tensor(new double[128], 2, 8, 8);
        
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            network.forward(input);
        });
        
        assertTrue(exception.getMessage().contains("Input shape"));
    }
    
    @Test
    public void testAddLayer() {
        DenseNetwork network = new DenseNetwork();
        
        // Add a flatten layer
        FlattenLayer flattenLayer = new FlattenLayer(new int[]{1, 8, 8});
        network.addTensorLayer(flattenLayer);
        assertEquals(1, network.getTensorLayers().size());
        
        // Add a fully connected layer
        FullyConnectedLayer fcLayer = new FullyConnectedLayer(
                flattenLayer.getOutputShape(),
                32,
                false,
                ActivationFunctions.relu()
        );
        network.addTensorLayer(fcLayer);
        assertEquals(2, network.getTensorLayers().size());
    }
    
    @Test
    public void testAddIncompatibleLayerThrowsException() {
        DenseNetwork network = new DenseNetwork();
        
        // Add a flatten layer with input shape [1, 8, 8]
        FlattenLayer flattenLayer = new FlattenLayer(new int[]{1, 8, 8});
        network.addTensorLayer(flattenLayer);
        
        // Try to add a layer with incompatible input shape
        FullyConnectedLayer fcLayer = new FullyConnectedLayer(
                new int[]{32}, // Wrong input shape, doesn't match output of flatten layer
                16,
                false,
                ActivationFunctions.relu()
        );
        
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            network.addTensorLayer(fcLayer);
        });
        
        assertTrue(exception.getMessage().contains("not compatible"));
    }
    
    @Test
    public void testGetSummary() {
        DenseNetwork network = DenseNetwork.createForOthello(8, 1);
        
        String summary = network.getSummary();
        
        // Verify summary information
        assertTrue(summary.contains("TENSOR Network Summary"));
        assertTrue(summary.contains("Input Shape: [1, 8, 8]"));
        assertTrue(summary.contains("Output Shape: [1]"));
        assertTrue(summary.contains("Layer Structure:"));
    }
}