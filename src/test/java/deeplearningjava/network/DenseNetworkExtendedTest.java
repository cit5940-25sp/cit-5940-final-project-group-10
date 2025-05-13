package deeplearningjava.network;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import deeplearningjava.api.Layer;
import deeplearningjava.api.TensorLayer;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.core.tensor.Tensor;
import deeplearningjava.layer.InputLayer;
import deeplearningjava.layer.OutputLayer;
import deeplearningjava.layer.StandardLayer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Extended tests for the DenseNetwork class.
 */
public class DenseNetworkExtendedTest {
    
    @Test
    public void testDefaultConstructor() {
        DenseNetwork network = new DenseNetwork();
        assertNotNull(network);
        assertTrue(network.getLayers().isEmpty());
        assertEquals(AbstractNetwork.NetworkType.DENSE, network.getType());
        assertFalse(network.isInTensorMode());
    }
    
    @Test
    public void testLayerListConstructor() {
        List<Layer> layers = new ArrayList<>();
        layers.add(new InputLayer(2));
        layers.add(new StandardLayer(3, ActivationFunctions.relu()));
        layers.add(new OutputLayer(1, ActivationFunctions.tanh(), false));
        
        DenseNetwork network = new DenseNetwork(layers);
        
        assertNotNull(network);
        assertEquals(layers.size(), network.getLayers().size());
        assertEquals(AbstractNetwork.NetworkType.DENSE, network.getType());
        assertFalse(network.isInTensorMode());
    }
    
    @Test
    public void testCreateDefault() {
        // Test regression model (single output)
        int[] layerSizes = {5, 4, 3, 1};
        DenseNetwork network = DenseNetwork.createDefault(layerSizes);
        
        assertNotNull(network);
        assertEquals(layerSizes.length, network.getLayers().size());
        assertEquals(layerSizes[0], network.getInputSize());
        assertEquals(layerSizes[layerSizes.length - 1], network.getOutputSize());
        
        // Test multi-class model (multiple outputs)
        int[] multiClassSizes = {5, 4, 3, 10};
        DenseNetwork multiClassNetwork = DenseNetwork.createDefault(multiClassSizes);
        
        assertEquals(multiClassSizes.length, multiClassNetwork.getLayers().size());
        assertEquals(multiClassSizes[multiClassSizes.length - 1], multiClassNetwork.getOutputSize());
        
        // Verify output layer is indeed the last layer
        Layer outputLayer = multiClassNetwork.getLayers().get(multiClassNetwork.getLayers().size() - 1);
        assertEquals(Layer.LayerType.OUTPUT, outputLayer.getType());
    }
    
    @Test
    public void testCreateForBoardGame() {
        int[] inputShape = {3, 8, 8}; // 3 channels, 8x8 board
        int[] hiddenLayerSizes = {128, 64};
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
        assertTrue(network.isInTensorMode());
        assertEquals(AbstractNetwork.NetworkType.TENSOR, network.getType());
        assertEquals(hiddenLayerSizes.length + 2, network.getLayerCount()); // +2 for flatten and output
        
        // Verify input and output shapes
        assertArrayEquals(inputShape, network.getInputShape());
        assertArrayEquals(new int[]{outputSize}, network.getOutputShape());
    }
    
    @Test
    public void testCreateForOthello() {
        int boardSize = 8;
        int channels = 3;
        
        DenseNetwork network = DenseNetwork.createForOthello(boardSize, channels);
        
        assertNotNull(network);
        assertTrue(network.isInTensorMode());
        
        // Verify input shape
        int[] expectedInputShape = {channels, boardSize, boardSize};
        assertArrayEquals(expectedInputShape, network.getInputShape());
        
        // Verify output shape (should be a single value for board evaluation)
        assertArrayEquals(new int[]{1}, network.getOutputShape());
    }
    
    @Test
    public void testStandardForward() {
        // Create a standard network
        int[] layerSizes = {2, 3, 1};
        DenseNetwork network = new DenseNetwork(
                layerSizes,
                ActivationFunctions.relu(),
                ActivationFunctions.tanh(),
                false
        );
        
        // Test forward pass
        double[] input = {0.5, -0.5};
        double[] output = network.forward(input);
        
        assertNotNull(output);
        assertEquals(layerSizes[layerSizes.length - 1], output.length);
    }
    
    @Test
    public void testTensorForward() {
        // Create a tensor network for board game
        int[] inputShape = {1, 4, 4}; // 1 channel, 4x4 board
        int[] hiddenLayerSizes = {16, 8};
        int outputSize = 1;
        
        DenseNetwork network = DenseNetwork.createForBoardGame(
                inputShape, 
                hiddenLayerSizes,
                outputSize,
                ActivationFunctions.relu(),
                ActivationFunctions.tanh(),
                false
        );
        
        // Create a test input tensor
        double[] data = new double[inputShape[0] * inputShape[1] * inputShape[2]];
        for (int i = 0; i < data.length; i++) {
            data[i] = (i % 2 == 0) ? 1.0 : -1.0; // Alternating values
        }
        Tensor input = new Tensor(data, inputShape);
        
        // Test forward pass
        Tensor output = network.forward(input);
        
        assertNotNull(output);
        assertArrayEquals(new int[]{outputSize}, output.getShape());
    }
    
    @Test
    public void testStandardTrain() {
        // Create a standard network
        int[] layerSizes = {2, 3, 1};
        DenseNetwork network = new DenseNetwork(
                layerSizes,
                ActivationFunctions.relu(),
                ActivationFunctions.tanh(),
                false
        );
        
        // Test training
        double[] input = {0.5, -0.5};
        double[] target = {1.0};
        double[] output = network.train(input, target);
        
        assertNotNull(output);
        assertEquals(target.length, output.length);
        
        // Test that the network is learning
        double initialLoss = network.calculateLoss(output, target);
        
        // Train for several iterations
        for (int i = 0; i < 10; i++) {
            output = network.train(input, target);
        }
        
        double finalLoss = network.calculateLoss(output, target);
        
        // Loss should decrease with training
        assertTrue(finalLoss <= initialLoss, 
                "Loss should decrease or stay the same during training");
    }
    
    @Test
    public void testTensorTrain() {
        // Create a tensor network
        int[] inputShape = {1, 3, 3}; // 1 channel, 3x3 board
        int[] hiddenLayerSizes = {9, 3};
        int outputSize = 1;
        
        DenseNetwork network = DenseNetwork.createForBoardGame(
                inputShape, 
                hiddenLayerSizes,
                outputSize,
                ActivationFunctions.relu(),
                ActivationFunctions.tanh(),
                false
        );
        
        // Create input and target tensors
        double[] inputData = new double[inputShape[0] * inputShape[1] * inputShape[2]];
        Arrays.fill(inputData, 0.5); // Uniform values
        Tensor input = new Tensor(inputData, inputShape);
        
        double[] targetData = new double[outputSize];
        Arrays.fill(targetData, 1.0); // Target is 1.0
        Tensor target = new Tensor(targetData, new int[]{outputSize});
        
        // Test training
        Tensor output = network.train(input, target);
        
        assertNotNull(output);
        assertArrayEquals(new int[]{outputSize}, output.getShape());
        
        // Test that the network is learning
        double initialLoss = network.calculateLoss(output, target);
        
        // Train for several iterations
        for (int i = 0; i < 10; i++) {
            output = network.train(input, target);
        }
        
        double finalLoss = network.calculateLoss(output, target);
        
        // Loss should decrease with training
        assertTrue(finalLoss <= initialLoss, 
                "Loss should decrease or stay the same during training");
    }
    
    @Test
    public void testLayerCount() {
        // Standard network
        int[] layerSizes = {2, 3, 4, 1};
        DenseNetwork standardNetwork = new DenseNetwork(
                layerSizes,
                ActivationFunctions.relu(),
                ActivationFunctions.tanh(),
                false
        );
        
        assertEquals(layerSizes.length, standardNetwork.getLayerCount());
        
        // Tensor network
        int[] inputShape = {1, 8, 8};
        int[] hiddenLayerSizes = {64, 32};
        int outputSize = 1;
        
        DenseNetwork tensorNetwork = DenseNetwork.createForBoardGame(
                inputShape, 
                hiddenLayerSizes,
                outputSize,
                ActivationFunctions.relu(),
                ActivationFunctions.tanh(),
                false
        );
        
        // Should include flatten layer, hidden layers, and output layer
        assertEquals(hiddenLayerSizes.length + 2, tensorNetwork.getLayerCount());
    }
    
    @Test
    public void testGetSummary() {
        // Standard network
        int[] layerSizes = {2, 3, 1};
        DenseNetwork standardNetwork = new DenseNetwork(
                layerSizes,
                ActivationFunctions.relu(),
                ActivationFunctions.tanh(),
                false
        );
        
        String standardSummary = standardNetwork.getSummary();
        
        assertTrue(standardSummary.contains("DENSE Network Summary"));
        assertTrue(standardSummary.contains("Layer Count: " + layerSizes.length));
        assertTrue(standardSummary.contains("Input Size: " + layerSizes[0]));
        assertTrue(standardSummary.contains("Output Size: " + layerSizes[layerSizes.length - 1]));
        
        // Tensor network
        int[] inputShape = {1, 4, 4};
        int[] hiddenLayerSizes = {16, 8};
        int outputSize = 1;
        
        DenseNetwork tensorNetwork = DenseNetwork.createForBoardGame(
                inputShape, 
                hiddenLayerSizes,
                outputSize,
                ActivationFunctions.relu(),
                ActivationFunctions.tanh(),
                false
        );
        
        String tensorSummary = tensorNetwork.getSummary();
        
        assertTrue(tensorSummary.contains("TENSOR Network Summary"));
        assertTrue(tensorSummary.contains("Layer Count: " + (hiddenLayerSizes.length + 2)));
        assertTrue(tensorSummary.contains("Input Shape: [1, 4, 4]"));
        assertTrue(tensorSummary.contains("Output Shape: [1]"));
    }
    
    @Test
    public void testSetLearningRate() {
        DenseNetwork network = new DenseNetwork();
        
        // Default learning rate
        double defaultRate = network.getLearningRate();
        assertEquals(0.01, defaultRate, 0.0001);
        
        // Set new learning rate
        double newRate = 0.05;
        network.setLearningRate(newRate);
        assertEquals(newRate, network.getLearningRate(), 0.0001);
        
        // Verify optimizer also has the new learning rate
        assertEquals(newRate, network.getOptimizer().getLearningRate(), 0.0001);
    }
    
    @Test
    public void testInputOutputSizeAndShape() {
        // Standard network
        int[] layerSizes = {2, 3, 4};
        DenseNetwork standardNetwork = new DenseNetwork(
                layerSizes,
                ActivationFunctions.relu(),
                ActivationFunctions.tanh(),
                false
        );
        
        assertEquals(layerSizes[0], standardNetwork.getInputSize());
        assertEquals(layerSizes[layerSizes.length - 1], standardNetwork.getOutputSize());
        
        assertArrayEquals(new int[]{layerSizes[0]}, standardNetwork.getInputShape());
        assertArrayEquals(new int[]{layerSizes[layerSizes.length - 1]}, standardNetwork.getOutputShape());
        
        // Tensor network
        int[] inputShape = {2, 6, 6};
        int[] hiddenLayerSizes = {36, 10};
        int outputSize = 1;
        
        DenseNetwork tensorNetwork = DenseNetwork.createForBoardGame(
                inputShape, 
                hiddenLayerSizes,
                outputSize,
                ActivationFunctions.relu(),
                ActivationFunctions.tanh(),
                false
        );
        
        assertArrayEquals(inputShape, tensorNetwork.getInputShape());
        assertArrayEquals(new int[]{outputSize}, tensorNetwork.getOutputShape());
        
        // Getting input/output size on tensor network should throw exception
        assertThrows(IllegalStateException.class, () -> {
            tensorNetwork.getInputSize();
        });
        
        assertThrows(IllegalStateException.class, () -> {
            tensorNetwork.getOutputSize();
        });
    }
    
    @Test
    public void testModeValidation() {
        // Create a standard network
        DenseNetwork standardNetwork = new DenseNetwork();
        // Add at least one layer to initialize the network
        standardNetwork.addLayer(new InputLayer(2));
        
        // Try to get tensor layers on standard network
        assertThrows(IllegalStateException.class, () -> {
            standardNetwork.getTensorLayers();
        });
        
        // Try to add tensor layer to standard network
        assertThrows(IllegalStateException.class, () -> {
            standardNetwork.addTensorLayer(null);
        });
        
        // Create a tensor network
        int[] inputShape = {1, 4, 4};
        int[] hiddenLayerSizes = {16};
        DenseNetwork tensorNetwork = DenseNetwork.createForBoardGame(
                inputShape, 
                hiddenLayerSizes,
                1,
                ActivationFunctions.relu(),
                ActivationFunctions.tanh(),
                false
        );
        
        // Try to get standard layers on tensor network
        assertThrows(IllegalStateException.class, () -> {
            tensorNetwork.getLayers();
        });
        
        // Try to add standard layer to tensor network
        assertThrows(IllegalStateException.class, () -> {
            tensorNetwork.addLayer(new InputLayer(2));
        });
    }
}