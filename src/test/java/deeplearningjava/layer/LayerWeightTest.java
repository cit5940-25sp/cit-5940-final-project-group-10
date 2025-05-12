package deeplearningjava.layer;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.core.tensor.Tensor;
import deeplearningjava.layer.tensor.FullyConnectedLayer;
import deeplearningjava.layer.tensor.ConvolutionalLayer;

public class LayerWeightTest {

    @Test
    public void testStandardLayerWeightsAndBiases() {
        // Create a standard layer
        StandardLayer layer = new StandardLayer(3, ActivationFunctions.relu());
        
        // Connect to another layer to create weights
        OutputLayer nextLayer = new OutputLayer(2, ActivationFunctions.tanh(), false);
        layer.connectTo(nextLayer);
        layer.initializeWeights(nextLayer.getSize());
        
        // Get the initial weights
        double[][] initialWeights = layer.getWeights();
        double[] initialBiases = layer.getBiases();
        
        // Verify dimensions
        assertEquals(3, initialWeights.length);
        assertEquals(2, initialWeights[0].length);
        assertEquals(3, initialBiases.length);
        
        // Create new weights and biases
        double[][] newWeights = {
            {0.1, 0.2},
            {0.3, 0.4},
            {0.5, 0.6}
        };
        
        double[] newBiases = {0.01, 0.02, 0.03};
        
        // Set new weights and biases
        layer.setWeights(newWeights);
        layer.setBiases(newBiases);
        
        // Get the updated weights and biases
        double[][] updatedWeights = layer.getWeights();
        double[] updatedBiases = layer.getBiases();
        
        // Verify that the weights and biases were updated
        assertArrayEquals(newWeights[0], updatedWeights[0], 0.0001);
        assertArrayEquals(newWeights[1], updatedWeights[1], 0.0001);
        assertArrayEquals(newWeights[2], updatedWeights[2], 0.0001);
        assertArrayEquals(newBiases, updatedBiases, 0.0001);
    }
    
    @Test
    public void testFullyConnectedLayerWeightsAndBiases() {
        // Create a fully connected layer
        int inputSize = 4;
        int outputSize = 3;
        int[] inputShape = new int[]{inputSize};
        
        FullyConnectedLayer layer = new FullyConnectedLayer(
                inputShape, outputSize, 
                false, ActivationFunctions.relu());
        
        // Get the initial weights and bias
        Tensor initialWeights = layer.getWeights();
        Tensor initialBias = layer.getBias();
        
        // Verify dimensions
        assertArrayEquals(new int[]{outputSize, inputSize}, initialWeights.getShape());
        assertArrayEquals(new int[]{outputSize}, initialBias.getShape());
        
        // Create new weights and bias tensors
        double[] weightData = new double[outputSize * inputSize];
        for (int i = 0; i < weightData.length; i++) {
            weightData[i] = 0.1 * (i + 1);
        }
        Tensor newWeights = new Tensor(weightData, new int[]{outputSize, inputSize});
        
        double[] biasData = new double[outputSize];
        for (int i = 0; i < biasData.length; i++) {
            biasData[i] = 0.01 * (i + 1);
        }
        Tensor newBias = new Tensor(biasData, new int[]{outputSize});
        
        // Set new weights and bias
        layer.setWeights(newWeights);
        layer.setBias(newBias);
        
        // Get the updated weights and bias
        Tensor updatedWeights = layer.getWeights();
        Tensor updatedBias = layer.getBias();
        
        // Verify that the weights and bias were updated
        assertArrayEquals(newWeights.getData(), updatedWeights.getData(), 0.0001);
        assertArrayEquals(newBias.getData(), updatedBias.getData(), 0.0001);
    }
    
    @Test
    public void testConvolutionalLayerWeightsAndBiases() {
        // Create a convolutional layer
        int[] inputShape = {1, 1, 8, 8}; // 1 batch, 1 channel, 8x8 input
        int numFilters = 2;
        int[] kernelSize = {3, 3};
        int[] stride = {1, 1};
        boolean padding = true;
        
        ConvolutionalLayer layer = new ConvolutionalLayer(
                inputShape, kernelSize, numFilters, stride, padding, ActivationFunctions.relu());
        
        // Get the initial kernels and bias
        Tensor initialKernels = layer.getKernels();
        Tensor initialBias = layer.getBias();
        
        // Verify dimensions - kernels should be [numFilters, inputChannels, kernelHeight, kernelWidth]
        assertArrayEquals(new int[]{numFilters, inputShape[1], kernelSize[0], kernelSize[1]}, 
                initialKernels.getShape());
        assertArrayEquals(new int[]{numFilters}, initialBias.getShape());
        
        // Create new kernels and bias tensors
        double[] kernelData = new double[numFilters * inputShape[1] * kernelSize[0] * kernelSize[1]];
        for (int i = 0; i < kernelData.length; i++) {
            kernelData[i] = 0.1 * (i + 1);
        }
        Tensor newKernels = new Tensor(kernelData, 
                new int[]{numFilters, inputShape[1], kernelSize[0], kernelSize[1]});
        
        double[] biasData = new double[numFilters];
        for (int i = 0; i < biasData.length; i++) {
            biasData[i] = 0.01 * (i + 1);
        }
        Tensor newBias = new Tensor(biasData, new int[]{numFilters});
        
        // Set new kernels and bias
        layer.setKernels(newKernels);
        layer.setBias(newBias);
        
        // Get the updated kernels and bias
        Tensor updatedKernels = layer.getKernels();
        Tensor updatedBias = layer.getBias();
        
        // Verify that the kernels and bias were updated
        assertArrayEquals(newKernels.getData(), updatedKernels.getData(), 0.0001);
        assertArrayEquals(newBias.getData(), updatedBias.getData(), 0.0001);
    }
    
    @Test
    public void testSetWeightsValidation() {
        // Create a standard layer
        StandardLayer layer = new StandardLayer(3, ActivationFunctions.relu());
        
        // Connect to another layer
        OutputLayer nextLayer = new OutputLayer(2, ActivationFunctions.tanh(), false);
        layer.connectTo(nextLayer);
        layer.initializeWeights(nextLayer.getSize());
        
        // Try to set weights with incorrect dimensions
        double[][] incorrectWeights = {
            {0.1}, // Should be [0.1, 0.2]
            {0.3}, // Should be [0.3, 0.4]
            {0.5}  // Should be [0.5, 0.6]
        };
        
        assertThrows(IllegalArgumentException.class, () -> {
            layer.setWeights(incorrectWeights);
        });
        
        // Try to set biases with incorrect dimension
        double[] incorrectBiases = {0.01, 0.02}; // Should be length 3
        
        assertThrows(IllegalArgumentException.class, () -> {
            layer.setBiases(incorrectBiases);
        });
    }
}