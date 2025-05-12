package deeplearningjava;

import deeplearningjava.api.Network;
import deeplearningjava.api.TensorNetwork;
import deeplearningjava.factory.OnnxNetworkLoader;
import deeplearningjava.layer.StandardLayer;
import deeplearningjava.layer.OutputLayer;
import deeplearningjava.layer.tensor.FullyConnectedLayer;
import deeplearningjava.network.DenseNetwork;
import deeplearningjava.onnx.OnnxModelLoader;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * Test class to verify that an ONNX model can be loaded correctly
 * and that the weights and biases can be accessed.
 * This test does not require JavaFX to run.
 */
public class OnnxModelLoaderTest {

    @Test
    public void testLoadOnnxModel() {
        // Get the path to the ONNX model
        String modelPath = OnnxNetworkLoader.getDefaultOthelloModelPath();
        System.out.println("Loading ONNX model from: " + modelPath);
        
        try {
            // Load the model using the OnnxModelLoader
            OnnxModelLoader modelLoader = new OnnxModelLoader(modelPath);
            boolean loaded = modelLoader.load();
            
            // Verify the model loaded successfully
            assertTrue(loaded, "Model should be loaded successfully");
            
            // Print the detected layer sizes
            int[] layerSizes = modelLoader.getLayerSizesArray();
            System.out.println("Detected layer sizes: " + java.util.Arrays.toString(layerSizes));
            
            // Create a vector network from the model
            DenseNetwork vectorNetwork = modelLoader.createVectorNetwork();
            assertNotNull(vectorNetwork, "Vector network should be created");
            
            // Print information about the network structure
            List<deeplearningjava.api.Layer> layers = vectorNetwork.getLayers();
            System.out.println("Network has " + layers.size() + " layers");
            
            // Inspect each layer in the network
            for (int i = 0; i < layers.size(); i++) {
                deeplearningjava.api.Layer layer = layers.get(i);
                System.out.println("Layer " + i + " type: " + layer.getClass().getSimpleName());
                
                // For standard layers (hidden layers), check weights and biases
                if (layer instanceof StandardLayer) {
                    StandardLayer stdLayer = (StandardLayer) layer;
                    double[][] weights = stdLayer.getWeights();
                    double[] biases = stdLayer.getBiases();
                    
                    System.out.println("  - Weights shape: " + weights.length + "x" + 
                                       (weights.length > 0 ? weights[0].length : 0));
                    System.out.println("  - Biases length: " + biases.length);
                    
                    // Print a sample of weights for verification
                    if (weights.length > 0 && weights[0].length > 0) {
                        System.out.println("  - Sample weights: " + 
                                           weights[0][0] + ", " + 
                                           (weights[0].length > 1 ? weights[0][1] : "N/A"));
                    }
                }
                
                // For output layer, check weights and biases
                if (layer instanceof OutputLayer) {
                    OutputLayer outLayer = (OutputLayer) layer;
                    double[][] weights = outLayer.getWeights();
                    double[] biases = outLayer.getBiases();
                    
                    System.out.println("  - Output weights shape: " + weights.length + "x" + 
                                       (weights.length > 0 ? weights[0].length : 0));
                    System.out.println("  - Output biases length: " + biases.length);
                }
            }
            
            // Create a tensor network from the model (for multi-dimensional input)
            int[] inputShape = {3, 8, 8}; // 3 channels, 8x8 board
            DenseNetwork tensorNetwork = modelLoader.createTensorNetwork(inputShape);
            assertNotNull(tensorNetwork, "Tensor network should be created");
            
            // Print information about the tensor network structure
            List<deeplearningjava.api.TensorLayer> tensorLayers = tensorNetwork.getTensorLayers();
            System.out.println("\nTensor Network has " + tensorLayers.size() + " layers");
            
            // Inspect each layer in the tensor network
            for (int i = 0; i < tensorLayers.size(); i++) {
                deeplearningjava.api.TensorLayer layer = tensorLayers.get(i);
                System.out.println("Layer " + i + " type: " + layer.getClass().getSimpleName());
                
                // For fully connected layers, check weights and biases
                if (layer instanceof FullyConnectedLayer) {
                    FullyConnectedLayer fcLayer = (FullyConnectedLayer) layer;
                    
                    // Get weights tensor
                    deeplearningjava.core.tensor.Tensor weights = fcLayer.getWeights();
                    int[] weightShape = weights.getShape();
                    
                    // Get bias tensor
                    deeplearningjava.core.tensor.Tensor bias = fcLayer.getBias();
                    int[] biasShape = bias.getShape();
                    
                    System.out.println("  - Weights shape: " + java.util.Arrays.toString(weightShape));
                    System.out.println("  - Bias shape: " + java.util.Arrays.toString(biasShape));
                    
                    // Print a sample of weights for verification
                    if (weights.getSize() > 0) {
                        System.out.println("  - Sample weights: " + weights.get(0) + 
                                          (weights.getSize() > 1 ? ", " + weights.get(1) : ""));
                    }
                }
            }
            
            // Test forward pass through the network with random input
            System.out.println("\nTesting forward pass with random input:");
            
            // For vector network
            double[] inputVector = new double[layerSizes[0]];
            for (int i = 0; i < inputVector.length; i++) {
                inputVector[i] = Math.random(); // Random values between 0 and 1
            }
            
            double[] outputVector = vectorNetwork.forward(inputVector);
            System.out.println("Vector network output: " + java.util.Arrays.toString(outputVector));
            
            // For tensor network
            double[] flatTensorInput = new double[inputShape[0] * inputShape[1] * inputShape[2]];
            for (int i = 0; i < flatTensorInput.length; i++) {
                flatTensorInput[i] = Math.random(); // Random values between 0 and 1
            }
            
            deeplearningjava.core.tensor.Tensor inputTensor = 
                new deeplearningjava.core.tensor.Tensor(flatTensorInput, inputShape);
            
            deeplearningjava.core.tensor.Tensor outputTensor = tensorNetwork.forward(inputTensor);
            System.out.println("Tensor network output shape: " + java.util.Arrays.toString(outputTensor.getShape()));
            System.out.println("Tensor network output: " + outputTensor.get(0));
            
        } catch (IOException e) {
            fail("Exception occurred while loading model: " + e.getMessage());
        }
    }
    
    @Test
    public void testOnnxNetworkLoader() {
        // Test the OnnxNetworkLoader utility methods
        String modelPath = OnnxNetworkLoader.getDefaultOthelloModelPath();
        System.out.println("Default Othello model path: " + modelPath);
        
        try {
            // Load vector network
            Network network = OnnxNetworkLoader.loadNetwork(modelPath);
            assertNotNull(network, "Network should be loaded successfully");
            System.out.println("Successfully loaded vector network");
            
            // Test the network with random input
            int inputSize = network.getInputSize();
            double[] input = new double[inputSize];
            for (int i = 0; i < input.length; i++) {
                input[i] = Math.random();
            }
            
            double[] output = network.forward(input);
            System.out.println("Vector network prediction: " + java.util.Arrays.toString(output));
            
            // Load tensor network for Othello
            TensorNetwork tensorNetwork = OnnxNetworkLoader.loadOthelloNetwork(modelPath, 8, 3);
            assertNotNull(tensorNetwork, "Tensor network should be loaded successfully");
            System.out.println("Successfully loaded tensor network for Othello");
            
            // Test the tensor network with random input
            int[] inputShape = {3, 8, 8};
            double[] flatInput = new double[3 * 8 * 8];
            for (int i = 0; i < flatInput.length; i++) {
                flatInput[i] = Math.random();
            }
            
            deeplearningjava.core.tensor.Tensor inputTensor = 
                new deeplearningjava.core.tensor.Tensor(flatInput, inputShape);
            
            deeplearningjava.core.tensor.Tensor outputTensor = tensorNetwork.forward(inputTensor);
            System.out.println("Tensor network output shape: " + java.util.Arrays.toString(outputTensor.getShape()));
            System.out.println("Tensor network prediction: " + outputTensor.get(0));
            
        } catch (IOException e) {
            fail("Exception occurred while using OnnxNetworkLoader: " + e.getMessage());
        }
    }
}