package deeplearningjava;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.Test;

import deeplearningjava.api.Network;
import deeplearningjava.api.TensorNetwork;
import deeplearningjava.core.tensor.Tensor;
import deeplearningjava.factory.OnnxNetworkLoader;
import deeplearningjava.network.DenseNetwork;
import deeplearningjava.onnx.OnnxModelLoader;

/**
 * Test for loading ONNX models and verifying weight/bias loading.
 */
public class OnnxModelLoadTest {
    
    private static final String DEFAULT_ONNX_PATH = System.getProperty("user.dir") + "/models/othello.onnx";
    private static final String ALTERNATE_ONNX_PATH = System.getProperty("user.dir") + 
            "/src/main/java/othello/models/othello_2d_2_hidden_layer_dense_model.onnx";
    
    @Test
    public void testOnnxModelLoaderArchitecture() throws IOException {
        // Load ONNX model
        OnnxModelLoader loader = new OnnxModelLoader(DEFAULT_ONNX_PATH);
        boolean loaded = loader.load();
        
        System.out.println("Using ONNX model: " + DEFAULT_ONNX_PATH);
        System.out.println("Model loaded successfully: " + loaded);
        
        // Get layer sizes
        System.out.println("Detected layer sizes: " + loader.getLayerSizes());
        
        // Check if weights were found
        System.out.println("Model has weights: " + loader.hasWeights());
        
        if (loaded) {
            // Create network from the model
            DenseNetwork network = loader.createVectorNetwork();
            
            // Print network information
            System.out.println("Network created with " + network.getLayerCount() + " layers");
            System.out.println("Layer sizes: ");
            for (int i = 0; i < network.getLayers().size(); i++) {
                System.out.println("  Layer " + i + ": " + network.getLayers().get(i).getSize() + " nodes");
            }
            
            // Create a sample input and test the network
            double[] input = new double[network.getInputSize()];
            Random random = new Random(42); // Fixed seed for reproducibility
            for (int i = 0; i < input.length; i++) {
                input[i] = random.nextDouble() * 2 - 1; // Values between -1 and 1
            }
            
            // Run an inference
            double[] output = network.forward(input);
            System.out.println("Test inference result: " + Arrays.toString(output));
        }
    }
    
    @Test
    public void testOnnxNetworkLoaderForOthello() throws IOException {
        System.out.println("Testing OnnxNetworkLoader.loadOthelloNetwork()");
        
        // Standard Othello parameters
        int boardSize = 8;
        int channels = 1;
        
        System.out.println("Loading Othello network with boardSize=" + boardSize + 
                ", channels=" + channels + " from " + DEFAULT_ONNX_PATH);
        
        // Load network using the factory loader
        TensorNetwork network = OnnxNetworkLoader.loadOthelloNetwork(DEFAULT_ONNX_PATH, boardSize, channels);
        
        System.out.println("Successfully loaded network: " + network.getClass().getSimpleName());
        System.out.println("Network has " + network.getLayerCount() + " layers");
        System.out.println("Input shape: " + Arrays.toString(network.getInputShape()));
        System.out.println("Output shape: " + Arrays.toString(network.getOutputShape()));
        
        // Create a sample input tensor with board dimensions
        Tensor input = new Tensor(new double[channels * boardSize * boardSize], 
                                  new int[]{1, channels, boardSize, boardSize});
        Random random = new Random(42);
        double[] data = input.getData();
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextDouble() * 2 - 1; // Values between -1 and 1
        }
        
        // Run tensor inference
        Tensor output = network.forward(input);
        System.out.println("Tensor inference result shape: " + Arrays.toString(output.getShape()));
        System.out.println("Tensor inference result: " + Arrays.toString(output.getData()));
    }
}