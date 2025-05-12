package othello.gamelogic.strategies;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import deeplearningjava.api.Network;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.network.FeedForwardNetwork;

/**
 * Tests for the NetworkWrapper class.
 */
public class NetworkWrapperTest {
    
    private Network networkMock;
    private NetworkWrapper wrapper;
    
    @BeforeEach
    public void setUp() {
        // Create a simple feed-forward network for testing
        int[] layerSizes = {2, 3, 1};
        networkMock = new FeedForwardNetwork(
            layerSizes,
            ActivationFunctions.sigmoid(),
            ActivationFunctions.linear(),
            false
        );
        
        wrapper = new NetworkWrapper(networkMock);
    }
    
    @Test
    public void testConstructor() {
        // Test that the wrapper is created successfully
        assertNotNull(wrapper);
        
        // Test that it contains the correct network
        assertEquals(networkMock, wrapper.getNetwork());
    }
    
    @Test
    public void testGetSetLearningRate() {
        // Set learning rate to a known value
        double learningRate = 0.05;
        wrapper.setLearningRate(learningRate);
        
        // Verify the wrapper's getLearningRate returns the expected value
        assertEquals(learningRate, wrapper.getLearningRate(), 1e-6);
        
        // Verify that the underlying network's learning rate was set correctly
        assertEquals(learningRate, networkMock.getLearningRate(), 1e-6);
        
        // Test with a different learning rate
        double newLearningRate = 0.01;
        wrapper.setLearningRate(newLearningRate);
        assertEquals(newLearningRate, wrapper.getLearningRate(), 1e-6);
    }
    
    @Test
    public void testFeedForward() {
        // Test the feedForward method delegates correctly to the network's forward method
        double[] inputs = {0.5, 0.7};
        
        // Get the result from the wrapper
        double[] wrapperOutput = wrapper.feedForward(inputs);
        
        // Get the result directly from the network for comparison
        double[] networkOutput = networkMock.forward(inputs);
        
        // Verify outputs match
        assertArrayEquals(networkOutput, wrapperOutput, 1e-6);
        
        // Basic validation of output properties
        assertNotNull(wrapperOutput);
        assertEquals(1, wrapperOutput.length); // Based on the network structure
    }
    
    @Test
    public void testTrainingIteration() {
        // Test the trainingIteration method delegates correctly to the network's train method
        double[] inputs = {0.5, 0.7};
        double[] targets = {1.0};
        
        // Get the result from the wrapper
        double[] wrapperOutput = wrapper.trainingIteration(inputs, targets);
        
        // Get the result directly from the network for comparison
        double[] networkOutput = networkMock.train(inputs, targets);
        
        // Since we're calling trainingIteration, the outputs might differ slightly due to 
        // network weight initialization, but they should both be valid outputs
        assertNotNull(wrapperOutput);
        assertEquals(networkOutput.length, wrapperOutput.length);
        assertEquals(targets.length, wrapperOutput.length);
    }
    
    @Test
    public void testWithLargerNetwork() {
        // Test with a more complex network
        int[] largerLayerSizes = {4, 8, 6, 2};
        Network largerNetwork = new FeedForwardNetwork(
            largerLayerSizes,
            ActivationFunctions.relu(),
            ActivationFunctions.linear(),
            true
        );
        
        NetworkWrapper largerWrapper = new NetworkWrapper(largerNetwork);
        
        // Test network getters/setters
        assertEquals(largerNetwork, largerWrapper.getNetwork());
        
        // Test with sample inputs and targets
        double[] inputs = {0.1, 0.2, 0.3, 0.4};
        double[] targets = {1.0, 0.0};
        
        // Perform feedforward
        double[] outputs = largerWrapper.feedForward(inputs);
        
        // Check basic properties
        assertNotNull(outputs);
        assertEquals(2, outputs.length);
        
        // Since this uses softmax output, check probability distribution properties
        double sum = 0;
        for (double val : outputs) {
            assertTrue(val >= 0 && val <= 1);
            sum += val;
        }
        assertEquals(1.0, sum, 1e-6);
        
        // Test training
        double[] trainingOutputs = largerWrapper.trainingIteration(inputs, targets);
        assertNotNull(trainingOutputs);
        assertEquals(2, trainingOutputs.length);
    }
    
    @Test
    public void testXORProblem() {
        // Create a network for the XOR problem
        int[] layerSizes = {2, 4, 1};
        Network xorNetwork = new FeedForwardNetwork(
            layerSizes,
            ActivationFunctions.sigmoid(),
            ActivationFunctions.sigmoid(),
            false
        );
        
        NetworkWrapper xorWrapper = new NetworkWrapper(xorNetwork);
        
        // Set a higher learning rate for faster convergence in this test
        xorWrapper.setLearningRate(0.5);
        
        // XOR training data
        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        
        double[][] targets = {
            {0},
            {1},
            {1},
            {0}
        };
        
        // Train for a few iterations
        double initialError = calculateError(xorWrapper, inputs, targets);
        
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < inputs.length; j++) {
                xorWrapper.trainingIteration(inputs[j], targets[j]);
            }
        }
        
        double finalError = calculateError(xorWrapper, inputs, targets);
        
        // Error should decrease after training
        assertTrue(finalError < initialError, 
                  "Error should decrease after training. Initial: " + initialError + ", Final: " + finalError);
    }
    
    private double calculateError(NetworkWrapper wrapper, double[][] inputs, double[][] targets) {
        double totalError = 0;
        
        for (int i = 0; i < inputs.length; i++) {
            double[] output = wrapper.feedForward(inputs[i]);
            double error = Math.pow(output[0] - targets[i][0], 2);
            totalError += error;
        }
        
        return totalError / inputs.length;
    }
}