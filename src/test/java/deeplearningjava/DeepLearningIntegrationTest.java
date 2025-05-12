package deeplearningjava;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import deeplearningjava.api.Network;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.network.FeedForwardNetwork;
import othello.gamelogic.strategies.NetworkWrapper;

public class DeepLearningIntegrationTest {

    @Test
    public void testXORProblem() {
        // Create a simple network to solve the XOR problem
        int[] layerSizes = {2, 4, 1};
        Network network = new FeedForwardNetwork(
            layerSizes,
            ActivationFunctions.sigmoid(),
            ActivationFunctions.sigmoid(),
            false
        );
        
        // Create a wrapper for the network to provide a consistent API
        NetworkWrapper wrapper = new NetworkWrapper(network);
        
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
        
        // Train the network for a few iterations
        wrapper.setLearningRate(0.5);
        
        double initialError = calculateError(wrapper, inputs, targets);
        
        // Train for 1000 iterations
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < inputs.length; j++) {
                wrapper.trainingIteration(inputs[j], targets[j]);
            }
        }
        
        double finalError = calculateError(wrapper, inputs, targets);
        
        // The error should decrease after training
        assertTrue(finalError < initialError);
        
        // Test predictions - they should be closer to the targets than random guessing
        for (int i = 0; i < inputs.length; i++) {
            double[] output = wrapper.feedForward(inputs[i]);
            
            if (targets[i][0] > 0.5) {
                // For cases where target is 1
                assertTrue(output[0] > 0.3);
            } else {
                // For cases where target is 0
                assertTrue(output[0] < 0.7);
            }
        }
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
    
    @Test
    public void testSoftmaxOutput() {
        // Create a network with softmax output layer for classification
        int[] layerSizes = {2, 3, 3};  // 3 output classes
        Network network = new FeedForwardNetwork(
            layerSizes,
            ActivationFunctions.relu(),
            ActivationFunctions.linear(),
            true  // Use softmax output
        );
        
        // Create a wrapper for the network
        NetworkWrapper wrapper = new NetworkWrapper(network);
        
        // Test data
        double[] input = {0.5, 0.7};
        
        // Feed forward and check the output properties
        double[] output = wrapper.feedForward(input);
        
        // Output should be valid probability distribution
        assertEquals(3, output.length);
        
        // Check that outputs sum approximately to 1.0
        double sum = 0;
        for (double val : output) {
            assertTrue(val >= 0 && val <= 1);
            sum += val;
        }
        assertEquals(1.0, sum, 1e-6);
    }
}