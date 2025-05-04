package deeplearningjava;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class DeepLearningIntegrationTest {

    @Test
    public void testXORProblem() {
        // Create a simple network to solve the XOR problem
        int[] layerSizes = {2, 4, 1};
        Network network = new Network(
            layerSizes,
            Node.SIGMOID,
            Node.SIGMOID_DERIVATIVE,
            Node.SIGMOID,
            Node.SIGMOID_DERIVATIVE,
            false
        );
        
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
        network.setLearningRate(0.5);
        
        double initialError = calculateError(network, inputs, targets);
        
        // Train for 1000 iterations
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < inputs.length; j++) {
                network.trainingIteration(inputs[j], targets[j]);
            }
        }
        
        double finalError = calculateError(network, inputs, targets);
        
        // The error should decrease after training
        assertTrue(finalError < initialError);
        
        // Test predictions - they should be closer to the targets than random guessing
        for (int i = 0; i < inputs.length; i++) {
            double[] output = network.feedForward(inputs[i]);
            
            if (targets[i][0] > 0.5) {
                // For cases where target is 1
                assertTrue(output[0] > 0.3);
            } else {
                // For cases where target is 0
                assertTrue(output[0] < 0.7);
            }
        }
    }
    
    private double calculateError(Network network, double[][] inputs, double[][] targets) {
        double totalError = 0;
        
        for (int i = 0; i < inputs.length; i++) {
            double[] output = network.feedForward(inputs[i]);
            double error = Math.pow(output[0] - targets[i][0], 2);
            totalError += error;
        }
        
        return totalError / inputs.length;
    }
    
    @Test
    public void testSoftmaxOutput() {
        // Create a network with softmax output layer for classification
        int[] layerSizes = {2, 3, 3};  // 3 output classes
        Network network = new Network(
            layerSizes,
            Node.RELU,
            Node.RELU_DERIVATIVE,
            null, // Not used with softmax
            null, // Not used with softmax
            true  // Use softmax output
        );
        
        // Test data
        double[] input = {0.5, 0.7};
        
        // Feed forward and check the output properties
        double[] output = network.feedForward(input);
        
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