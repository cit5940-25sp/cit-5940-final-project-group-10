package deeplearningjava.network;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import deeplearningjava.api.BaseNetwork;

/**
 * Tests for the AbstractNetwork class.
 */
public class AbstractNetworkTest {
    
    /**
     * Simple concrete implementation of AbstractNetwork for testing.
     */
    private static class TestNetwork extends AbstractNetwork {
        private int layerCount = 0;
        private boolean initialized = false;
        
        public TestNetwork(NetworkType type) {
            super(type);
        }
        
        @Override
        public int getLayerCount() {
            return layerCount;
        }
        
        @Override
        public boolean isInitialized() {
            return initialized;
        }
        
        @Override
        protected void appendDetails(StringBuilder summary) {
            summary.append("Test Network Details");
        }
        
        // For testing
        public void setLayerCount(int count) {
            this.layerCount = count;
        }
        
        public void setInitialized(boolean initialized) {
            this.initialized = initialized;
        }
        
        // For testing validateNetwork
        public void testValidateNetwork() {
            validateNetwork();
        }
        
        // For testing validateBatchParameters
        public void testValidateBatchParameters(Object[] inputs, Object[] targets, int epochs) {
            validateBatchParameters(inputs, targets, epochs);
        }
    }
    
    @Test
    public void testConstructor() {
        TestNetwork network = new TestNetwork(BaseNetwork.NetworkType.DENSE);
        
        assertNotNull(network);
        assertEquals(BaseNetwork.NetworkType.DENSE, network.getType());
        assertEquals(0.01, network.getLearningRate(), 0.0001); // Default learning rate
    }
    
    @Test
    public void testSetLearningRate() {
        TestNetwork network = new TestNetwork(BaseNetwork.NetworkType.DENSE);
        
        network.setLearningRate(0.05);
        assertEquals(0.05, network.getLearningRate(), 0.0001);
        
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            network.setLearningRate(0);
        });
        
        assertTrue(exception.getMessage().contains("must be positive"));
    }
    
    @Test
    public void testGetSummary() {
        TestNetwork network = new TestNetwork(BaseNetwork.NetworkType.CUSTOM);
        network.setLayerCount(3);
        
        String summary = network.getSummary();
        
        assertTrue(summary.contains("CUSTOM Network Summary"));
        assertTrue(summary.contains("Layer Count: 3"));
        assertTrue(summary.contains("Learning Rate: 0.01"));
        assertTrue(summary.contains("Test Network Details"));
    }
    
    @Test
    public void testValidateNetwork() {
        TestNetwork network = new TestNetwork(BaseNetwork.NetworkType.DENSE);
        
        // Test with uninitialized network
        network.setInitialized(false);
        
        Exception exception = assertThrows(IllegalStateException.class, () -> {
            network.testValidateNetwork();
        });
        
        assertTrue(exception.getMessage().contains("has not been initialized"));
        
        // Test with initialized network
        network.setInitialized(true);
        
        // Should not throw an exception
        network.testValidateNetwork();
    }
    
    @Test
    public void testValidateBatchParameters() {
        TestNetwork network = new TestNetwork(BaseNetwork.NetworkType.CONVOLUTIONAL);
        
        // Valid parameters
        Double[] inputs = {1.0, 2.0};
        Double[] targets = {0.5, 0.7};
        
        // Should not throw an exception
        network.testValidateBatchParameters(inputs, targets, 10);
        
        // Test with null inputs
        Exception exception1 = assertThrows(NullPointerException.class, () -> {
            network.testValidateBatchParameters(null, targets, 10);
        });
        
        assertTrue(exception1.getMessage().contains("inputs must not be null"));
        
        // Test with null targets
        Exception exception2 = assertThrows(NullPointerException.class, () -> {
            network.testValidateBatchParameters(inputs, null, 10);
        });
        
        assertTrue(exception2.getMessage().contains("targets must not be null"));
        
        // Test with empty inputs
        Exception exception3 = assertThrows(IllegalArgumentException.class, () -> {
            network.testValidateBatchParameters(new Double[0], targets, 10);
        });
        
        assertTrue(exception3.getMessage().contains("inputs cannot be empty"));
        
        // Test with mismatched inputs/targets lengths
        Exception exception4 = assertThrows(IllegalArgumentException.class, () -> {
            network.testValidateBatchParameters(inputs, new Double[]{0.5}, 10);
        });
        
        assertTrue(exception4.getMessage().contains("Number of inputs"));
        
        // Test with non-positive epochs
        Exception exception5 = assertThrows(IllegalArgumentException.class, () -> {
            network.testValidateBatchParameters(inputs, targets, 0);
        });
        
        assertTrue(exception5.getMessage().contains("epochs must be positive"));
    }
}