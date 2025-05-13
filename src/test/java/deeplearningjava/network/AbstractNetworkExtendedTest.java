package deeplearningjava.network;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

/**
 * Extended tests for the AbstractNetwork class.
 */
public class AbstractNetworkExtendedTest {
    
    // Simple concrete implementation for testing AbstractNetwork
    private static class TestNetwork extends AbstractNetwork {
        
        public TestNetwork(NetworkType type) {
            super(type);
        }
        
        @Override
        protected void appendDetails(StringBuilder summary) {
            summary.append("Test Network Details");
        }
        
        @Override
        public boolean isInitialized() {
            return true; // Always return true for testing
        }
        
        @Override
        public int getLayerCount() {
            return 3; // Fixed for testing
        }
    }
    
    @Test
    public void testConstructor() {
        AbstractNetwork network = new TestNetwork(AbstractNetwork.NetworkType.DENSE);
        assertNotNull(network);
        assertEquals(AbstractNetwork.NetworkType.DENSE, network.getType());
        assertEquals(0.01, network.getLearningRate(), 0.0001); // Default learning rate
    }
    
    @Test
    public void testNullNetworkTypeThrowsException() {
        Exception exception = assertThrows(NullPointerException.class, () -> {
            new TestNetwork(null);
        });
        
        assertTrue(exception.getMessage().contains("networkType must not be null"));
    }
    
    @Test
    public void testSetValidLearningRate_0_1() {
        double rate = 0.1;
        AbstractNetwork network = new TestNetwork(AbstractNetwork.NetworkType.DENSE);
        network.setLearningRate(rate);
        assertEquals(rate, network.getLearningRate(), 0.0001);
    }
    
    @Test
    public void testSetValidLearningRate_0_01() {
        double rate = 0.01;
        AbstractNetwork network = new TestNetwork(AbstractNetwork.NetworkType.DENSE);
        network.setLearningRate(rate);
        assertEquals(rate, network.getLearningRate(), 0.0001);
    }
    
    @Test
    public void testSetValidLearningRate_0_001() {
        double rate = 0.001;
        AbstractNetwork network = new TestNetwork(AbstractNetwork.NetworkType.DENSE);
        network.setLearningRate(rate);
        assertEquals(rate, network.getLearningRate(), 0.0001);
    }
    
    @Test
    public void testSetValidLearningRate_1_0() {
        double rate = 1.0;
        AbstractNetwork network = new TestNetwork(AbstractNetwork.NetworkType.DENSE);
        network.setLearningRate(rate);
        assertEquals(rate, network.getLearningRate(), 0.0001);
    }
    
    @Test
    public void testSetInvalidLearningRateThrowsException_0_0() {
        double rate = 0.0;
        AbstractNetwork network = new TestNetwork(AbstractNetwork.NetworkType.DENSE);
        
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            network.setLearningRate(rate);
        });
        
        assertTrue(exception.getMessage().contains("Learning rate must be positive"));
    }
    
    @Test
    public void testSetInvalidLearningRateThrowsException_negative_0_1() {
        double rate = -0.1;
        AbstractNetwork network = new TestNetwork(AbstractNetwork.NetworkType.DENSE);
        
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            network.setLearningRate(rate);
        });
        
        assertTrue(exception.getMessage().contains("Learning rate must be positive"));
    }
    
    @Test
    public void testSetInvalidLearningRateThrowsException_negative_1_0() {
        double rate = -1.0;
        AbstractNetwork network = new TestNetwork(AbstractNetwork.NetworkType.DENSE);
        
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            network.setLearningRate(rate);
        });
        
        assertTrue(exception.getMessage().contains("Learning rate must be positive"));
    }
    
    @Test
    public void testGetSummary() {
        AbstractNetwork network = new TestNetwork(AbstractNetwork.NetworkType.DENSE);
        network.setLearningRate(0.05);
        
        String summary = network.getSummary();
        
        // Check summary includes expected information
        assertTrue(summary.contains("DENSE Network Summary"));
        assertTrue(summary.contains("Layer Count: 3"));
        assertTrue(summary.contains("Learning Rate: 0.05"));
        assertTrue(summary.contains("Test Network Details"));
    }
    
    @Test
    public void testValidateNetwork() {
        AbstractNetwork validNetwork = new TestNetwork(AbstractNetwork.NetworkType.DENSE);
        
        // Should not throw an exception
        validNetwork.validateNetwork();
        
        // Create a network that returns false for isInitialized
        AbstractNetwork invalidNetwork = new AbstractNetwork(AbstractNetwork.NetworkType.DENSE) {
            @Override
            protected void appendDetails(StringBuilder summary) {
                // No details
            }
            
            @Override
            public boolean isInitialized() {
                return false;
            }
            
            @Override
            public int getLayerCount() {
                return 0;
            }
        };
        
        Exception exception = assertThrows(IllegalStateException.class, () -> {
            invalidNetwork.validateNetwork();
        });
        
        assertTrue(exception.getMessage().contains("Network has not been initialized"));
    }
    
    @Test
    public void testValidateBatchParameters() {
        AbstractNetwork network = new TestNetwork(AbstractNetwork.NetworkType.DENSE);
        
        // Valid parameters
        Double[] inputs = {1.0, 2.0, 3.0};
        Double[] targets = {0.1, 0.2, 0.3};
        int epochs = 10;
        
        // Should not throw an exception
        network.validateBatchParameters(inputs, targets, epochs);
        
        // Test various invalid scenarios
        
        // Null inputs
        assertThrows(NullPointerException.class, () -> {
            network.validateBatchParameters(null, targets, epochs);
        });
        
        // Null targets
        assertThrows(NullPointerException.class, () -> {
            network.validateBatchParameters(inputs, null, epochs);
        });
        
        // Empty inputs
        assertThrows(IllegalArgumentException.class, () -> {
            network.validateBatchParameters(new Double[0], targets, epochs);
        });
        
        // Mismatched lengths
        assertThrows(IllegalArgumentException.class, () -> {
            network.validateBatchParameters(inputs, new Double[]{0.1, 0.2}, epochs);
        });
        
        // Invalid epochs
        assertThrows(IllegalArgumentException.class, () -> {
            network.validateBatchParameters(inputs, targets, 0);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            network.validateBatchParameters(inputs, targets, -1);
        });
    }
    
    @Test
    public void testNetworkTypes() {
        // Test all network types can be used
        for (AbstractNetwork.NetworkType type : AbstractNetwork.NetworkType.values()) {
            AbstractNetwork network = new TestNetwork(type);
            assertEquals(type, network.getType());
            
            // Verify summary contains the type
            String summary = network.getSummary();
            assertTrue(summary.contains(type.toString()));
        }
    }
}