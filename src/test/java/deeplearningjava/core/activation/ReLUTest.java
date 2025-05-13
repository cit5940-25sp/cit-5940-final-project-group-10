package deeplearningjava.core.activation;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the ReLU activation function.
 */
public class ReLUTest {

    @Test
    public void testSingleton() {
        // Verify singleton pattern works properly
        ReLU instance1 = ReLU.getInstance();
        ReLU instance2 = ReLU.getInstance();
        
        // Both references should point to the same object
        assertSame(instance1, instance2, "ReLU should use singleton pattern");
    }

    @Test
    public void testApply() {
        ActivationFunction relu = ReLU.getInstance();
        
        // Positive input
        assertEquals(5.0, relu.apply(5.0), 0.0001, "ReLU.apply(5.0) should return 5.0");
        
        // Zero
        assertEquals(0.0, relu.apply(0.0), 0.0001, "ReLU.apply(0.0) should return 0.0");
        
        // Negative input
        assertEquals(0.0, relu.apply(-5.0), 0.0001, "ReLU.apply(-5.0) should return 0.0");
        
        // Large positive
        assertEquals(100.0, relu.apply(100.0), 0.0001, "ReLU.apply(100.0) should return 100.0");
        
        // Large negative
        assertEquals(0.0, relu.apply(-100.0), 0.0001, "ReLU.apply(-100.0) should return 0.0");
        
        // Small positive
        assertEquals(0.5, relu.apply(0.5), 0.0001, "ReLU.apply(0.5) should return 0.5");
        
        // Small negative
        assertEquals(0.0, relu.apply(-0.5), 0.0001, "ReLU.apply(-0.5) should return 0.0");
    }

    @Test
    public void testDerivative() {
        ActivationFunction relu = ReLU.getInstance();
        
        // Positive input: derivative is 1
        assertEquals(1.0, relu.derivative(5.0), 0.0001, "ReLU.derivative(5.0) should return 1.0");
        
        // Negative input: derivative is 0
        assertEquals(0.0, relu.derivative(-5.0), 0.0001, "ReLU.derivative(-5.0) should return 0.0");
        
        // Large positive
        assertEquals(1.0, relu.derivative(100.0), 0.0001, "ReLU.derivative(100.0) should return 1.0");
        
        // Large negative
        assertEquals(0.0, relu.derivative(-100.0), 0.0001, "ReLU.derivative(-100.0) should return 0.0");
    }
    
    @Test
    public void testDerivativeAtZero() {
        // The derivative at x=0 is undefined (discontinuity), but typically defined as 0
        ActivationFunction relu = ReLU.getInstance();
        assertEquals(0.0, relu.derivative(0.0), 0.0001,
                     "ReLU.derivative(0.0) should be defined as 0.0");
    }
    
    @Test
    public void testGetName() {
        ActivationFunction relu = ReLU.getInstance();
        assertEquals("ReLU", relu.getName(), "Name should be 'ReLU'");
    }
}