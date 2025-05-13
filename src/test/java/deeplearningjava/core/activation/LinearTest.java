package deeplearningjava.core.activation;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the Linear activation function.
 */
public class LinearTest {

    @Test
    public void testSingleton() {
        // Verify singleton pattern works properly
        Linear instance1 = Linear.getInstance();
        Linear instance2 = Linear.getInstance();
        
        // Both references should point to the same object
        assertSame(instance1, instance2, "Linear should use singleton pattern");
    }

    @Test
    public void testApply() {
        ActivationFunction linear = Linear.getInstance();
        
        // Zero input
        assertEquals(0.0, linear.apply(0.0), 0.0001, "Linear.apply(0.0) should return exactly 0.0");
        
        // Positive input
        assertEquals(1.0, linear.apply(1.0), 0.0001, "Linear.apply(1.0) should return exactly 1.0");
        
        // Negative input
        assertEquals(-1.0, linear.apply(-1.0), 0.0001, "Linear.apply(-1.0) should return exactly -1.0");
        
        // Larger positive
        assertEquals(123.456, linear.apply(123.456), 0.0001, "Linear.apply(123.456) should return exactly 123.456");
        
        // Larger negative
        assertEquals(-98.765, linear.apply(-98.765), 0.0001, "Linear.apply(-98.765) should return exactly -98.765");
    }

    @Test
    public void testDerivative() {
        // The derivative of the linear function is always 1
        ActivationFunction linear = Linear.getInstance();
        
        // Test with various inputs, all should return 1.0
        assertEquals(1.0, linear.derivative(0.0), 0.0001, "Linear.derivative(0.0) should always return 1.0");
        assertEquals(1.0, linear.derivative(1.0), 0.0001, "Linear.derivative(1.0) should always return 1.0");
        assertEquals(1.0, linear.derivative(-1.0), 0.0001, "Linear.derivative(-1.0) should always return 1.0");
        assertEquals(1.0, linear.derivative(123.456), 0.0001, "Linear.derivative(123.456) should always return 1.0");
        assertEquals(1.0, linear.derivative(-98.765), 0.0001, "Linear.derivative(-98.765) should always return 1.0");
    }
    
    @Test
    public void testInputEqualsOutput() {
        // Linear function should always output the same value as input
        ActivationFunction linear = Linear.getInstance();
        
        for (double x = -100; x <= 100; x += 10) {
            assertEquals(x, linear.apply(x), 0.0001,
                        "Linear activation should output the input value");
        }
        
        // Test with some non-integer values
        double[] testValues = {0.1, 0.25, 0.5, 0.75, -0.1, -0.25, -0.5, -0.75};
        for (double x : testValues) {
            assertEquals(x, linear.apply(x), 0.0001,
                        "Linear activation should output the input value");
        }
    }
    
    @Test
    public void testGetName() {
        ActivationFunction linear = Linear.getInstance();
        assertEquals("Linear", linear.getName(), "Name should be 'Linear'");
    }
}