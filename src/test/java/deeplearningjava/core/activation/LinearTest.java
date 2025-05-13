package deeplearningjava.core.activation;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
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

    @ParameterizedTest
    @CsvSource({
        "0.0, 0.0", // Zero input
        "1.0, 1.0", // Positive input
        "-1.0, -1.0", // Negative input
        "123.456, 123.456", // Larger positive
        "-98.765, -98.765"  // Larger negative
    })
    public void testApply(double input, double expected) {
        ActivationFunction linear = Linear.getInstance();
        assertEquals(expected, linear.apply(input), 0.0001, 
                     "Linear.apply(" + input + ") should return exactly " + expected);
    }

    @ParameterizedTest
    @CsvSource({
        "0.0, 1.0",
        "1.0, 1.0",
        "-1.0, 1.0",
        "123.456, 1.0",
        "-98.765, 1.0"
    })
    public void testDerivative(double input, double expected) {
        // The derivative of the linear function is always 1
        ActivationFunction linear = Linear.getInstance();
        assertEquals(expected, linear.derivative(input), 0.0001,
                     "Linear.derivative(" + input + ") should always return 1.0");
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