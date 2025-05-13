package deeplearningjava.core.activation;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
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

    @ParameterizedTest
    @CsvSource({
        "5.0, 5.0", // Positive input
        "0.0, 0.0", // Zero
        "-5.0, 0.0", // Negative input
        "100.0, 100.0", // Large positive
        "-100.0, 0.0", // Large negative
        "0.5, 0.5", // Small positive
        "-0.5, 0.0"  // Small negative
    })
    public void testApply(double input, double expected) {
        ActivationFunction relu = ReLU.getInstance();
        assertEquals(expected, relu.apply(input), 0.0001, 
                     "ReLU.apply(" + input + ") should return " + expected);
    }

    @ParameterizedTest
    @CsvSource({
        "5.0, 1.0", // Positive input: derivative is 1
        "-5.0, 0.0", // Negative input: derivative is 0
        "100.0, 1.0", // Large positive
        "-100.0, 0.0", // Large negative
    })
    public void testDerivative(double input, double expected) {
        ActivationFunction relu = ReLU.getInstance();
        assertEquals(expected, relu.derivative(input), 0.0001,
                     "ReLU.derivative(" + input + ") should return " + expected);
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