package deeplearningjava.core.activation;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the Tanh activation function.
 */
public class TanhTest {

    @Test
    public void testSingleton() {
        // Verify singleton pattern works properly
        Tanh instance1 = Tanh.getInstance();
        Tanh instance2 = Tanh.getInstance();
        
        // Both references should point to the same object
        assertSame(instance1, instance2, "Tanh should use singleton pattern");
    }

    @ParameterizedTest
    @CsvSource({
        "0.0, 0.0", // Input 0, output 0
        "1.0, 0.7615941559", // Small positive
        "-1.0, -0.7615941559", // Small negative
        "2.0, 0.9640275801", // Larger positive
        "-2.0, -0.9640275801"  // Larger negative
    })
    public void testApply(double input, double expected) {
        ActivationFunction tanh = Tanh.getInstance();
        assertEquals(expected, tanh.apply(input), 0.0001, 
                     "Tanh.apply(" + input + ") should return approximately " + expected);
    }
    
    @Test
    public void testApplyLargeValues() {
        ActivationFunction tanh = Tanh.getInstance();
        // Very large positive value should approach 1.0
        assertTrue(tanh.apply(100.0) > 0.999, "Tanh of large positive should approach 1.0");
        
        // Very large negative value should approach -1.0
        assertTrue(tanh.apply(-100.0) < -0.999, "Tanh of large negative should approach -1.0");
    }

    @ParameterizedTest
    @CsvSource({
        "0.0, 1.0", // Derivative at 0 is 1
        "1.0, 0.4199743416", // Derivative at 1.0
        "-1.0, 0.4199743416"  // Derivative at -1.0
    })
    public void testDerivative(double input, double expected) {
        ActivationFunction tanh = Tanh.getInstance();
        assertEquals(expected, tanh.derivative(input), 0.0001,
                     "Tanh.derivative(" + input + ") should return approximately " + expected);
    }
    
    @Test
    public void testDerivativeRelationship() {
        // The derivative of tanh at x should be 1 - tanh²(x)
        ActivationFunction tanh = Tanh.getInstance();
        
        for (double x = -5; x <= 5; x += 0.5) {
            double tanhValue = tanh.apply(x);
            double expectedDerivative = 1 - (tanhValue * tanhValue);
            assertEquals(expectedDerivative, tanh.derivative(x), 0.0001,
                         "Tanh derivative should equal 1 - tanh²(x)");
        }
    }
    
    @Test
    public void testMathTanhComparison() {
        // Verify that our tanh implementation matches Java's Math.tanh
        ActivationFunction tanh = Tanh.getInstance();
        
        for (double x = -5; x <= 5; x += 0.5) {
            assertEquals(Math.tanh(x), tanh.apply(x), 0.0001,
                        "Tanh implementation should match Math.tanh");
        }
    }
    
    @Test
    public void testGetName() {
        ActivationFunction tanh = Tanh.getInstance();
        assertEquals("Tanh", tanh.getName(), "Name should be 'Tanh'");
    }
}