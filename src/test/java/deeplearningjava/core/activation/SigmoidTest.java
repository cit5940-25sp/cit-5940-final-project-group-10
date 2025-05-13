package deeplearningjava.core.activation;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the Sigmoid activation function.
 */
public class SigmoidTest {

    @Test
    public void testSingleton() {
        // Verify singleton pattern works properly
        Sigmoid instance1 = Sigmoid.getInstance();
        Sigmoid instance2 = Sigmoid.getInstance();
        
        // Both references should point to the same object
        assertSame(instance1, instance2, "Sigmoid should use singleton pattern");
    }

    @ParameterizedTest
    @CsvSource({
        "0.0, 0.5", // Input 0, output 0.5
        "1.0, 0.7310585786", // Small positive
        "-1.0, 0.2689414214", // Small negative
        "2.0, 0.8807970779", // Larger positive
        "-2.0, 0.1192029221"  // Larger negative
    })
    public void testApply(double input, double expected) {
        ActivationFunction sigmoid = Sigmoid.getInstance();
        assertEquals(expected, sigmoid.apply(input), 0.0001, 
                     "Sigmoid.apply(" + input + ") should return approximately " + expected);
    }
    
    @Test
    public void testApplyLargeValues() {
        ActivationFunction sigmoid = Sigmoid.getInstance();
        // Very large positive value should approach 1.0
        assertTrue(sigmoid.apply(100.0) > 0.999, "Sigmoid of large positive should approach 1.0");
        
        // Very large negative value should approach 0.0
        assertTrue(sigmoid.apply(-100.0) < 0.001, "Sigmoid of large negative should approach 0.0");
    }

    @ParameterizedTest
    @CsvSource({
        "0.0, 0.25", // Derivative at 0 is 0.25
        "2.0, 0.1049935854", // Derivative at 2.0
        "-2.0, 0.1049935854"  // Derivative at -2.0
    })
    public void testDerivative(double input, double expected) {
        ActivationFunction sigmoid = Sigmoid.getInstance();
        assertEquals(expected, sigmoid.derivative(input), 0.0001,
                     "Sigmoid.derivative(" + input + ") should return approximately " + expected);
    }
    
    @Test
    public void testDerivativeRelationship() {
        // The derivative of sigmoid at x should be sigmoid(x) * (1 - sigmoid(x))
        ActivationFunction sigmoid = Sigmoid.getInstance();
        
        for (double x = -5; x <= 5; x += 0.5) {
            double sigValue = sigmoid.apply(x);
            double expectedDerivative = sigValue * (1 - sigValue);
            assertEquals(expectedDerivative, sigmoid.derivative(x), 0.0001,
                         "Sigmoid derivative should equal sigmoid(x) * (1 - sigmoid(x))");
        }
    }
    
    @Test
    public void testGetName() {
        ActivationFunction sigmoid = Sigmoid.getInstance();
        assertEquals("Sigmoid", sigmoid.getName(), "Name should be 'Sigmoid'");
    }
}