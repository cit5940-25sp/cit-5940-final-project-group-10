package deeplearningjava.core.activation;

import org.junit.jupiter.api.Test;
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

    @Test
    public void testApply() {
        ActivationFunction sigmoid = Sigmoid.getInstance();
        
        // Input 0, output 0.5
        assertEquals(0.5, sigmoid.apply(0.0), 0.0001, 
                     "Sigmoid.apply(0.0) should return approximately 0.5");
        
        // Small positive
        assertEquals(0.7310585786, sigmoid.apply(1.0), 0.0001, 
                     "Sigmoid.apply(1.0) should return approximately 0.7310585786");
        
        // Small negative
        assertEquals(0.2689414214, sigmoid.apply(-1.0), 0.0001, 
                     "Sigmoid.apply(-1.0) should return approximately 0.2689414214");
        
        // Larger positive
        assertEquals(0.8807970779, sigmoid.apply(2.0), 0.0001, 
                     "Sigmoid.apply(2.0) should return approximately 0.8807970779");
        
        // Larger negative
        assertEquals(0.1192029221, sigmoid.apply(-2.0), 0.0001, 
                     "Sigmoid.apply(-2.0) should return approximately 0.1192029221");
    }
    
    @Test
    public void testApplyLargeValues() {
        ActivationFunction sigmoid = Sigmoid.getInstance();
        // Very large positive value should approach 1.0
        assertTrue(sigmoid.apply(100.0) > 0.999, "Sigmoid of large positive should approach 1.0");
        
        // Very large negative value should approach 0.0
        assertTrue(sigmoid.apply(-100.0) < 0.001, "Sigmoid of large negative should approach 0.0");
    }

    @Test
    public void testDerivative() {
        ActivationFunction sigmoid = Sigmoid.getInstance();
        
        // Derivative at 0 is 0.25
        assertEquals(0.25, sigmoid.derivative(0.0), 0.0001,
                     "Sigmoid.derivative(0.0) should return approximately 0.25");
        
        // Derivative at 2.0
        assertEquals(0.1049935854, sigmoid.derivative(2.0), 0.0001,
                     "Sigmoid.derivative(2.0) should return approximately 0.1049935854");
        
        // Derivative at -2.0
        assertEquals(0.1049935854, sigmoid.derivative(-2.0), 0.0001,
                     "Sigmoid.derivative(-2.0) should return approximately 0.1049935854");
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