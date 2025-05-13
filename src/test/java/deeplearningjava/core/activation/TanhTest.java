package deeplearningjava.core.activation;

import org.junit.jupiter.api.Test;
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

    @Test
    public void testApply() {
        ActivationFunction tanh = Tanh.getInstance();
        
        // Input 0, output 0
        assertEquals(0.0, tanh.apply(0.0), 0.0001, 
                     "Tanh.apply(0.0) should return approximately 0.0");
        
        // Small positive
        assertEquals(0.7615941559, tanh.apply(1.0), 0.0001, 
                     "Tanh.apply(1.0) should return approximately 0.7615941559");
        
        // Small negative
        assertEquals(-0.7615941559, tanh.apply(-1.0), 0.0001, 
                     "Tanh.apply(-1.0) should return approximately -0.7615941559");
        
        // Larger positive
        assertEquals(0.9640275801, tanh.apply(2.0), 0.0001, 
                     "Tanh.apply(2.0) should return approximately 0.9640275801");
        
        // Larger negative
        assertEquals(-0.9640275801, tanh.apply(-2.0), 0.0001, 
                     "Tanh.apply(-2.0) should return approximately -0.9640275801");
    }
    
    @Test
    public void testApplyLargeValues() {
        ActivationFunction tanh = Tanh.getInstance();
        // Very large positive value should approach 1.0
        assertTrue(tanh.apply(100.0) > 0.999, "Tanh of large positive should approach 1.0");
        
        // Very large negative value should approach -1.0
        assertTrue(tanh.apply(-100.0) < -0.999, "Tanh of large negative should approach -1.0");
    }

    @Test
    public void testDerivative() {
        ActivationFunction tanh = Tanh.getInstance();
        
        // Derivative at 0 is 1
        assertEquals(1.0, tanh.derivative(0.0), 0.0001,
                     "Tanh.derivative(0.0) should return approximately 1.0");
        
        // Derivative at 1.0
        assertEquals(0.4199743416, tanh.derivative(1.0), 0.0001,
                     "Tanh.derivative(1.0) should return approximately 0.4199743416");
        
        // Derivative at -1.0
        assertEquals(0.4199743416, tanh.derivative(-1.0), 0.0001,
                     "Tanh.derivative(-1.0) should return approximately 0.4199743416");
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