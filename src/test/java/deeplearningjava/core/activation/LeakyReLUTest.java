package deeplearningjava.core.activation;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the LeakyReLU activation function.
 */
public class LeakyReLUTest {

    @Test
    public void testSingleton() {
        // Verify singleton pattern works properly
        LeakyReLU instance1 = LeakyReLU.getInstance();
        LeakyReLU instance2 = LeakyReLU.getInstance();
        
        // Both references should point to the same object
        assertSame(instance1, instance2, "LeakyReLU should use singleton pattern for default instance");
    }
    
    @Test
    public void testCustomAlpha() {
        // Test creating instances with custom alpha values
        double alpha1 = 0.05;
        double alpha2 = 0.1;
        
        LeakyReLU leakyReLU1 = LeakyReLU.withAlpha(alpha1);
        LeakyReLU leakyReLU2 = LeakyReLU.withAlpha(alpha2);
        
        // Verify alpha values were set correctly
        assertEquals(alpha1, leakyReLU1.getAlpha(), 0.0001, "Alpha value should be set to " + alpha1);
        assertEquals(alpha2, leakyReLU2.getAlpha(), 0.0001, "Alpha value should be set to " + alpha2);
        
        // Verify instances are different
        assertNotSame(leakyReLU1, leakyReLU2, "Custom LeakyReLU instances should be different objects");
        
        // Test constructor directly
        LeakyReLU leakyReLU3 = new LeakyReLU(0.2);
        assertEquals(0.2, leakyReLU3.getAlpha(), 0.0001, "Constructor should set alpha value");
    }

    @Test
    public void testApply() {
        // Use alpha = 0.01 for these tests
        ActivationFunction leakyReLU = LeakyReLU.withAlpha(0.01);
        
        // Positive input
        assertEquals(5.0, leakyReLU.apply(5.0), 0.0001, 
                     "LeakyReLU.apply(5.0) should return approximately 5.0");
        
        // Zero
        assertEquals(0.0, leakyReLU.apply(0.0), 0.0001, 
                     "LeakyReLU.apply(0.0) should return approximately 0.0");
        
        // Negative input with alpha 0.01
        assertEquals(-0.05, leakyReLU.apply(-5.0), 0.0001, 
                     "LeakyReLU.apply(-5.0) should return approximately -0.05");
        
        // Large positive
        assertEquals(100.0, leakyReLU.apply(100.0), 0.0001, 
                     "LeakyReLU.apply(100.0) should return approximately 100.0");
        
        // Large negative with alpha 0.01
        assertEquals(-1.0, leakyReLU.apply(-100.0), 0.0001, 
                     "LeakyReLU.apply(-100.0) should return approximately -1.0");
    }
    
    @Test
    public void testApplyWithDifferentAlphas() {
        double[] alphas = {0.01, 0.05, 0.1, 0.2};
        double inputValue = -10.0;
        
        for (double alpha : alphas) {
            LeakyReLU leakyReLU = LeakyReLU.withAlpha(alpha);
            double expected = alpha * inputValue;
            assertEquals(expected, leakyReLU.apply(inputValue), 0.0001,
                        "LeakyReLU with alpha " + alpha + " should scale negative input by alpha");
        }
    }

    @Test
    public void testDerivative() {
        // Use alpha = 0.01 for these tests
        ActivationFunction leakyReLU = LeakyReLU.withAlpha(0.01);
        
        // Positive input: derivative is 1
        assertEquals(1.0, leakyReLU.derivative(5.0), 0.0001,
                     "LeakyReLU.derivative(5.0) should return 1.0");
        
        // Negative input: derivative is alpha
        assertEquals(0.01, leakyReLU.derivative(-5.0), 0.0001,
                     "LeakyReLU.derivative(-5.0) should return 0.01");
        
        // Large positive
        assertEquals(1.0, leakyReLU.derivative(100.0), 0.0001,
                     "LeakyReLU.derivative(100.0) should return 1.0");
        
        // Large negative
        assertEquals(0.01, leakyReLU.derivative(-100.0), 0.0001,
                     "LeakyReLU.derivative(-100.0) should return 0.01");
    }
    
    @Test
    public void testDerivativeWithDifferentAlphas() {
        double[] alphas = {0.01, 0.05, 0.1, 0.2};
        double inputValue = -10.0;
        
        for (double alpha : alphas) {
            LeakyReLU leakyReLU = LeakyReLU.withAlpha(alpha);
            assertEquals(alpha, leakyReLU.derivative(inputValue), 0.0001,
                        "LeakyReLU derivative with alpha " + alpha + " should be alpha for negative input");
            
            // Positive input should always have derivative of 1.0 regardless of alpha
            assertEquals(1.0, leakyReLU.derivative(10.0), 0.0001,
                        "LeakyReLU derivative should be 1.0 for positive input");
        }
    }
    
    @Test
    public void testDerivativeAtZero() {
        // The derivative at x=0 is typically defined as alpha
        double alpha = 0.01;
        LeakyReLU leakyReLU = LeakyReLU.withAlpha(alpha);
        assertEquals(alpha, leakyReLU.derivative(0.0), 0.0001,
                     "LeakyReLU.derivative(0.0) should be defined as alpha");
    }
    
    @Test
    public void testGetName() {
        LeakyReLU leakyReLU1 = LeakyReLU.getInstance();
        assertTrue(leakyReLU1.getName().contains("LeakyReLU"), 
                   "Name should contain 'LeakyReLU'");
        assertTrue(leakyReLU1.getName().contains(String.valueOf(leakyReLU1.getAlpha())), 
                   "Name should contain the alpha value");
        
        // Test with custom alpha
        double customAlpha = 0.15;
        LeakyReLU leakyReLU2 = LeakyReLU.withAlpha(customAlpha);
        assertTrue(leakyReLU2.getName().contains("LeakyReLU"), 
                   "Name should contain 'LeakyReLU'");
        assertTrue(leakyReLU2.getName().contains(String.valueOf(customAlpha)), 
                   "Name should contain the custom alpha value");
    }
}