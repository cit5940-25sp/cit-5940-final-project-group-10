package deeplearningjava.core.activation;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the ActivationFunctions factory class.
 */
public class ActivationFunctionsTest {

    @Test
    public void testGetByName() {
        // Test retrieving functions by name (case insensitive)
        assertNotNull(ActivationFunctions.get("relu"), "Should find ReLU by name");
        assertNotNull(ActivationFunctions.get("ReLU"), "Should find ReLU by name (case insensitive)");
        assertNotNull(ActivationFunctions.get("sigmoid"), "Should find Sigmoid by name");
        assertNotNull(ActivationFunctions.get("tanh"), "Should find Tanh by name");
        assertNotNull(ActivationFunctions.get("linear"), "Should find Linear by name");
        assertNotNull(ActivationFunctions.get("LeakyReLU"), "Should find LeakyReLU by name");
        
        // Test with an invalid name
        assertNull(ActivationFunctions.get("invalid_function"), "Should return null for invalid name");
    }
    
    @Test
    public void testFactoryMethods() {
        // Test each factory method returns the correct function type
        assertTrue(ActivationFunctions.relu() instanceof ReLU, "relu() should return ReLU instance");
        assertTrue(ActivationFunctions.sigmoid() instanceof Sigmoid, "sigmoid() should return Sigmoid instance");
        assertTrue(ActivationFunctions.tanh() instanceof Tanh, "tanh() should return Tanh instance");
        assertTrue(ActivationFunctions.linear() instanceof Linear, "linear() should return Linear instance");
        assertTrue(ActivationFunctions.leakyRelu() instanceof LeakyReLU, "leakyRelu() should return LeakyReLU instance");
    }
    
    @Test
    public void testLeakyReluWithCustomAlpha() {
        // Test LeakyReLU with custom alpha value
        double customAlpha = 0.15;
        ActivationFunction leakyReLU = ActivationFunctions.leakyRelu(customAlpha);
        
        assertTrue(leakyReLU instanceof LeakyReLU, "Should return LeakyReLU instance");
        
        // Test the alpha value was set correctly
        LeakyReLU concreteLeakyReLU = (LeakyReLU) leakyReLU;
        assertEquals(customAlpha, concreteLeakyReLU.getAlpha(), 0.0001, 
                     "Custom alpha value should be set");
        
        // Test that it scales negative inputs by the custom alpha
        assertEquals(-customAlpha, leakyReLU.apply(-1.0), 0.0001, 
                     "Should scale negative input by custom alpha");
    }
    
    @Test
    public void testCustomActivationFunctionRegistration() {
        // Create and register a custom activation function
        ActivationFunction customFunction = new ActivationFunction() {
            @Override
            public double apply(double x) {
                return Math.pow(x, 2); // x^2 function
            }
            
            @Override
            public double derivative(double x) {
                return 2 * x; // Derivative of x^2 is 2x
            }
            
            @Override
            public String getName() {
                return "Square";
            }
        };
        
        // Register the custom function
        ActivationFunctions.registerFunction(customFunction);
        
        // Try to retrieve it
        ActivationFunction retrieved = ActivationFunctions.get("square");
        assertNotNull(retrieved, "Should be able to retrieve registered custom function");
        assertEquals("Square", retrieved.getName(), "Should retrieve function with correct name");
        
        // Test function behavior
        assertEquals(4.0, retrieved.apply(2.0), 0.0001, "Custom function should work correctly");
        assertEquals(4.0, retrieved.derivative(2.0), 0.0001, "Custom function derivative should work");
    }
    
    @Test
    public void testSingletonBehavior() {
        // Factory methods should return singleton instances
        assertSame(ActivationFunctions.relu(), ActivationFunctions.relu(), 
                   "relu() should return the same instance each time");
        assertSame(ActivationFunctions.sigmoid(), ActivationFunctions.sigmoid(), 
                   "sigmoid() should return the same instance each time");
        assertSame(ActivationFunctions.tanh(), ActivationFunctions.tanh(), 
                   "tanh() should return the same instance each time");
        assertSame(ActivationFunctions.linear(), ActivationFunctions.linear(), 
                   "linear() should return the same instance each time");
        assertSame(ActivationFunctions.leakyRelu(), ActivationFunctions.leakyRelu(), 
                   "leakyRelu() should return the same instance each time");
        
        // Also check get() method returns singletons
        assertSame(ActivationFunctions.get("relu"), ActivationFunctions.relu(), 
                   "get(\"relu\") should return same instance as relu()");
    }
    
    @Test
    public void testLeakyReluWithDifferentAlphas() {
        // LeakyReLU with different alphas should be different instances
        ActivationFunction leakyReLU1 = ActivationFunctions.leakyRelu(0.01);
        ActivationFunction leakyReLU2 = ActivationFunctions.leakyRelu(0.02);
        
        assertNotSame(leakyReLU1, leakyReLU2, 
                      "LeakyReLU instances with different alphas should be different");
    }
}