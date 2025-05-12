package deeplearningjava.core.activation;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the activation function implementations.
 */
public class ActivationFunctionTest {

    @Test
    public void testSigmoidActivation() {
        ActivationFunction sigmoid = ActivationFunctions.sigmoid();
        
        // Test basic sigmoid properties
        assertEquals(0.5, sigmoid.apply(0.0), 0.0001);
        
        // Large positive values should be very close to 1
        double largePositive = sigmoid.apply(100.0);
        assertTrue(largePositive >= 0.99, "Sigmoid of large positive should be at least 0.99");
        
        // Large negative values should be very close to 0
        double largeNegative = sigmoid.apply(-100.0);
        assertTrue(largeNegative <= 0.01, "Sigmoid of large negative should be at most 0.01");
        
        // Test derivative
        double x = 0.0;
        double activationValue = sigmoid.apply(x);
        double expectedDerivative = activationValue * (1 - activationValue);
        assertEquals(expectedDerivative, sigmoid.derivative(x), 0.0001);
    }
    
    @Test
    public void testTanhActivation() {
        ActivationFunction tanh = ActivationFunctions.tanh();
        
        // Test basic tanh properties
        assertEquals(0.0, tanh.apply(0.0), 0.0001);
        
        // Large positive values should be very close to 1
        double largePositive = tanh.apply(100.0);
        assertTrue(largePositive >= 0.99, "Tanh of large positive should be at least 0.99");
        
        // Large negative values should be very close to -1
        double largeNegative = tanh.apply(-100.0);
        assertTrue(largeNegative <= -0.99, "Tanh of large negative should be at most -0.99");
        
        // Test derivative
        double x = 0.0;
        double activationValue = tanh.apply(x);
        double expectedDerivative = 1 - (activationValue * activationValue);
        assertEquals(expectedDerivative, tanh.derivative(x), 0.0001);
    }
    
    @Test
    public void testReLUActivation() {
        ActivationFunction relu = ActivationFunctions.relu();
        
        // Test basic ReLU properties
        assertEquals(0.0, relu.apply(0.0), 0.0001);
        assertEquals(5.0, relu.apply(5.0), 0.0001);
        assertEquals(0.0, relu.apply(-5.0), 0.0001);
        
        // Test derivative
        assertEquals(0.0, relu.derivative(-1.0), 0.0001);
        assertEquals(1.0, relu.derivative(1.0), 0.0001);
        
        // Test x = 0 case (undefined derivative, but we choose 0)
        assertEquals(0.0, relu.derivative(0.0), 0.0001);
    }
    
    @Test
    public void testLeakyReLUActivation() {
        double alpha = 0.01;
        ActivationFunction leakyRelu = ActivationFunctions.leakyRelu(alpha);
        
        // Test basic Leaky ReLU properties
        assertEquals(0.0, leakyRelu.apply(0.0), 0.0001);
        assertEquals(5.0, leakyRelu.apply(5.0), 0.0001);
        assertEquals(-5.0 * alpha, leakyRelu.apply(-5.0), 0.0001);
        
        // Test derivative
        assertEquals(alpha, leakyRelu.derivative(-1.0), 0.0001);
        assertEquals(1.0, leakyRelu.derivative(1.0), 0.0001);
        
        // Test x = 0 case (undefined derivative, but we choose alpha)
        assertEquals(alpha, leakyRelu.derivative(0.0), 0.0001);
        
        // Test with different alpha
        alpha = 0.2;
        leakyRelu = ActivationFunctions.leakyRelu(alpha);
        assertEquals(-5.0 * alpha, leakyRelu.apply(-5.0), 0.0001);
        assertEquals(alpha, leakyRelu.derivative(-1.0), 0.0001);
    }
    
    @Test
    public void testLinearActivation() {
        ActivationFunction linear = ActivationFunctions.linear();
        
        // Test basic linear properties
        assertEquals(0.0, linear.apply(0.0), 0.0001);
        assertEquals(5.0, linear.apply(5.0), 0.0001);
        assertEquals(-5.0, linear.apply(-5.0), 0.0001);
        
        // Test derivative (should always be 1.0)
        assertEquals(1.0, linear.derivative(-1.0), 0.0001);
        assertEquals(1.0, linear.derivative(0.0), 0.0001);
        assertEquals(1.0, linear.derivative(1.0), 0.0001);
    }
    
    @Test
    public void testCustomActivation() {
        // Create a custom activation function (e.g., sine function)
        ActivationFunction sine = new ActivationFunction() {
            @Override
            public double apply(double input) {
                return Math.sin(input);
            }
            
            @Override
            public double derivative(double input) {
                return Math.cos(input);
            }
            
            @Override
            public String getName() {
                return "Sine";
            }
        };
        
        // Test basic properties
        assertEquals(0.0, sine.apply(0.0), 0.0001);
        assertEquals(1.0, sine.apply(Math.PI / 2), 0.0001);
        assertEquals(0.0, sine.apply(Math.PI), 0.0001);
        assertEquals(-1.0, sine.apply(3 * Math.PI / 2), 0.0001);
        
        // Test derivative
        assertEquals(1.0, sine.derivative(0.0), 0.0001);
        assertEquals(0.0, sine.derivative(Math.PI / 2), 0.0001);
        assertEquals(-1.0, sine.derivative(Math.PI), 0.0001);
    }
}