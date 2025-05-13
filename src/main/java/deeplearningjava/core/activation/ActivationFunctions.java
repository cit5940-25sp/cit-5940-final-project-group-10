package deeplearningjava.core.activation;

import java.util.HashMap;
import java.util.Map;

/**
 * Factory class for obtaining activation function instances.
 */
public class ActivationFunctions {
    
    private static final Map<String, ActivationFunction> FUNCTIONS = new HashMap<>();
    
    static {
        // Register standard activation functions
        registerFunction(ReLU.getInstance());
        registerFunction(Sigmoid.getInstance());
        registerFunction(Tanh.getInstance());
        registerFunction(Linear.getInstance());
        
        // Register LeakyReLU with both the full name and the simple name for backward compatibility
        LeakyReLU leakyReLU = LeakyReLU.getInstance();
        registerFunction(leakyReLU);
        FUNCTIONS.put("leakyrelu", leakyReLU); // Register with lowercase simplified name
    }
    
    private ActivationFunctions() {
        // Prevent instantiation
    }
    
    /**
     * Registers an activation function in the registry.
     * @param function The activation function to register
     */
    public static void registerFunction(ActivationFunction function) {
        FUNCTIONS.put(function.getName().toLowerCase(), function);
    }
    
    /**
     * Gets an activation function by name.
     * @param name The name of the activation function
     * @return The activation function, or null if not found
     */
    public static ActivationFunction get(String name) {
        return FUNCTIONS.get(name.toLowerCase());
    }
    
    /**
     * Gets the ReLU activation function.
     * @return The ReLU function
     */
    public static ActivationFunction relu() {
        return ReLU.getInstance();
    }
    
    /**
     * Gets the Sigmoid activation function.
     * @return The Sigmoid function
     */
    public static ActivationFunction sigmoid() {
        return Sigmoid.getInstance();
    }
    
    /**
     * Gets the Tanh activation function.
     * @return The Tanh function
     */
    public static ActivationFunction tanh() {
        return Tanh.getInstance();
    }
    
    /**
     * Gets the Linear activation function.
     * @return The Linear function
     */
    public static ActivationFunction linear() {
        return Linear.getInstance();
    }
    
    /**
     * Gets the LeakyReLU activation function with default alpha.
     * @return The LeakyReLU function
     */
    public static ActivationFunction leakyRelu() {
        return LeakyReLU.getInstance();
    }
    
    /**
     * Gets a LeakyReLU activation function with custom alpha.
     * @param alpha The alpha value for negative slope
     * @return A LeakyReLU function with the specified alpha
     */
    public static ActivationFunction leakyRelu(double alpha) {
        return LeakyReLU.withAlpha(alpha);
    }
}