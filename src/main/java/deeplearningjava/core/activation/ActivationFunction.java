package deeplearningjava.core.activation;

/**
 * Interface for neural network activation functions.
 * Each activation function should provide both the function itself
 * and its derivative for use in backpropagation.
 */
public interface ActivationFunction {
    
    /**
     * Applies the activation function to the input value.
     * @param x The input value
     * @return The activated value
     */
    double apply(double x);
    
    /**
     * Calculates the derivative of the activation function at the given input.
     * @param x The input value
     * @return The derivative at that point
     */
    double derivative(double x);
    
    /**
     * Gets the name of this activation function.
     * @return The activation function name
     */
    String getName();
}