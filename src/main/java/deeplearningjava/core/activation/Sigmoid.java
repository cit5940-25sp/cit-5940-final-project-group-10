package deeplearningjava.core.activation;

/**
 * Sigmoid activation function.
 * f(x) = 1 / (1 + e^(-x))
 */
public class Sigmoid implements ActivationFunction {
    
    private static final Sigmoid INSTANCE = new Sigmoid();
    
    // Private constructor for singleton pattern
    private Sigmoid() {}
    
    /**
     * Gets the singleton instance of Sigmoid.
     * @return The Sigmoid instance
     */
    public static Sigmoid getInstance() {
        return INSTANCE;
    }
    
    @Override
    public double apply(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    @Override
    public double derivative(double x) {
        double sigmoid = apply(x);
        return sigmoid * (1.0 - sigmoid);
    }
    
    @Override
    public String getName() {
        return "Sigmoid";
    }
}