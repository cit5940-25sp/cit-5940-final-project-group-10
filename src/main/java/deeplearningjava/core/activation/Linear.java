package deeplearningjava.core.activation;

/**
 * Linear activation function (identity function).
 * f(x) = x
 */
public class Linear implements ActivationFunction {
    
    private static final Linear INSTANCE = new Linear();
    
    // Private constructor for singleton pattern
    private Linear() {}
    
    /**
     * Gets the singleton instance of Linear.
     * @return The Linear instance
     */
    public static Linear getInstance() {
        return INSTANCE;
    }
    
    @Override
    public double apply(double x) {
        return x;
    }
    
    @Override
    public double derivative(double x) {
        return 1.0;
    }
    
    @Override
    public String getName() {
        return "Linear";
    }
}