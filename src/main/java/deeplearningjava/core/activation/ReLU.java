package deeplearningjava.core.activation;

/**
 * Rectified Linear Unit (ReLU) activation function.
 * f(x) = max(0, x)
 */
public class ReLU implements ActivationFunction {
    
    private static final ReLU INSTANCE = new ReLU();
    
    // Private constructor for singleton pattern
    private ReLU() {}
    
    /**
     * Gets the singleton instance of ReLU.
     * @return The ReLU instance
     */
    public static ReLU getInstance() {
        return INSTANCE;
    }
    
    @Override
    public double apply(double x) {
        return Math.max(0.0, x);
    }
    
    @Override
    public double derivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }
    
    @Override
    public String getName() {
        return "ReLU";
    }
}