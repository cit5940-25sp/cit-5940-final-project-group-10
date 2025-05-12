package deeplearningjava.core.activation;

/**
 * Leaky Rectified Linear Unit (Leaky ReLU) activation function.
 * f(x) = x if x > 0, alpha * x otherwise
 */
public class LeakyReLU implements ActivationFunction {
    
    private static final LeakyReLU INSTANCE = new LeakyReLU(0.01);
    private final double alpha;
    
    /**
     * Creates a LeakyReLU with the specified alpha value.
     * @param alpha Slope for negative inputs
     */
    public LeakyReLU(double alpha) {
        this.alpha = alpha;
    }
    
    /**
     * Gets the singleton instance of LeakyReLU with default alpha.
     * @return The default LeakyReLU instance
     */
    public static LeakyReLU getInstance() {
        return INSTANCE;
    }
    
    /**
     * Creates a new LeakyReLU with custom alpha.
     * @param alpha The alpha value for negative input slope
     * @return A new LeakyReLU instance
     */
    public static LeakyReLU withAlpha(double alpha) {
        return new LeakyReLU(alpha);
    }
    
    @Override
    public double apply(double x) {
        return x > 0 ? x : alpha * x;
    }
    
    @Override
    public double derivative(double x) {
        return x > 0 ? 1.0 : alpha;
    }
    
    @Override
    public String getName() {
        return "LeakyReLU(α=" + alpha + ")";
    }
    
    /**
     * Gets the alpha value used by this LeakyReLU.
     * @return The alpha value
     */
    public double getAlpha() {
        return alpha;
    }
}