package deeplearningjava.core.activation;

/**
 * Hyperbolic tangent (tanh) activation function.
 * f(x) = tanh(x)
 */
public class Tanh implements ActivationFunction {
    
    private static final Tanh INSTANCE = new Tanh();
    
    // Private constructor for singleton pattern
    private Tanh() {}
    
    /**
     * Gets the singleton instance of Tanh.
     * @return The Tanh instance
     */
    public static Tanh getInstance() {
        return INSTANCE;
    }
    
    @Override
    public double apply(double x) {
        return Math.tanh(x);
    }
    
    @Override
    public double derivative(double x) {
        double tanh = Math.tanh(x);
        return 1.0 - tanh * tanh;
    }
    
    @Override
    public String getName() {
        return "Tanh";
    }
}