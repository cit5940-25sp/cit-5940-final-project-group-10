package deeplearningjava.network.optimizer;

/**
 * Stochastic Gradient Descent (SGD) optimizer.
 * Simple optimizer that updates parameters proportional to their negative gradients.
 */
public class SGD implements Optimizer {
    
    private double learningRate;
    
    /**
     * Creates an SGD optimizer with the specified learning rate.
     * @param learningRate The learning rate
     */
    public SGD(double learningRate) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        this.learningRate = learningRate;
    }
    
    @Override
    public double updateParameter(double parameter, double gradient) {
        return parameter - learningRate * gradient;
    }
    
    @Override
    public void reset() {
        // SGD has no state to reset
    }
    
    @Override
    public void setLearningRate(double learningRate) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        this.learningRate = learningRate;
    }
    
    @Override
    public double getLearningRate() {
        return learningRate;
    }
}