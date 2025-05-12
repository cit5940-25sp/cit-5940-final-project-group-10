package deeplearningjava.network.optimizer;

/**
 * Interface for neural network optimizers.
 * Optimizers are used to update network parameters during training.
 */
public interface Optimizer {
    
    /**
     * Updates a parameter based on its gradient.
     * @param parameter The current parameter value
     * @param gradient The gradient of the parameter
     * @return The updated parameter value
     */
    double updateParameter(double parameter, double gradient);
    
    /**
     * Resets any accumulated state in the optimizer.
     */
    void reset();
    
    /**
     * Sets the learning rate for the optimizer.
     * @param learningRate The learning rate
     */
    void setLearningRate(double learningRate);
    
    /**
     * Gets the current learning rate.
     * @return The learning rate
     */
    double getLearningRate();
}