package deeplearningjava.api;

/**
 * Interface for components that can be trained.
 */
public interface Trainable {
    
    /**
     * Performs one training iteration.
     * @param inputs Input values
     * @param targets Target output values
     * @return The actual outputs after this training iteration
     */
    double[] train(double[] inputs, double[] targets);
    
    /**
     * Performs multiple training iterations.
     * @param inputs Array of input sets
     * @param targets Array of target output sets
     * @param epochs Number of complete passes through the data
     * @return The loss (error) after training
     */
    double trainBatch(double[][] inputs, double[][] targets, int epochs);
}