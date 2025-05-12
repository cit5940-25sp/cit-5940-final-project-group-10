package deeplearningjava.api;

/**
 * Base interface for all neural network types.
 * This is the unified interface for all network implementations in the framework.
 */
public interface BaseNetwork {
    
    /**
     * Sets the learning rate for training.
     * 
     * @param learningRate The learning rate (must be positive)
     * @throws IllegalArgumentException if learningRate is not positive
     */
    void setLearningRate(double learningRate);
    
    /**
     * Gets the current learning rate.
     * 
     * @return The learning rate
     */
    double getLearningRate();
    
    /**
     * Gets the number of layers in the network.
     * 
     * @return The number of layers
     */
    int getLayerCount();
    
    /**
     * Checks if the network has been initialized.
     * 
     * @return true if the network is ready for processing, false otherwise
     */
    boolean isInitialized();
    
    /**
     * Gets the type of the network.
     * 
     * @return The network type
     */
    NetworkType getType();
    
    /**
     * Enum defining the available network types.
     */
    enum NetworkType {
        DENSE,              // Fully connected neural network
        CONVOLUTIONAL,      // Convolutional neural network
        RECURRENT,          // Recurrent neural network (for future implementation)
        AUTOENCODER,        // Autoencoder network (for future implementation)
        TENSOR,             // Tensor-based neural network
        CUSTOM              // Custom network architectures
    }
}