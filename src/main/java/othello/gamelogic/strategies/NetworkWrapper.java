package othello.gamelogic.strategies;

import deeplearningjava.api.Network;

/**
 * Wrapper class to provide the old API methods for networks in Othello.
 * This allows the Othello game to use the new API without changing its code.
 */
public class NetworkWrapper {
    private final Network network;
    
    /**
     * Creates a new network wrapper.
     * 
     * @param network The new API network to wrap
     */
    public NetworkWrapper(Network network) {
        this.network = network;
    }
    
    /**
     * Gets the learning rate of the network.
     * 
     * @return The learning rate
     */
    public double getLearningRate() {
        return network.getLearningRate();
    }
    
    /**
     * Sets the learning rate of the network.
     * 
     * @param learningRate The new learning rate
     */
    public void setLearningRate(double learningRate) {
        network.setLearningRate(learningRate);
    }
    
    /**
     * Performs a forward pass through the network.
     * 
     * @param inputs The input values
     * @return The output values
     */
    public double[] feedForward(double[] inputs) {
        return network.forward(inputs);
    }
    
    /**
     * Performs a training iteration with backpropagation.
     * 
     * @param inputs The input values
     * @param targetOutputs The target output values
     * @return The actual outputs
     */
    public double[] trainingIteration(double[] inputs, double[] targetOutputs) {
        return network.train(inputs, targetOutputs);
    }
    
    /**
     * Gets the underlying new API network.
     * 
     * @return The wrapped network
     */
    public Network getNetwork() {
        return network;
    }
}