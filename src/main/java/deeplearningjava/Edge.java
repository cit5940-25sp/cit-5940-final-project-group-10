package deeplearningjava;

import java.util.Objects;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

public class Edge {
    private final Node sourceNode;
    private final Node targetNode;
    private double weight;

    private static final Random random = new Random();

    public Edge(Node sourceNode, Node targetNode) {
        this.sourceNode = Objects.requireNonNull(sourceNode, "sourceNode must not be null");
        this.targetNode = Objects.requireNonNull(targetNode, "targetNode must not be null");
        
        this.weight = 0;
    }


    public void initializeWeight(int fan_in, int fan_out) {
        DoubleUnaryOperator targetActivation = targetNode.getActivationFunction();

        fan_in = Math.max(1, fan_in);
        fan_out = Math.max(1, fan_out);

        if (targetActivation == Node.RELU
                || targetActivation == Node.LEAKY_RELU
                || targetActivation == Node.GELU) {
            double stddev = Math.sqrt(2.0/fan_in);
            this.weight = random.nextGaussian() * stddev;
        } else if (targetActivation == Node.SIGMOID || targetActivation == Node.TANH) {
            double limit = Math.sqrt(6.0 / (fan_in + fan_out));
            this.weight = (random.nextDouble() * 2.0 - 1.0) * limit;
        } else {
            double limit = 1.0 / Math.sqrt(fan_in);
            this.weight = (random.nextDouble() * 2.0 - 1.0) * limit;
        }
    }

    public void updateWeight(double learningRate) {
        double gradientWrtWright = targetNode.getGradient() * sourceNode.getValue();
        this.weight -= learningRate * gradientWrtWright;
    }

    public Node getSourceNode() {
        return this.sourceNode;
    }

    public Node getTargetNode() {
        return this.targetNode;
    }

    public double getWeight() {
        return this.weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }
}