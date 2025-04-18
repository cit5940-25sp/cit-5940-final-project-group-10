package deeplearningjava;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.DoubleUnaryOperator;
import java.util.Random;

public class Node {
    private double value = 0;
    private double bias;
    private double gradient = 0;
    private double netInput = 0;

    private final List<Edge> outgoingConnections = new ArrayList<>();
    private final List<Edge> incomingConnections = new ArrayList<>();

    private final DoubleUnaryOperator activationFunction;
    private final DoubleUnaryOperator activationDerivative;
    private static final Random random = new Random();

    public Node (DoubleUnaryOperator activationFunction, DoubleUnaryOperator activationDerivative) {
        this.activationFunction = Objects.requireNonNull(activationFunction, "activationFunction must not be null");
        this.activationDerivative = Objects.requireNonNull(activationDerivative, "activationDerivative must not be null");
        this.bias = (random.nextDouble() - 0.5) * 0.2;
    }

    public double calculateNetInput() {
        this.netInput = 0.0;

        if (!incomingConnections.isEmpty()) {
            for (Edge edge : incomingConnections) {
                this.netInput += edge.getSourceNode().getValue() * edge.getWeight();
            }
        }
        this.netInput += this.bias;

        return this.netInput;
    }

    public void applyStandardActivation() {
        this.value = this.activationFunction.applyAsDouble(this.netInput);
    }

    public void calculateStandardOutputGradient(double targetValue) {
        double errorDerivative = this.value - targetValue;
        double activationInputDerivative = this.activationDerivative.applyAsDouble(this.netInput);
        this.gradient = errorDerivative * activationInputDerivative;
    }

    public void calculateHiddenGradient() {
        if (outgoingConnections.isEmpty()) {
            this.gradient = 0;
            return;
        }
        double downstreamWeightedSum = 0;
        for (Edge edge : outgoingConnections) {
            downstreamWeightedSum += edge.getWeight() * edge.getTargetNode().getGradient();
        }
        double activationInputDerivative = this.activationDerivative.applyAsDouble(this.netInput);
        this.gradient = downstreamWeightedSum * activationInputDerivative;
    }

    public void updateBias(double learningRate) {
        this.bias -= learningRate * this.gradient;
    }

    public void updateIncomingWeights(double learningRate) {
        for (Edge edge : incomingConnections) {
            edge.updateWeight(learningRate);
        }
    }

    public double getValue() {
        return this.value;
    }

    public double getBias() {
        return this.bias;
    }

    public double getGradient() {
        return this.gradient;
    }

    public double getNetInput() {
        return this.netInput;
    }

    public List<Edge> getOutgoingConnections() {
        return this.outgoingConnections;
    }

    public List<Edge> getIncomingConnections() {
        return this.incomingConnections;
    }
    public DoubleUnaryOperator getActivationFunction() {
        return this.activationFunction;
    }

    public void setValue(double value) {
        this.value = value;
        this.netInput = value;
    }

    public void setGradient(double gradient) {
        this.gradient = gradient;
    }

    public void setInputLayerValue(double value) {
        this.value = value;
        this.netInput = value;
    }

    private static double sigmoidImpl(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    private static double sigmoidDerivativeImpl(double x) {
        double output = sigmoidImpl(x);
        return output * (1.0 - output);
    }
    public static final DoubleUnaryOperator SIGMOID = Node::sigmoidImpl;
    public static final DoubleUnaryOperator SIGMOID_DERIVATIVE = Node::sigmoidDerivativeImpl;

    private static double tanhImpl(double x) {
        return Math.tanh(x);
    }
    private static double tanhDerivativeImpl(double x) {
        double output = Math.tanh(x);
        return output * (1.0 - output);
    }
    public static final DoubleUnaryOperator TANH = Node::tanhImpl;
    public static final DoubleUnaryOperator TANH_DERIVATIVE = Node::tanhDerivativeImpl;

    private static double reluImpl(double x) {
        return Math.max(0.0, x);
    }
    private static double reluDerivativeImpl(double x) {
        if (x > 0) {
            return 1.0;
        } else {
            return 0.0;
        }
    }
    public static final DoubleUnaryOperator RELU = Node::reluImpl;
    public static final DoubleUnaryOperator RELU_DERIVATIVE = Node::reluDerivativeImpl;

    public static final double LEAKY_RELU_ALPHA = 0.01;
    private static double leakyReluImpl(double x) {
        if (x > 0) {
            return x;
        } else {
            return x * LEAKY_RELU_ALPHA;
        }
    }
    private static double leakyReluDerivativeImpl(double x) {
        if (x > 0) {
            return 1.0;
        } else {
            return LEAKY_RELU_ALPHA;
        }
    }
    public static final DoubleUnaryOperator LEAKY_RELU = Node::leakyReluImpl;
    public static final DoubleUnaryOperator LEAKY_RELU_DERIVATIVE = Node::leakyReluDerivativeImpl;

    private static double linearImpl(double x) {
        return x;
    }
    private static double linearDerivativeImpl(double c) {
        return 1.0;
    }
    public static final DoubleUnaryOperator LINEAR = Node::linearImpl;
    public static final DoubleUnaryOperator LINEAR_DERIVATIVE = Node::linearDerivativeImpl;

    private static final double SQRT_2_OVER_PI = Math.sqrt(2.0 / Math.PI);
    private static final double GELU_C = 0.044715;
    private static double geluImpl(double x) {
        double x_cubed = Math.pow(x, 3);
        double inner = SQRT_2_OVER_PI * (x + GELU_C * x_cubed);
        return 0.5 * x * (1.0 + Math.tanh(inner));
    }
    private static double geluDerivativeImpl(double x) {
        double x_squared = x * x;
        double x_cubed = x_squared * x;
        double inner = SQRT_2_OVER_PI * (x + GELU_C * x_cubed);
        double tanh_inner = Math.tanh(inner);
        double sech_squared_inner = 1.0 - tanh_inner * tanh_inner;
        double inner_derivative = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_C * x_squared);
        return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_squared_inner * inner_derivative;
    }
    public static final DoubleUnaryOperator GELU = Node::geluImpl;
    public static final DoubleUnaryOperator GELU_DERIVATIVE = Node::geluDerivativeImpl;
}