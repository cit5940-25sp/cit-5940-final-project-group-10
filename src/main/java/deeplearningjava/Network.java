package deeplearningjava;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Collections;
import java.util.function.DoubleUnaryOperator;
import java.util.Arrays;

public class Network {

    private final List<Layer> layers;
    private double learningRate = 0.1;

    public Network() {
        this.layers = new ArrayList<>();
    }

    public Network (int[] layerSizes,
                    DoubleUnaryOperator hiddenActivation,
                    DoubleUnaryOperator hiddenActivationDerivative,
                    DoubleUnaryOperator outputActivation,
                    DoubleUnaryOperator outputActivationDerivative,
                    boolean useSoftmaxOutput) {
        Objects.requireNonNull(layerSizes, "layerSizes must not be null");
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("Network must have at least an " +
                    "input and output layer (layerSizes length >= 2).");
        }
        Objects.requireNonNull(hiddenActivation, "hiddenActivation must not be null");
        Objects.requireNonNull(hiddenActivationDerivative, "hiddenActivationDerivative must not be null");
        if (!useSoftmaxOutput) {
            Objects.requireNonNull(outputActivation, "outputActivation must not be null");
            Objects.requireNonNull(outputActivationDerivative, "outputActivationDerivative must not be null");
        }
        for(int i = 0; i < layerSizes.length - 1; i++) {
            if (layerSizes[i] <= 0) {
                throw new IllegalArgumentException("Layer " + (i + 1) + "size must be positive");
            }
        }

        this.layers = new ArrayList<>(layerSizes.length);

        Layer inputLayer = new Layer(layerSizes[0], Node.LINEAR, Node.LINEAR_DERIVATIVE);
        this.layers.add(inputLayer);
        System.out.printf("Created Input Layer (%d nodes)\n", inputLayer.getSize());

        for (int i = 1; i < layerSizes.length - 1; i++) {
            Layer hiddenLayer = new Layer(layerSizes[i], hiddenActivation, hiddenActivationDerivative);
            this.layers.add(hiddenLayer);

            Layer previousLayer = this.layers.get(i - 1);
            previousLayer.connectTo(hiddenLayer);
            previousLayer.initializeWeights(hiddenLayer.getSize());
            System.out.printf("Created Hidden Layer %d (size %d), connected Layer %d to it and initialized weights.\n",
                    i, hiddenLayer.getSize(), i - 1);
        }

        Layer outputLayer = new Layer(
                layerSizes[layerSizes.length - 1],
                outputActivation,
                outputActivationDerivative,
                useSoftmaxOutput
        );
        this.layers.add(outputLayer);

        Layer lastHiddenOrInputLayer = this.layers.get(layers.size() - 2);
        lastHiddenOrInputLayer.connectTo(outputLayer);
        lastHiddenOrInputLayer.initializeWeights(outputLayer.getSize());
        System.out.printf("Created Output Layer %d (size %d, Softmax=%b), connected Layer %d to it and initialized weights.\n",
                layers.size() - 1, outputLayer.getSize(), useSoftmaxOutput, layers.size() - 2);
    }

    public void setLearningRate(double learningRate) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        this.learningRate = learningRate;
    }

    public void addLayer(Layer layer) {
        Objects.requireNonNull(layer, "layer must not be null");

        if (!this.layers.isEmpty()) {
            Layer previousLayer = layers.get(layers.size() - 1);
            previousLayer.connectTo(layer);
            previousLayer.initializeWeights(layer.getSize());
            System.out.printf("Connected Layer %d (size %d) to Layer %d (size %d) and initialized weights.\n",
                    layers.size() -1, previousLayer.getSize(), layers.size(), layer.getSize());
        } else {
            System.out.printf("Added Input Layer %d (size %d).\n", layers.size(), layer.getSize());
        }

        this.layers.add(layer);
    }

    public double[] feedForward(double[] inputs) {
        if (layers.isEmpty()) {
            throw new IllegalStateException("Network cannot be empty.");
        }

        if (inputs.length != layers.get(0).getSize()) {
            throw new IllegalArgumentException(String.format(
                    "Input size (%d) must match input layer size (%d).",
                    inputs.length, layers.get(0).getSize()));
        }

        layers.get(0).setOutputs(inputs);

        for (int i = 1; i < layers.size(); i++) {
            layers.get(i).calculateOutputs();
        }

        Layer outputLayer = layers.get(layers.size() - 1);
        List<Node> outputNodes = outputLayer.getNodes();
        double[] networkOutput = new double[outputLayer.getSize()];
        for (int i = 0; i < outputNodes.size(); i++) {
            networkOutput[i] = outputNodes.get(i).getValue();
        }
        return networkOutput;
    }

    public void backPropagate(double[] targetOutputs) {
        if (layers.isEmpty()) {
            throw new IllegalStateException("Network cannot be empty.");
        }
        if (layers.size() < 2) {
            System.out.println("Warning: Backpropagation requires at least an input and output layer.");
            return;
        }

        Layer outputLayer = layers.get(layers.size() - 1);
        if (targetOutputs.length != outputLayer.getSize()) {
            throw new IllegalArgumentException(String.format(
                    "Target output size (%d) must match output layer size (%d).",
                    targetOutputs.length, outputLayer.getSize()));
        }

        outputLayer.calculateOutputGradients(targetOutputs);

        for (int i = layers.size() - 2; i > 0; i--) {
            layers.get(i).calculateHiddenGradients();
        }

        for (int i = layers.size() - 1; i > 0; i--) {
            layers.get(i).updateParameters(this.learningRate);
        }
    }

    public double[] trainingIteration(double[] inputs, double[] targetOutputs) {
        double[] output = feedForward(inputs);
        backPropagate(targetOutputs);
        return output;
    }

    public List<Layer> getLayers() {
        return Collections.unmodifiableList(this.layers);
    }
}