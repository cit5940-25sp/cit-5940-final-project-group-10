package deeplearningjava;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.DoubleUnaryOperator;

public class Layer {

    private final List<Node> nodes;
    private final int size;
    private final boolean isSoftmaxLayer;

    public Layer(int size, DoubleUnaryOperator activationFunction, DoubleUnaryOperator activationDerivative) {
        if (size <= 0) {
            throw new IllegalArgumentException("size must be positive");
        }
        this.size = size;
        this.nodes = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            this.nodes.add(new Node(activationFunction, activationDerivative));
        }
        this.isSoftmaxLayer = false;
    }

    public Layer(int size, DoubleUnaryOperator activationFunction,
                 DoubleUnaryOperator activationDerivative, boolean isSoftmaxLayer) {
        if (size <= 0) {
            throw new IllegalArgumentException("size must be positive");
        }
        this.size = size;
        this.isSoftmaxLayer = isSoftmaxLayer;
        this.nodes = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            if (isSoftmaxLayer) {
                this.nodes.add(new Node(Node.LINEAR, Node.LINEAR_DERIVATIVE));
            } else {
                this.nodes.add(new Node(activationFunction, activationDerivative));
            }
        }
    }

    public List<Node> getNodes() {
        return this.nodes;
    }

    public int getSize() {
        return this.size;
    }

    public void connectTo(Layer nextLayer) {
        Objects.requireNonNull(nextLayer, "nextLayer must not be null");
        for (Node sourceNode : this.nodes) {
            for (Node targetNode : nextLayer.nodes) {
                Edge edge = new Edge(sourceNode, targetNode);

                sourceNode.getOutgoingConnections().add(edge);
                targetNode.getIncomingConnections().add(edge);
            }
        }
    }

    public void initializeWeights(int nextLayerSize) {
        int fan_in = this.getSize();
        int fan_out = nextLayerSize;

        for(Node node : this.nodes) {
            for (Edge edge : node.getOutgoingConnections()) {
                edge.initializeWeight(fan_in, fan_out);
            }
        }
    }

    public void calculateOutputs() {
        double[] netInputs = new double[this.size];
        for (int i = 0; i < this.size; i++) {
            netInputs[i] = nodes.get(i).calculateNetInput();
        }

        if (isSoftmaxLayer) {
            applySoftMaxActivation(netInputs);
        } else {
            for (Node node : this.nodes) {
                node.applyStandardActivation();
            }
        }
    }

    private void applySoftMaxActivation(double[] netInputs) {
        double maxNetInput = Double.NEGATIVE_INFINITY;
        for (double netInput : netInputs) {
            if (netInput > maxNetInput) {
                maxNetInput = netInput;
            }
        }

        double[] exps = new double[this.size];
        for (int i = 0; i < this.size; i++) {
            exps[i] = Math.exp(netInputs[i] - maxNetInput);
        }

        double sumExps = 0.0;
        for (double expVal : exps) {
            sumExps += expVal;
        }

        if (sumExps < 1e-15) {
            double uniformProb = 1.0 / size;
            for (Node node : this.nodes) {
                node.setValue(uniformProb);
            }
            System.out.println("Warning: sum of exp values is too small, " +
                    "softmax layer will be set to uniform probability.");
        } else {
            for (int i = 0; i < this.size; i++) {
                this.nodes.get(i).setValue(exps[i] / sumExps);
            }
            System.out.println("Softmax layer activated.");
        }
    }

    public void setOutputs(double[] inputs) {
        if (inputs.length != this.getSize()) {
            throw new IllegalArgumentException("Number of inputs (" + inputs.length
                    + ") must match layer size (" + this.size + ")");
        }
        if (isSoftmaxLayer) {

        }
        for (int i = 0; i < this.size; i++) {
            nodes.get(i).setValue(inputs[i]);
        }
    }

    public void calculateOutputGradients(double[] targetOutputs) {
        if (targetOutputs.length != this.getSize()) {
            throw new IllegalArgumentException("Number of targets (" + targetOutputs.length
                    + ") must match layer size (" + this.size + ")");
        }
        if (isSoftmaxLayer) {
            for (int i = 0; i < this.size; i++) {
                this.nodes.get(i).setGradient(this.nodes.get(i).getValue() - targetOutputs[i]);
            }
        } else {
            for (int i = 0 ; i < this.size; i++) {
                this.nodes.get(i).calculateStandardOutputGradient(targetOutputs[i]);
            }
        }

    }

    public void calculateHiddenGradients() {
        for (Node node : this.nodes) {
            node.calculateHiddenGradient();
        }
    }

    public void updateWeights(double learningRate) {
        for (Node node : this.nodes) {
            node.calculateHiddenGradient();
        }
    }

    public void updateParameters(double learningRate) {
        for (Node node : this.nodes) {
            node.updateBias(learningRate);
            node.updateIncomingWeights(learningRate);
        }
    }
}
