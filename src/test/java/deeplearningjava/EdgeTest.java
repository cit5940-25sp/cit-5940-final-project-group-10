package deeplearningjava;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class EdgeTest {

    @Test
    public void testEdgeConstructor() {
        Node source = new Node(Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        Node target = new Node(Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        
        Edge edge = new Edge(source, target);
        
        assertNotNull(edge);
        assertSame(source, edge.getSourceNode());
        assertSame(target, edge.getTargetNode());
        assertEquals(0.0, edge.getWeight(), 1e-10);
    }

    @Test
    public void testConstructorWithNullSourceThrowsException() {
        Node target = new Node(Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        
        Exception exception = assertThrows(NullPointerException.class, () -> {
            new Edge(null, target);
        });
        
        assertTrue(exception.getMessage().contains("sourceNode must not be null"));
    }

    @Test
    public void testConstructorWithNullTargetThrowsException() {
        Node source = new Node(Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        
        Exception exception = assertThrows(NullPointerException.class, () -> {
            new Edge(source, null);
        });
        
        assertTrue(exception.getMessage().contains("targetNode must not be null"));
    }

    @Test
    public void testInitializeWeightForSigmoid() {
        Node source = new Node(Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        Node target = new Node(Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        
        Edge edge = new Edge(source, target);
        
        int fanIn = 10;
        int fanOut = 5;
        
        edge.initializeWeight(fanIn, fanOut);
        
        // The weight should be initialized based on Xavier/Glorot initialization
        // for SIGMOID activation functions
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        
        // We can't determine exact value due to randomness, but should be within range
        assertTrue(Math.abs(edge.getWeight()) <= limit);
    }

    @Test
    public void testInitializeWeightForReLU() {
        Node source = new Node(Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        Node target = new Node(Node.RELU, Node.RELU_DERIVATIVE);
        
        Edge edge = new Edge(source, target);
        
        int fanIn = 10;
        int fanOut = 5;
        
        edge.initializeWeight(fanIn, fanOut);
        
        // Weight should be initialized with He initialization for ReLU
        // We can't assert specific value due to randomness
        assertNotEquals(0.0, edge.getWeight());
    }

    @Test
    public void testSetWeight() {
        Node source = new Node(Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        Node target = new Node(Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        
        Edge edge = new Edge(source, target);
        edge.setWeight(0.5);
        
        assertEquals(0.5, edge.getWeight(), 1e-10);
    }

    @Test
    public void testUpdateWeight() {
        Node source = new Node(Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        Node target = new Node(Node.SIGMOID, Node.SIGMOID_DERIVATIVE);
        
        Edge edge = new Edge(source, target);
        edge.setWeight(0.5);
        
        // Set up the values needed for update
        source.setValue(1.0);
        target.setGradient(0.2);
        
        double learningRate = 0.1;
        edge.updateWeight(learningRate);
        
        // New weight = old weight - learning_rate * gradient * input
        // = 0.5 - 0.1 * 0.2 * 1.0 = 0.5 - 0.02 = 0.48
        assertEquals(0.48, edge.getWeight(), 1e-10);
    }
}