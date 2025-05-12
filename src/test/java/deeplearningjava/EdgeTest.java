package deeplearningjava;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import deeplearningjava.core.Edge;
import deeplearningjava.core.Node;
import deeplearningjava.core.activation.ActivationFunctions;

public class EdgeTest {

    @Test
    public void testEdgeConstructor() {
        Node source = new Node(ActivationFunctions.sigmoid());
        Node target = new Node(ActivationFunctions.sigmoid());
        
        Edge edge = new Edge(source, target);
        
        assertNotNull(edge);
        assertSame(source, edge.getSourceNode());
        assertSame(target, edge.getTargetNode());
        assertNotEquals(0.0, edge.getWeight()); // Weight is randomly initialized
    }

    @Test
    public void testConstructorWithNullSourceThrowsException() {
        Node target = new Node(ActivationFunctions.sigmoid());
        
        Exception exception = assertThrows(NullPointerException.class, () -> {
            new Edge(null, target);
        });
        
        assertNotNull(exception.getMessage());
    }

    @Test
    public void testConstructorWithNullTargetThrowsException() {
        Node source = new Node(ActivationFunctions.sigmoid());
        
        Exception exception = assertThrows(NullPointerException.class, () -> {
            new Edge(source, null);
        });
        
        assertNotNull(exception.getMessage());
    }

    @Test
    public void testInitializeWeight() {
        Node source = new Node(ActivationFunctions.sigmoid());
        Node target = new Node(ActivationFunctions.sigmoid());
        
        Edge edge = new Edge(source, target);
        
        int fanIn = 10;
        int fanOut = 5;
        
        edge.initializeWeight(fanIn, fanOut);
        
        // We can't determine exact value due to randomness, but it should be initialized
        assertNotEquals(0.0, edge.getWeight());
    }

    @Test
    public void testSetWeight() {
        Node source = new Node(ActivationFunctions.sigmoid());
        Node target = new Node(ActivationFunctions.sigmoid());
        
        Edge edge = new Edge(source, target);
        edge.setWeight(0.5);
        
        assertEquals(0.5, edge.getWeight(), 1e-10);
    }

    @Test
    public void testUpdateWeight() {
        Node source = new Node(ActivationFunctions.sigmoid());
        Node target = new Node(ActivationFunctions.sigmoid());
        
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