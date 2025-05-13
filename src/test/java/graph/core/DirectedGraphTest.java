package graph.core;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.List;
import java.util.Set;

/**
 * Tests for the DirectedGraph class.
 */
public class DirectedGraphTest {
    
    private DirectedGraph<String> graph;
    
    @BeforeEach
    public void setUp() {
        graph = new DirectedGraph<>();
    }
    
    @Test
    public void testAddNode() {
        // Add a single node
        graph.addNode("A");
        
        // Check nodes in graph
        Set<String> nodes = graph.getNodes();
        assertEquals(1, nodes.size());
        assertTrue(nodes.contains("A"));
        
        // Add same node again (should be idempotent)
        graph.addNode("A");
        assertEquals(1, graph.getNodes().size());
        
        // Add another node
        graph.addNode("B");
        assertEquals(2, graph.getNodes().size());
        assertTrue(graph.getNodes().contains("B"));
    }
    
    @Test
    public void testAddEdge() {
        // Add an edge directly (should also create the nodes)
        graph.addEdge("A", "B");
        
        // Check nodes created
        Set<String> nodes = graph.getNodes();
        assertEquals(2, nodes.size());
        assertTrue(nodes.contains("A"));
        assertTrue(nodes.contains("B"));
        
        // Check edge created
        List<String> neighborsOfA = graph.getNeighbors("A");
        assertEquals(1, neighborsOfA.size());
        assertEquals("B", neighborsOfA.get(0));
        
        // B should have no outgoing edges
        List<String> neighborsOfB = graph.getNeighbors("B");
        assertTrue(neighborsOfB.isEmpty());
    }
    
    @Test
    public void testMultipleEdges() {
        // Create a graph with multiple edges
        graph.addEdge("A", "B");
        graph.addEdge("A", "C");
        graph.addEdge("B", "D");
        graph.addEdge("C", "D");
        
        // Check edges
        List<String> neighborsOfA = graph.getNeighbors("A");
        assertEquals(2, neighborsOfA.size());
        assertTrue(neighborsOfA.contains("B"));
        assertTrue(neighborsOfA.contains("C"));
        
        List<String> neighborsOfB = graph.getNeighbors("B");
        assertEquals(1, neighborsOfB.size());
        assertEquals("D", neighborsOfB.get(0));
        
        List<String> neighborsOfC = graph.getNeighbors("C");
        assertEquals(1, neighborsOfC.size());
        assertEquals("D", neighborsOfC.get(0));
        
        List<String> neighborsOfD = graph.getNeighbors("D");
        assertTrue(neighborsOfD.isEmpty());
    }
    
    @Test
    public void testGetNeighborsNonExistentNode() {
        // Try to get neighbors for a non-existent node
        List<String> neighbors = graph.getNeighbors("X");
        
        // Should return empty list
        assertNotNull(neighbors);
        assertTrue(neighbors.isEmpty());
    }
    
    @Test
    public void testEmptyGraph() {
        // Test empty graph
        assertTrue(graph.getNodes().isEmpty());
    }
    
    @Test
    public void testSelfLoop() {
        // Add a self-loop
        graph.addEdge("A", "A");
        
        // Check node created
        assertEquals(1, graph.getNodes().size());
        
        // Check self-loop created
        List<String> neighbors = graph.getNeighbors("A");
        assertEquals(1, neighbors.size());
        assertEquals("A", neighbors.get(0));
    }
}