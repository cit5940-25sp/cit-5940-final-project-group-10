package graph.core;

import java.util.*;

/**
 * Represents a directed graph data structure.
 * @param <T> The type of data stored in the graph nodes
 */
public class DirectedGraph<T> {
    private Map<T, List<T>> adjacencyList;
    
    /**
     * Creates a new empty directed graph
     */
    public DirectedGraph() {
        adjacencyList = new HashMap<>();
    }
    
    /**
     * Adds a node to the graph
     * @param node The node to add
     */
    public void addNode(T node) {
        adjacencyList.putIfAbsent(node, new ArrayList<>());
    }
    
    /**
     * Adds a directed edge from source to destination
     * @param source The source node
     * @param destination The destination node
     */
    public void addEdge(T source, T destination) {
        // Add nodes if they don't exist
        addNode(source);
        addNode(destination);
        
        // Add the edge
        adjacencyList.get(source).add(destination);
    }
    
    /**
     * Gets all nodes in the graph
     * @return Set of all nodes
     */
    public Set<T> getNodes() {
        return adjacencyList.keySet();
    }
    
    /**
     * Gets all neighbors (outgoing edges) of a node
     * @param node The node to get neighbors for
     * @return List of neighbor nodes
     */
    public List<T> getNeighbors(T node) {
        return adjacencyList.getOrDefault(node, Collections.emptyList());
    }
}