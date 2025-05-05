package graph.search;

import graph.core.TreeNode;
import java.util.ArrayList;
import java.util.List;

/**
 * A specialized tree node for game tree search algorithms.
 * @param <T> The type of game state stored in the node
 */
public class GameTreeNode<T> implements TreeNode<T> {
    private T data;
    private GameTreeNode<T> parent;
    private List<GameTreeNode<T>> children;
    private double score;
    
    // For minimax/alpha-beta
    private double alpha = Double.NEGATIVE_INFINITY;
    private double beta = Double.POSITIVE_INFINITY;
    
    // For MCTS
    private int visits = 0;
    private double totalScore = 0;
    
    /**
     * Creates a new game tree node with the given data
     * @param data The game state data
     */
    public GameTreeNode(T data) {
        this.data = data;
        this.children = new ArrayList<>();
    }
    
    @Override
    public T getData() {
        return data;
    }
    
    @Override
    public List<GameTreeNode<T>> getChildren() {
        return children;
    }
    
    @Override
    public TreeNode<T> getParent() {
        return parent;
    }
    
    @Override
    public void addChild(TreeNode<T> child) {
        if (child instanceof GameTreeNode) {
            GameTreeNode<T> gameChild = (GameTreeNode<T>) child;
            gameChild.parent = this;
            children.add(gameChild);
        }
    }
    
    /**
     * Adds a child with the given data
     * @param childData The data for the child node
     * @return The created child node
     */
    public GameTreeNode<T> addChild(T childData) {
        GameTreeNode<T> child = new GameTreeNode<>(childData);
        addChild(child);
        return child;
    }
    
    @Override
    public boolean isLeaf() {
        return children.isEmpty();
    }
    
    /**
     * Gets the score for this node
     * @return The score
     */
    public double getScore() {
        return score;
    }
    
    /**
     * Sets the score for this node
     * @param score The score to set
     */
    public void setScore(double score) {
        this.score = score;
    }
    
    /**
     * Gets the alpha value for alpha-beta pruning
     * @return The alpha value
     */
    public double getAlpha() {
        return alpha;
    }
    
    /**
     * Sets the alpha value for alpha-beta pruning
     * @param alpha The alpha value to set
     */
    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }
    
    /**
     * Gets the beta value for alpha-beta pruning
     * @return The beta value
     */
    public double getBeta() {
        return beta;
    }
    
    /**
     * Sets the beta value for alpha-beta pruning
     * @param beta The beta value to set
     */
    public void setBeta(double beta) {
        this.beta = beta;
    }
    
    /**
     * Gets the number of visits for MCTS
     * @return The number of visits
     */
    public int getVisits() {
        return visits;
    }
    
    /**
     * Increments the visit count for MCTS
     */
    public void incrementVisits() {
        visits++;
    }
    
    /**
     * Gets the total score for MCTS
     * @return The total score
     */
    public double getTotalScore() {
        return totalScore;
    }
    
    /**
     * Adds to the total score for MCTS
     * @param score The score to add
     */
    public void addScore(double score) {
        totalScore += score;
    }
    
    /**
     * Calculates the UCB1 value for MCTS selection
     * @param explorationParam The exploration parameter
     * @return The UCB1 value
     */
    public double getUCB1(double explorationParam) {
        if (visits == 0) {
            return Double.POSITIVE_INFINITY;
        }
        
        double exploitation = totalScore / visits;
        double exploration = explorationParam * Math.sqrt(Math.log(((GameTreeNode<T>)parent).visits) / visits);
        
        return exploitation + exploration;
    }
}