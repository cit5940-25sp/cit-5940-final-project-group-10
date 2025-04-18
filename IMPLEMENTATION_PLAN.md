# Othello Game Implementation Plan

## Overview

This document outlines a structured implementation plan for completing the Othello game with neural network AI. The project consists of two main components:

1. **Othello Game Logic** - Core game mechanics and rule implementation
2. **AI Strategies** - Computer player implementations including neural network integration

## Project Structure

The project follows a modular Java application structure using Maven for build management and JavaFX for the GUI.

```
src/
├── main/
│   ├── java/
│   │   ├── deeplearningjava/  # Neural network framework (already implemented)
│   │   ├── othello/           # Main game package
│   │   │   ├── gamelogic/     # Game rules and logic
│   │   │   └── gui/           # User interface (already implemented)
│   │   ├── util/              # Utility classes
│   │   └── module-info.java   # Module definition
│   └── resources/
│       └── othello/           # FXML and resources
└── test/
    └── java/
        ├── deeplearningjava/  # Neural network tests (already implemented)
        ├── othello/           # Game logic tests (to be implemented)
        └── util/              # Utility tests
```

## Code Organization

The codebase will maintain separation between the reusable neural network framework and the Othello game as specified in the project structure documentation. This ensures the deep learning framework remains independent and reusable for other applications while the game-specific code stays within the Othello package.

### Recommended Structure

```
src/main/java/
├── deeplearningjava/  # Neural network framework (independent, reusable)
│   ├── Edge.java      # Already implemented
│   ├── Layer.java     # Already implemented
│   ├── Network.java   # Already implemented
│   └── Node.java      # Already implemented
├── graph/             # New general-purpose graph package (reusable)
│   ├── core/
│   │   ├── DirectedGraph.java  # Generic graph structure
│   │   ├── Tree.java           # Generic tree implementation
│   │   └── TreeNode.java       # Base tree node interface
│   ├── traversal/
│   │   ├── BreadthFirst.java   # BFS traversal
│   │   └── DepthFirst.java     # DFS traversal
│   └── search/               # Game tree search algorithms
│       ├── GameTreeNode.java   # Game-specific tree node
│       ├── MiniMax.java        # Minimax algorithm with alpha-beta pruning
│       ├── Expectimax.java     # Expectimax algorithm
│       └── MonteCarloTreeSearch.java # MCTS algorithm
├── othello/
│   ├── App.java       # Already implemented
│   ├── Constants.java # Already implemented
│   ├── gamelogic/
│   │   ├── BoardSpace.java     # Already implemented
│   │   ├── ComputerPlayer.java # To be expanded
│   │   ├── HumanPlayer.java    # Already implemented
│   │   ├── OthelloGame.java    # To be expanded
│   │   ├── Player.java         # To be expanded
│   │   └── strategies/         # New package for AI strategies
│   │       ├── Strategy.java         # Interface for all strategies
│   │       ├── StrategyFactory.java  # Factory for creating strategies
│   │       ├── BoardEvaluator.java   # Interface for evaluation functions
│   │       ├── WeightedEvaluator.java # Uses position weights
│   │       ├── MobilityEvaluator.java # Based on move availability
│   │       ├── MinimaxStrategy.java   # Implementation using minimax
│   │       ├── ExpectimaxStrategy.java # Implementation using expectimax
│   │       ├── MCTSStrategy.java      # Implementation using MCTS
│   │       ├── CustomStrategy.java    # Custom implementation
│   │       ├── NeuralStrategy.java    # Implementation using neural network
│   │       └── BoardToInputMapper.java # Converts game state to NN inputs
│   └── gui/
│       ├── GameController.java # Already implemented
│       └── GUISpace.java       # Already implemented
├── util/
│   └── HelloWorld.java # Already implemented
└── module-info.java    # Already implemented
```

### Design Rationale

1. **Separation of Concerns:**
   - The `deeplearningjava` package remains independent of the game logic
   - The neural network can be reused in other applications
   - The new `graph` package provides reusable tree/graph structures

2. **Reusable Graph Components:**
   - Common tree traversal algorithms (minimax, expectimax, MCTS) implemented once
   - Same graph structure used by all strategy implementations
   - Reduces code duplication and ensures consistent implementation

3. **Integration Points:**
   - The `othello.gamelogic.strategies.neural` package bridges game logic and neural network
   - Strategy implementations use the graph package's algorithms
   - `NeuralStrategy` implements the Strategy interface but uses the neural network for evaluation

4. **Modular Structure:**
   - Each strategy is in its own subpackage for clean organization
   - Common utilities (evaluation, factory) are shared across strategies
   - Graph algorithms are separated from game-specific logic

5. **Expansion Path:**
   - New strategies can be added as new packages
   - Improvements to the neural network or graph frameworks benefit all applications using them
   - Optimizations to traversal algorithms automatically benefit all strategies

This structure maintains the design goals in the project structure document while adding a new reusable component: "The clean separation between these components allows for independent development and testing, as well as potential reuse of the neural network framework in other applications." The same principle now applies to the graph package.

## Implementation Phases

### Phase 1: Core Game Logic Implementation

This phase focuses on implementing the foundational game mechanics to make the game playable.

#### 1.1 Player Available Moves (Player.java)

- Implement `getAvailableMoves(BoardSpace[][] board)` method
  - Find all empty spaces on the board
  - For each empty space, check in all 8 directions (N, NE, E, SE, S, SW, W, NW)
  - Identify valid moves where player's piece can flank opponent's pieces
  - Return a map of destinations to origins

```java
public Map<BoardSpace, List<BoardSpace>> getAvailableMoves(BoardSpace[][] board) {
    Map<BoardSpace, List<BoardSpace>> moves = new HashMap<>();
    // Implementation details...
    return moves;
}
```

#### 1.2 Space Claiming (OthelloGame.java)

- Implement `takeSpace(Player actingPlayer, Player opponent, int x, int y)` method
  - Check if space is already owned by the acting player
  - Claim the space for the acting player
  - Update player's owned spaces list

- Implement `takeSpaces(Player actingPlayer, Player opponent, Map<BoardSpace, List<BoardSpace>> availableMoves, BoardSpace selectedDestination)` method
  - Get list of origins for the selected destination
  - For each origin, flip all opponent pieces along the path to the destination
  - Update both players' owned spaces lists

### Phase 2: Graph Package and AI Strategy Implementation

This phase is split into two parts: first creating a reusable graph package, then implementing the game-specific strategies.

#### 2.1 Graph Package Implementation

Create a new graph package to provide reusable tree and graph structures:

```java
// In graph.core package - Base tree node interface
public interface TreeNode<T> {
    T getData();
    List<? extends TreeNode<T>> getChildren();
    TreeNode<T> getParent();
    void addChild(TreeNode<T> child);
    boolean isLeaf();
}

// In graph.search package - Game-specific tree node 
public class GameTreeNode<T> implements TreeNode<T> {
    private T data;
    private GameTreeNode<T> parent;
    private List<GameTreeNode<T>> children = new ArrayList<>();
    private double score;
    
    // Additional properties for search algorithms
    private int visits = 0;  // For MCTS
    private double alpha = Double.NEGATIVE_INFINITY;  // For minimax
    private double beta = Double.POSITIVE_INFINITY;   // For minimax
    
    // Implementation of methods...
}

// In graph.search package - Minimax implementation
public class MiniMax {
    public static <T> double search(GameTreeNode<T> node, int depth, 
                                   boolean maximizingPlayer, double alpha, double beta,
                                   BiFunction<T, Boolean, Double> evaluator) {
        // Implementation of minimax with alpha-beta pruning
    }
}
```

#### 2.2 Strategy Interface and Common Utilities

Create a common foundation for all strategies:

```java
// In othello.gamelogic.strategies package - Strategy interface
public interface Strategy {
    BoardSpace getBestMove(OthelloGame game, Player currentPlayer, Player opponent);
}

// In othello.gamelogic.strategies package - Factory for creating strategies
public class StrategyFactory {
    public static Strategy createStrategy(String strategyName) {
        return switch(strategyName) {
            case "minimax" -> new MinimaxStrategy();
            case "expectimax" -> new ExpectimaxStrategy();
            case "mcts" -> new MCTSStrategy();
            case "custom" -> new CustomStrategy();
            default -> throw new IllegalArgumentException("Unknown strategy: " + strategyName);
        };
    }
}

// In othello.gamelogic.strategies package - Board evaluation interface
public interface BoardEvaluator {
    double evaluate(BoardSpace[][] board, Player player, Player opponent);
}

// In othello.gamelogic.strategies package - Position-weighted evaluator
public class WeightedEvaluator implements BoardEvaluator {
    @Override
    public double evaluate(BoardSpace[][] board, Player player, Player opponent) {
        // Use position weights from Constants.BOARD_WEIGHTS
        // Return score based on weighted positions
    }
}

// In othello.gamelogic.strategies package - Mobility-based evaluator
public class MobilityEvaluator implements BoardEvaluator {
    @Override
    public double evaluate(BoardSpace[][] board, Player player, Player opponent) {
        // Evaluate based on number of available moves
        // Return score based on mobility difference
    }
}
```

#### 2.3 Computer Player Implementation

Update the ComputerPlayer class to use the StrategyFactory and Strategy interface:

```java
public class ComputerPlayer extends Player {
    private final Strategy strategy;
    
    public ComputerPlayer(String strategyName) {
        this.strategy = StrategyFactory.createStrategy(strategyName);
    }
    
    public BoardSpace getBestMove(OthelloGame game, Player opponent) {
        return strategy.getBestMove(game, this, opponent);
    }
}
```

#### 2.4 Strategy Implementations Using Graph Package

Each strategy will use the graph package's search algorithms:

##### Minimax Strategy Implementation

The minimax strategy will use the graph package's MiniMax algorithm:

```java
public class MinimaxStrategy implements Strategy {
    private final BoardEvaluator evaluator;
    private final int maxDepth;
    
    public MinimaxStrategy() {
        this.evaluator = new WeightedEvaluator();
        this.maxDepth = 4; // Configurable depth
    }
    
    @Override
    public BoardSpace getBestMove(OthelloGame game, Player currentPlayer, Player opponent) {
        // Create game state representation
        GameState initialState = new GameState(game.getBoard(), currentPlayer, opponent);
        
        // Create root node for the search tree
        GameTreeNode<GameState> rootNode = new GameTreeNode<>(initialState);
        
        // Use graph package's MiniMax implementation
        double bestScore = graph.search.MiniMax.search(
            rootNode, 
            maxDepth, 
            true, // maximizing player
            Double.NEGATIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            (state, isMax) -> evaluator.evaluate(state.getBoard(), 
                                               isMax ? currentPlayer : opponent,
                                               isMax ? opponent : currentPlayer)
        );
        
        // Find child with the best score
        return getBestChildMove(rootNode);
    }
}
```

##### Expectimax Strategy Implementation

Similar to minimax, but using the graph package's Expectimax algorithm:

```java
public class ExpectimaxStrategy implements Strategy {
    private final BoardEvaluator evaluator;
    private final int maxDepth;
    
    public ExpectimaxStrategy() {
        this.evaluator = new WeightedEvaluator();
        this.maxDepth = 3; // May need to be lower than minimax due to branching
    }
    
    @Override
    public BoardSpace getBestMove(OthelloGame game, Player currentPlayer, Player opponent) {
        // Create game state representation
        GameState initialState = new GameState(game.getBoard(), currentPlayer, opponent);
        
        // Create root node for the search tree
        GameTreeNode<GameState> rootNode = new GameTreeNode<>(initialState);
        
        // Use graph package's Expectimax implementation
        double bestScore = graph.search.Expectimax.search(
            rootNode, 
            maxDepth, 
            true, // maximizing player
            (state, isMax) -> evaluator.evaluate(state.getBoard(), 
                                               isMax ? currentPlayer : opponent,
                                               isMax ? opponent : currentPlayer)
        );
        
        // Find child with the best score
        return getBestChildMove(rootNode);
    }
}
```

##### Monte Carlo Tree Search Implementation

The MCTS strategy will use the graph package's MonteCarloTreeSearch algorithm:

```java
public class MCTSStrategy implements Strategy {
    private final double explorationParameter;
    private final int simulationCount;
    
    public MCTSStrategy() {
        this.explorationParameter = Constants.EXPLORATION_PARAM;
        this.simulationCount = 1000; // Number of simulations to run
    }
    
    @Override
    public BoardSpace getBestMove(OthelloGame game, Player currentPlayer, Player opponent) {
        // Create game state representation
        GameState initialState = new GameState(game.getBoard(), currentPlayer, opponent);
        
        // Create root node for the search tree
        GameTreeNode<GameState> rootNode = new GameTreeNode<>(initialState);
        
        // Use graph package's MCTS implementation
        graph.search.MonteCarloTreeSearch.search(
            rootNode,
            simulationCount,
            explorationParameter,
            this::simulateRandomGame,  // Game simulation function
            this::getGameResult        // Result evaluation function
        );
        
        // Choose the child with the highest visit count
        return getMostVisitedChildMove(rootNode);
    }
    
    // Helper methods for simulating random game outcomes
    private GameState simulateRandomGame(GameState state) {
        // Implementation for random game simulation
    }
    
    private double getGameResult(GameState finalState, Player player) {
        // Implementation for determining win/loss/draw
    }
}
```

##### Custom Strategy Implementation

Create a custom strategy that could combine multiple approaches or use the neural network:

```java
public class CustomStrategy implements Strategy {
    private final BoardEvaluator evaluator;
    private final Strategy earlygameStrategy;
    private final Strategy midgameStrategy;
    private final Strategy endgameStrategy;
    private final int midgameThreshold = 20; // Board spaces filled
    private final int endgameThreshold = 50; // Board spaces filled
    
    public CustomStrategy() {
        this.evaluator = new MobilityEvaluator(); // Different evaluator
        
        // Initialize different strategies for different game phases
        this.earlygameStrategy = new MCTSStrategy();
        this.midgameStrategy = new MinimaxStrategy();
        this.endgameStrategy = new ExpectimaxStrategy();
    }
    
    @Override
    public BoardSpace getBestMove(OthelloGame game, Player currentPlayer, Player opponent) {
        int filledSpaces = countFilledSpaces(game.getBoard());
        
        // Use different strategies based on game phase
        if (filledSpaces < midgameThreshold) {
            return earlygameStrategy.getBestMove(game, currentPlayer, opponent);
        } else if (filledSpaces < endgameThreshold) {
            return midgameStrategy.getBestMove(game, currentPlayer, opponent);
        } else {
            return endgameStrategy.getBestMove(game, currentPlayer, opponent);
        }
    }
    
    private int countFilledSpaces(BoardSpace[][] board) {
        // Count non-empty spaces on the board
    }
}
```

#### 2.5 Computer Decision Integration

- Implement `computerDecision(ComputerPlayer computer)` in OthelloGame
  - Get available moves for the computer player
  - Use the computer's strategy to select the best move
  - Return the selected destination

```java
public BoardSpace computerDecision(ComputerPlayer computer) {
    Map<BoardSpace, List<BoardSpace>> availableMoves = getAvailableMoves(computer);
    if (availableMoves.isEmpty()) {
        return null;
    }
    
    Player opponent = (computer == playerOne) ? playerTwo : playerOne;
    return computer.getBestMove(this, opponent);
}
```

### Phase 3: Neural Network Integration

This phase focuses on integrating the neural network framework with the Othello game.

#### 3.1 Neural Network Strategy

Create a strategy that uses the neural network for board evaluation:

```java
public class NeuralStrategy implements Strategy {
    private final Network network;
    
    public NeuralStrategy(Network network) {
        this.network = network;
    }
    
    @Override
    public BoardSpace getBestMove(OthelloGame game, Player currentPlayer, Player opponent) {
        // Implementation details...
    }
    
    private double evaluateWithNetwork(BoardSpace[][] board, Player player, Player opponent) {
        // Convert board to network input
        // Feed forward through the network
        // Return the evaluation score
    }
}
```

#### 3.2 Board Representation

Create utilities to convert between game state and neural network input/output:

```java
public class BoardToInputMapper {
    public static double[] mapToInput(BoardSpace[][] board, Player currentPlayer) {
        // Convert board state to array of doubles
        // Format: 1 for current player's pieces, -1 for opponent's, 0 for empty
    }
    
    public static BoardSpace mapToMove(double[] output, Map<BoardSpace, List<BoardSpace>> availableMoves) {
        // Convert network output to board space
    }
}
```

#### 3.3 Neural Network Training

Create a training framework for the neural network:

```java
public class NeuralNetworkTrainer {
    private final Network network;
    
    public NeuralNetworkTrainer(Network network) {
        this.network = network;
    }
    
    public void trainOnGameData(List<GameRecord> gameRecords) {
        // Implementation details...
    }
    
    public void selfPlay(int games) {
        // Implementation details...
    }
}
```

### Phase 4: Testing and Documentation

This phase focuses on ensuring the correctness of the implementation and providing comprehensive documentation.

#### 4.1 Unit Tests

Create unit tests for all implemented components:

```java
public class PlayerTest {
    @Test
    public void testGetAvailableMoves() {
        // Test implementation...
    }
}

public class MinimaxStrategyTest {
    @Test
    public void testGetBestMove() {
        // Test implementation...
    }
}
```

#### 4.2 Documentation

Add comprehensive JavaDoc to all implemented classes and methods following the guidelines in CLAUDE.md:

```java
/**
 * Represents a strategy for computer players based on the Minimax algorithm.
 * Uses alpha-beta pruning for improved performance and a position-weighted
 * evaluation function.
 */
public class MinimaxStrategy implements Strategy {
    // Implementation...
}
```

## Timeline

- **Week 1**: Complete Phase 1 (Core Game Logic)
  - Implement Player.getAvailableMoves()
  - Implement OthelloGame.takeSpace() and takeSpaces()
  - Test basic game functionality

- **Week 2 (Part 1)**: Create Graph Package
  - Implement base graph and tree structures
  - Implement TreeNode and GameTreeNode classes
  - Implement traversal algorithms (BFS, DFS)
  - Implement game-specific algorithms:
    - Minimax with alpha-beta pruning
    - Expectimax
    - Monte Carlo Tree Search

- **Week 2 (Part 2)**: Complete Phase 2 (AI Strategy Implementation)
  - Implement Strategy interface and StrategyFactory
  - Implement evaluation functions
  - Implement concrete strategies:
    - MinimaxStrategy (using graph.traversal.MiniMax)
    - ExpectimaxStrategy (using graph.traversal.Expectimax)
    - MCTSStrategy (using graph.traversal.MonteCarloTreeSearch)
    - CustomStrategy
  - Implement ComputerPlayer decision-making logic

- **Week 3**: Complete Phase 3 (Neural Network Integration)
  - Integrate deep learning framework with strategy evaluation
  - Implement NeuralStrategy using the existing neural network
  - Train initial models for board evaluation
  - Implement BoardToInputMapper to convert game states to neural inputs

- **Week 4**: Complete Phase 4 (Testing, Documentation, and Refinement)
  - Test all strategies against each other
  - Develop benchmark suite to compare strategy performance
  - Complete documentation for all components
  - Optimize for performance and code quality

## Build and Test Instructions

- Build and run: `mvn clean javafx:run`
- Run tests: `mvn test`
- Run single test: `mvn test -Dtest=TestClassName#testMethodName`
- Compile only: `mvn compile`

## Future Enhancements

- Implement additional AI strategies
- Add tournament mode for comparing strategies
- Create a training pipeline for neural networks
- Add game state visualization and analysis tools
- Implement UI improvements for better user experience