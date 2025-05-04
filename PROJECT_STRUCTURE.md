# Project Structure: Othello Game with Neural Network Framework

## Overview

This project consists of two main components:
1. **Othello Game** - A Java implementation of the classic Othello/Reversi game
2. **Deep Learning Framework** - A reusable neural network library that can be used for AI players

The project is structured as a modular Java application using the Java Platform Module System (JPMS) and Maven for dependency management and building.

## Module Structure

The application is packaged as a single module named `othello`, which exports several packages:

```java
module othello {
    requires javafx.controls;
    requires javafx.fxml;

    opens othello to javafx.fxml;
    exports othello;
    exports othello.gui;
    opens othello.gui to javafx.fxml;
    exports othello.gamelogic;
    opens othello.gamelogic to javafx.fxml;
    exports deeplearningjava;
    exports util;
}
```

This module configuration:
- Requires JavaFX dependencies for the GUI
- Exports the main packages for external use
- Opens specific packages to JavaFX FXML for UI construction

## Package Structure

### 1. Othello Game

#### 1.1 `othello` (Root Package)
- **App.java** - Entry point for the JavaFX application
- **Constants.java** - Game constants and configuration

#### 1.2 `othello.gamelogic`
- **BoardSpace.java** - Represents a space on the game board
- **OthelloGame.java** - Main game logic and rules
- **Player.java** - Abstract player class
- **HumanPlayer.java** - Implementation for human players
- **ComputerPlayer.java** - Implementation for AI-controlled players

#### 1.3 `othello.gui`
- **GUISpace.java** - UI component for board spaces
- **GameController.java** - JavaFX controller for the game interface
- **game-view.fxml** - JavaFX layout file (resources directory)

### 2. Deep Learning Framework

#### 2.1 `deeplearningjava`
- **Node.java** - Neural network node/neuron implementation
- **Edge.java** - Connection between nodes with weights
- **Layer.java** - Collection of nodes forming a network layer
- **Network.java** - Complete neural network with layers

The deep learning framework is designed to be independent of the Othello game, making it reusable for other applications. It implements feedforward neural networks with various activation functions including:
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- GELU
- Linear
- Softmax (for output layer)

### 3. Utilities

#### 3.1 `util`
- **HelloWorld.java** - Simple utility class (example/test)

## Test Structure

Following Maven conventions, tests are organized in parallel to the main source code:

### 1. Deep Learning Tests

#### 1.1 `deeplearningjava` (Test Package)
- **NodeTest.java** - Tests for neural network nodes
- **EdgeTest.java** - Tests for connections between nodes
- **LayerTest.java** - Tests for neural network layers
- **NetworkTest.java** - Tests for the complete neural network
- **DeepLearningIntegrationTest.java** - Integration tests for the framework

### 2. Utility Tests

#### 2.1 `util` (Test Package)
- **HelloWorldTest.java** - Simple test case

## Build Configuration

The project uses Maven for build management with configuration in `pom.xml`:

- Java 20 target
- JUnit 5 for testing
- JavaFX dependencies
- JaCoCo for test coverage

## Key Design Patterns

1. **Model-View-Controller (MVC)** - Separates game logic from UI
   - Model: `othello.gamelogic` classes
   - View: FXML layouts and GUI classes
   - Controller: `GameController.java`

2. **Strategy Pattern** - Different player implementations
   - Abstract `Player` class with specific strategies (`HumanPlayer`, `ComputerPlayer`)

3. **Factory Method** - Creation of different game components
   - Methods for creating player instances based on strategy

4. **Composite Pattern** - Neural network structure
   - `Network` contains `Layer`s contains `Node`s connected by `Edge`s

## Test Coverage

The deep learning framework has extensive test coverage:
- 94.3% instruction coverage overall
- 87.9% branch coverage
- 93.7% line coverage
- 96.6% method coverage

For detailed test coverage analysis, see `DEEPLEARNING_TEST_COVERAGE.md`.

## Future Integration Points

The project is designed to allow the neural network framework to be used for AI players in the Othello game:

1. Create a `NeuralNetworkPlayer` implementation that extends `ComputerPlayer`
2. Train the neural network on game states and moves
3. Use the trained network for move prediction

This integration would demonstrate how a general-purpose neural network framework can be applied to game AI.

## Directory Structure Summary

```
src/
├── main/
│   ├── java/
│   │   ├── deeplearningjava/       # Neural network framework
│   │   ├── othello/                # Main game package
│   │   │   ├── gamelogic/          # Game rules and logic
│   │   │   └── gui/                # User interface
│   │   ├── util/                   # Utility classes
│   │   └── module-info.java        # Module definition
│   └── resources/
│       └── othello/                # FXML and resources
└── test/
    └── java/
        ├── deeplearningjava/       # Neural network tests
        └── util/                   # Utility tests
```

## Conclusion

This project demonstrates a well-structured, modular Java application that separates concerns between:
1. The Othello game implementation
2. The reusable neural network framework
3. Testing infrastructure

The clean separation between these components allows for independent development and testing, as well as potential reuse of the neural network framework in other applications.