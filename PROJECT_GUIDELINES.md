# Othello Final Project

## Overview
In this final project, you will design and implement the board game of Othello with varying levels of computer AI decision making algorithms.

This project will involve creating a game of Othello allowing three modes of play:
- A human player against another human player
- A human player against a computer player
- A computer player vs another computer player

You will be designing various Artificial Intelligence algorithms that allow the computer to make decisions. In the end, you should be able to play the game using the provided GUI interface, and the code should be testable without the interface as well.

The concepts covered in this project will involve working with a grid board using some traversal algorithms, building many trees that implement popular decision making and game theory algorithms, and working with briefly interfacing with GUIs and the dependency management system Maven.

## Starter Files
**Topics:** Graph Traversal, Game Theory, Decision Trees, Artificial Intelligence

## Introduction to Game AI
We can model the execution of two-player games like Othello as a graph. In this graph, we have nodes—these nodes represent the current state of the game, including the locations of the pieces on the board and the identifier for the player whose turn it is to make a move. We can transition from board state to board state by making moves. If a board state can lead to another state through a move, we would connect the nodes representing those moves with edges.

This high-level view admits a few additonal definitions:

- The **successors**, or board states reachable by making any of the possible moves, are the children of a node.
- The set of all possible board states available in *n* turns are those descendents found on a path of length *n* originating at the current board state.
- The full execution of a game is represented by a path from the start state of the game to some terminal node.

Using artificial intelligence to play games well is therefore a problem of enumerating and evaluating paths in the graph that represents the gameplay. Throughout the course of this project, you will attempt to make capable computer players by interacting with these graphs.

## Introduction to Othello
Othello, sometimes known as Reversi, is a board game that involves flipping your opponent's colored discs on an 8x8 grid board. The objective of the game is to have the majority of the discs showing at the end of the game be your color.

In order to achieve this, moves must be made that "outflank" the opponent's discs. This means that you may place a disc on the board where your discs will border the opponent's discs on either side.

For example, the game begins with the given configuration:

[*Image: Initial Othello board setup with 2 black and 2 white pieces in the center*]

Black always starts the game. Black can make the following moves highlighted below:

[*Image: Initial board with highlighted valid moves for Black*]

Say Black moves to take the top left White disc using their top right Black disc. The board would change and White would be given the following moves:

[*Image: Updated board after Black's move with highlighted valid moves for White*]

### Additional Moves

### Ending the Game
The game ends when either player cannot make a move. This may occur when the board is not full.

If the board is not full but a player cannot make any moves, their turn is skipped and the other player may take another turn.

You do not need to worry about implementing the above two behaviors, as they have been implemented for you in the GUI.

### Multiple Paths
If there are multiple paths to the same space on the board from multiple of your discs, all of your opponents discs on those paths get flipped. For instance, if it is Black's turn and we have the following board:

[*Image: Board showing multiple paths scenario*]

Choosing the second space from the right will get you the following board state:

[*Image: Board after multiple paths move*]

These rules are all you need to know for the purposes of this project. You may read a more in-depth coverage of the rules [here](https://www.worldothello.org/about/about-othello/othello-rules/official-rules/english).

## Part 0: Starter Code Walkthrough
**Files:** GameController.java, App.java

This section is meant to give you a basic overview of how the starter code and Maven works, as the structure of it will slightly differ from the starter files you've been given before in this class. You do not, and should not need to modify any of these files. Please reach out to the course staff as soon as possible if your starter files do not function as detailed below.

Upon unzipping the starter files, your folder structure should more or less look like below:

[*Image: Folder structure diagram*]

There are a few extra files that you may not have seen before, mainly pom.xml, a few files related to Maven, the resources folder, and module-info.java.

**Do not massively change the file structure from this image unless you absolutely know what you are doing!**

You may make new files and classes in the gamelogic package; they should all have `package othello.gamelogic` in their headers to work properly.

Feel free to make your own packages for organization purposes as well.

### Maven
pom.xml and the other miscellaneous mvn files are related to Maven. Maven is a build automation tool, and for the purposes of this assignment, it manages certain dependencies so that you do not need to install anything to get it working.

The most important dependency included with Maven in this project is JavaFX, which is the framework used for the GUI. Additionally, JUnit 5 is included in the dependencies as well.

As soon as you open the starter code as a project in IntelliJ, Maven should handle the dependencies for you, thus requiring no action on your end to install any plugins or files. We will detail below how you can check if this aspect of the starter code is working.

### JavaFX
JavaFX is a modern GUI framework for Java that we will be using for this assignment. The resources folder contains a GUI view, while the module-info.java file links all the GUI and logical packages of the application together.

The App.java class runs the GUI, allowing the GameController.java class to create both the view and the model of the game. These parts encompass a Model-View-Controller (MVC) design pattern.

If you run App.java, you should get the following window popping up with no errors in the console:

[*Image: Screenshot of the GUI window*]

If you see this view (this should open in a new window), you are good to go, and JavaFX is properly working. If you do not see this view or have an error, please contact the teaching staff as soon as possible so that we can diagnose the issue.

The GUI code has been written for you and is housed in the gui package. You do not need to write any extra GUI code unless you dive into some JavaFX documentation for extra features for your final submission. Your game logic will mainly be written within the gamelogic package.

### Program Arguments
In order to run the game, App.java enforces that there needs to be 2 program arguments, each representing a player in the game. Similar to HW5, you can specify these arguments through the program's run configuration.

Right click on App.java -> Modify Run Configuration. In the "Program argument:" box, you will enter 2 strings separated by a space. These strings are: "human", "minimax", "expectimax", "mcts", and "custom". As their names imply, they either represent a human player that uses user input, or a computer strategy. For instance, a human vs. human game would have the arguments `human human`. A human vs. computer using MCTS would have the arguments `human mcts`. Two computers each using the Minimax strategy would have the arguments `minimax minimax`.

[*Image: Screenshot of run configuration*]

App.java will warn you if you do not properly have the right arguments. Be sure to confirm what arguments you are using before running the GUI code.

## Part 1: Implementing Othello
**Files:** Player.java, OthelloGame.java, BoardSpace.java

The first part of this assignment will involve getting the basic functionality of Othello to work with 2 human players. These two players are specified with the program arguments `human human`.

You will implement all of the TODO items within Player.java and OthelloGame.java labeled with "Part 1". This excludes the computerDecision() method within OthelloGame, as you will implement computer logic in the next part.

The methods within the above classes revolve around finding the next available spaces given a board state. Refer to the introduction sections for what moves are valid given certain board states. In order to accomplish this task, you will need to explore some of the other classes and pre-written logic within the gamelogic package, mainly the BoardSpace.java file.

You may also find it helpful to look at the GameController's methods, particularly showMove() and selectSpace(), as these methods are invoked upon initially showing the available moves and clicking valid spaces, respectively.

Pay attention to the data structures that should be input and output from each method, as it is important for the data to output correctly to the GUI.

### Testing this section

With the way the methods are organized, you should be able to write unit tests by simulating real boards and expecting certain spaces to be the output.

Additionally, as you have a GUI, you can play a human vs. human game to see if your methods output the correct spaces at a glance.

## Part 2: Computer AI
**Files:** ComputerPlayer.java, and any other files you make for the Strategy Design Pattern

In this next section, you will implement the logic for the computer decision-making AI. First, you will implement a Strategy Design Pattern to allow different strategies to be chosen depending on the program argument input.

You will begin in ComputerPlayer.java. The final result from a computer AI decision should be a single BoardSpace. From here, you should utilize the Strategy Design Pattern to create classes and interfaces that change the strategy used for each new ComputerPlayer depending on the program arguments.

Then, you will implement one of two backtracking algorithms, either Minimax or Expectimax, as one of your strategies within the design. Next, implement Monte Carlo Tree Search, a heuristic search algorithm. Finally, after implementing those two algorithms, you must define your own computer AI strategy. The three named algorithms are detailed below.

### Minimax
Minimax is a backtracking algorithm that can be used for decision making in games. The concept of the algorithm revolves around two players: a maximizer that tries to get the highest score possible and a minimizer that tries to get the lowest score possible. In the context of Othello, a "score" can be defined by evaluating a board through certain heuristics. Boards are evaluated once some max depth is reached, and we "backtrack" through the tree, alternating between the minimizer and maximizer until we maximize the children of the root node.

#### Board Evaluation
Boards must be evaluated through some structured manner in order to determine what board states are "good" for the maximizer and minimizer to select. To do this, you will need to simulate future states of the Othello board without affecting the original board itself. You may use the copy constructor within BoardSpace for this purpose, or find your own way to make a copy of the board.

To evaluate a board, we need to provide some heuristic for what makes certain spaces good or bad. In Othello, the most valuable spaces to take are the corner spaces, as they can never be flipped once taken, and the edge spaces, as they are quite stable once taken and less likely to be flipped. Conversely, the spaces adjacent to the corners and edges are disadvantageous, as they open the opponent up to taking those good spaces.

With these ideas in mind, we can evaluate a board by assigning weights to a board and summing up the weights of all of a current player's spaces subtracted by sum of the weight of all of their opponent's spaces. We have provided a sample set of board weights for an 8x8 board in the Constants class.

For example, if White has the board state:

[*Image: Example board for evaluation*]

Then the evaluated board state given the provided weights is

(2 + 2 + 2)[White Spaces] - (2 + 2 + 2 + 2 - 10 + 30)[Black Spaces] = -22

#### Minimax Example
Let's use the board below, with a computer on Black making a decision:

[*Image: Sample board for Minimax example*]

There are 5 possible boards you must simulate. Those boards then lead to a variety of other board states. Let's just say for this example that they lead to 1 or 2 board states each, with the following evaluated weights at a max depth of 2:

[*Image: Tree of board states with weights*]

At this depth, it is the minimizing player round, thus we choose the minimum among the children of each node, backtracking up the tree:

[*Image: Minimax tree after minimizing*]

Then finally, it is the maximizing player's turn, and among the children, the node containing 8 would be the most optimal choice. Thus, the result of this Minimax strategy decision would be the BoardSpace associated with the child containing 8.

This is just a dummy example with fake boards. In a real board, the number of children for each simulated board state will grow very rapidly given late-game boards. You will need to design your own Node class to store certain information about this strategy. Be wary about edge cases- for instance, what might happen if there are no available moves for a given simulated board?

### Expectimax
Minimax is an algorithm that assumes the opponent (the minimizer) will play optimally. However, this isn't entirely realistic, as human players will often make mistakes or take nonoptimal moves at the moment for future gain. Expecimax is another backtracking algorithm that models these probabilistic interactions.

For the sake of this assignment, you will be designing Expectimax to be similar to Minimax, but with the minimizing nodes being replaced with chance nodes instead. Refer to the previous Minimax section about designing a general backtracking algorithm. You may use the same board evaluation in this algorithm.

Chance nodes are nodes that take the expected value of their children to determine their values. We will model this behavior below.

#### Expectimax Example
Let's use the example from Minimax, with the following board and tree:

[*Image: Sample board and tree for Expectimax*]

In our version of Expectimax, the nodes at depth 1 are now chance nodes rather than minimizing nodes. While the 4 nodes with 1 child each will look the same with expected values, the rightmost node will have a different value:

[*Image: Expectimax tree with chance nodes*]

Thus, the final maximization will result in choosing the rightmost node with value 10 instead, resulting in a different answer than Minimax. As you can see, Expectimax's probabilistic nature took more risks that could lead to future gain for the opponent (the -10 node) and the current player (the +30 node), modeling the unpredictable nature of strategy games like this better than pure optimal play.

### Monte Carlo Tree Search
Monte Carlo Tree Search, abbreviated MCTS, is a heuristic search algorithm that is used for probabilistic sampling and exploration of a decision tree for certain games. MCTS builds a tree using 4 main steps:

1. **Selection**: traversing the tree to a leaf node, selecting a promising node for expansion
2. **Expansion**: expanding the tree to by adding child nodes
3. **Simulation**: play a game using random legal moves until the game ends
4. **Backpropagation**: update the node statistics based on the simulation result

MCTS aims to balance exploration of new nodes and exploitation of existing nodes within the selection process, in order to statistically produce winning moves based on many random simulations. The 4 steps above are usually run with a set amount of iterations, or until some time constraint or timer.

As MCTS has many variations and possible different adaptations, this section will detail one variation of the algorithm that you can implement for this project. You may decide to design certain aspects differently, but the process should still resemble MCTS and it's 4 steps in the end!

#### Selection
Selection involves traversing the current tree until you find a leaf node. To select a node, you should use the Upper Confidence Bound for Trees or UCT formula, which is as follows:

![UCT Formula: wi/ni + c * sqrt(ln(N)/ni)](https://latex.codecogs.com/png.latex?%5Cfrac%7Bw_i%7D%7Bn_i%7D%20&plus;%20c%20%5Csqrt%7B%5Cfrac%7B%5Cln%20N%7D%7Bn_i%7D%7D)

We apply this formula to every child of the currently traversed node, and select the node that yields the highest value from this formula.

Given the parameters of this formula, each node in your MCTS tree should store the number of simulations that resulted in wins and how many times it has been visited.

For the constant c, it represents the ratio in which we value exploration and exploitation. A higher value means we value exploration, while a lower value means we value exploitation. A good default value you can use for this is sqrt(2). This value is included in the Constants class as EXPLORATION_PARAM.

This formula essentially allows a balance of exploration and exploitation. As certain branches of the tree become more visited, the right side of the formula will get smaller, disincentivizing future traversals of that side.

This formula will be used starting from the root until a leaf node is reached, where we proceed to the next step.

#### Unexplored Nodes and Ties

Throughout the process of building this tree, you will come across nodes that haven't been visited. Due to the formula, if a node hasn't been visited, it will result in an undefined value, due to division by zero. In these cases, you should set the value to infinity, as we want to incentivize traversing these unexplored nodes.

If two nodes end up being tied in value, either due to both of them being unexplored or both of them resulting in the same value from the UCT formula, you may choose randomly among tied nodes.

#### Expansion
Upon reaching a leaf node, you should expand the tree by adding all possible moves for the leaf's current board state as children of the node. Then you must selecting a random move among these moves to proceed onto the next step.

#### Simulation
From your randomly selected child's board state, you should simulate a random game of Othello until the game ends. This involves both players selecting random legal moves from the list of possible moves, until both players cannot make a move, as per Othello rules.

At the end of this simulation, there should be a definitive result as to whether the current player has won or lost the Othello simulation.

#### Backpropagation
Finally, with the result of having won or lost the simulation, we update the tree starting from the leaf node. Each node on the path to the root should update the visit count and the win count, if the simulation resulted in a win.

Upon returning to the root, we can begin the process anew from Selection.

#### MCTS Example
To begin, we should define how many times we will run MCTS. Let us say we will run MCTS 100 times. You should adjust the number of iterations based on the performance of your code, but you should be able to run thousands of iterations without too much waiting.

First, we should have make node that records no wins or losses, and no visits:

[*Image: Initial MCTS node*]

We can then begin the 4-step process. As we are at the root node with no children, Selection just return us the root as a leaf. We then perform Expansion, we add a child to this leaf by selecting a random legal move from the current board state stored in the root:

[*Image: MCTS tree after expansion*]

Then, we perform Simulation. Let's say the simulation results in a victory for the current player. Then, we perform Backpropagation. Going back starting from the expanded node we performed simulation on, we update the visit and win counts:

[*Image: MCTS tree after backpropagation*]

Upon looping back to Selection once more, we can now apply the UCT formula to select the next node. For each of the yet unvisited nodes, the formula yields infinity, while the visited node yields 1 from the UCT formula. As a result, we will continue the process on one of the unvisited nodes.

### Custom Strategy
Finally, as your last computer AI strategy, propose your own algorithm for selecting the computer's next space. You can either define your own algorithm or research an existing decision-making algorithm with the similar level of complexity as the previous 3 algorithms above.

In order to judge the complexity or efficacy of your algorithm, we will be running a computer vs. computer game with Expectimax vs. your custom algorithm and/or Minimax at a max depth of 2. Your algorithm should win at least 50% of the time against these algorithms in order to get full credit.

Some examples may include some graph-based approaches, using heuristics based on datasets of historically good moves, or probabilistic exploration of tree nodes. However, you should avoid any algorithms that you think would require a lot of extra overhead or significant adjustment to classes like OthelloGame or even the GUI.

## Part 3: Extended Features and Design Patterns
You will need to implement at least 1 optional feature to get full credit on this assignment. You may choose from the list below or propose your own extended feature, as long as it is as significantly complex as the ones below.

- Extra decision making algorithms besides the required 3
- Adjusting the GUI interface, adding animations or extra buttons
- Decision making algorithms that require heavy extra overhead, such as tracking past board states across turns
- Custom Board sizes, like 6x6 or 10x10
- Saving and Loading Board States from files

Additionally, you will need to implement 1 more design pattern other than Strategy within your project. Feel free to adapt your proposed design pattern from your project proposal into the structure of the starter code.

## Part 4: Submission

### Grading Breakdown

#### Core Features (35%)
Your project should include:

- Functional game of Othello with human vs. human, human vs. computer, and computer vs computer modes
- 2 decision making algorithms (1 backtracking, 1 heuristic search) implemented using the Strategy Design Pattern
- 1 custom/alternative algorithm (team-proposed)
- Proper use of the provided packages and file organization
- Able to run your code through the GUI

#### Extended Features (5%)
See Part 3 for more information.

#### Design (10%)
Your program should implement the Strategy Design Pattern and 1 additional design pattern.

#### GitHub (5%)
Use of Git & GitHub for:
- version control purposes (frequent commits; pull requests to implement features; multiple students contributing)
- project planning purposes (project tracking; enumerating features)

#### Testing (30%)
From this category, 25 out of the 30 percentage points are allocated for reaching 80% testing coverage. The remaining 5 percentage points are allocated for qualitative analysis of your tests: do you have many specific tests, or one large test? Are your tests meaningful, or do they just aim to execute as many lines of code as possible? Do you handle edge cases? Do you test both for inputs & outputs as well as state changes?

#### Documentation (5%)
Is your code commented? Do you have a UML diagram that correctly describes all of your class relationships?

#### Presentation (5%)
Do you have all of the required sections?
- Project design
- Initial project plans
- Choices made for optional requirements
- Project demo
- Challenges faced/project retrospective
- TA Q&A

#### Team Assessment (5%)
Points awarded for completing an anonymous Google form. All of the points for this category come from completion of the form, but I (Harry) reserve the right to adjust any group member's overall score as a result of information that teams share about their members.