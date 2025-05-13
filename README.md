# Repo Layout

Below is a high level summary of how we organized our project code

We used Claude and ChatGPT to help write the deep learning code.  


## UML Diagram

```
── Othello Classes.png
```
## Data and scripts used in neural net training
```
├── learning_data
│   ├── othello_dataset.csv
│   ├── othello_training.ipynb
│   └── preprocess_parallel.py
```
## Pretrained network
```
├── models
│   └── othello.onnx
```
## Deep Learning Classes
```├── src
│   ├── main
│   │   ├── java
│   │   │   ├── deeplearningjava
│   │   │   │   ├── OnnxFileReader.java
│   │   │   │   ├── api
│   │   │   │   │   ├── BaseNetwork.java
│   │   │   │   │   ├── ConvolutionalNetwork.java
│   │   │   │   │   ├── Layer.java
│   │   │   │   │   ├── Network.java
│   │   │   │   │   ├── Serializable.java
│   │   │   │   │   ├── TensorLayer.java
│   │   │   │   │   ├── TensorNetwork.java
│   │   │   │   │   └── Trainable.java
│   │   │   │   ├── core
│   │   │   │   │   ├── Edge.java
│   │   │   │   │   ├── Node.java
│   │   │   │   │   ├── activation
│   │   │   │   │   │   ├── ActivationFunction.java
│   │   │   │   │   │   ├── ActivationFunctions.java
│   │   │   │   │   │   ├── LeakyReLU.java
│   │   │   │   │   │   ├── Linear.java
│   │   │   │   │   │   ├── ReLU.java
│   │   │   │   │   │   ├── Sigmoid.java
│   │   │   │   │   │   └── Tanh.java
│   │   │   │   │   └── tensor
│   │   │   │   │       ├── Tensor.java
│   │   │   │   │       └── TensorOperations.java
│   │   │   │   ├── factory
│   │   │   │   │   ├── NetworkFactory.java
│   │   │   │   │   └── OnnxNetworkLoader.java
│   │   │   │   ├── layer
│   │   │   │   │   ├── AbstractLayer.java
│   │   │   │   │   ├── BatchNormLayer.java
│   │   │   │   │   ├── InputLayer.java
│   │   │   │   │   ├── Layer.java
│   │   │   │   │   ├── LayerBridge.java
│   │   │   │   │   ├── LayerType.java
│   │   │   │   │   ├── OutputLayer.java
│   │   │   │   │   ├── StandardLayer.java
│   │   │   │   │   └── tensor
│   │   │   │   │       ├── AbstractTensorLayer.java
│   │   │   │   │       ├── ConvolutionalLayer.java
│   │   │   │   │       ├── FlattenLayer.java
│   │   │   │   │       ├── FullyConnectedLayer.java
│   │   │   │   │       └── PoolingLayer.java
│   │   │   │   ├── network
│   │   │   │   │   ├── AbstractNetwork.java
│   │   │   │   │   ├── ConvolutionalNeuralNetwork.java
│   │   │   │   │   ├── DenseNetwork.java
│   │   │   │   │   ├── FeedForwardNetwork.java
│   │   │   │   │   ├── NeuralNetwork.java
│   │   │   │   │   └── optimizer
│   │   │   │   │       ├── Optimizer.java
│   │   │   │   │       └── SGD.java
│   │   │   │   └── onnx
│   │   │   │       └── OnnxModelLoader.java
```

## Graph Classes
```
│   │   │   ├── graph
│   │   │   │   ├── core
│   │   │   │   │   └── TreeNode.java
│   │   │   │   ├── search
│   │   │   │   │   ├── Expectimax.java
│   │   │   │   │   ├── GameTreeNode.java
│   │   │   │   │   ├── MiniMax.java
│   │   │   │   │   └── MonteCarloTreeSearch.java
```
## Othello Game Classes
```
│   │   │   ├── othello
│   │   │   │   ├── App.java
│   │   │   │   ├── Constants.java
│   │   │   │   ├── gamelogic
│   │   │   │   │   ├── BoardSpace.java
│   │   │   │   │   ├── ComputerPlayer.java
│   │   │   │   │   ├── GameState.java
│   │   │   │   │   ├── HumanPlayer.java
│   │   │   │   │   ├── OthelloGame.java
│   │   │   │   │   ├── Player.java
```
## Strategies
```
│   │   │   │   │   └── strategies
│   │   │   │   │       ├── BoardEvaluator.java
│   │   │   │   │       ├── BoardToInputMapper.java
│   │   │   │   │       ├── ExpectimaxStrategy.java
│   │   │   │   │       ├── MCTSStrategy.java
│   │   │   │   │       ├── MinimaxStrategy.java
│   │   │   │   │       ├── NetworkWrapper.java
│   │   │   │   │       ├── NeuralStrategy.java
│   │   │   │   │       ├── Strategy.java
│   │   │   │   │       ├── StrategyFactory.java
│   │   │   │   │       ├── TensorNetworkWrapper.java
│   │   │   │   │       └── WeightedEvaluator.java
```
## GUI
``` 
│   │   │   │   ├── gui
│   │   │   │   │   ├── GUISpace.java
│   │   │   │   │   └── GameController.java
```
## Misc. Extra Resources
```
│   │   │   │   └── models
│   │   │   │       ├── othello_1d_2_hidden_layer_dense_model.onnx
│   │   │   │       ├── othello_2d_2_hidden_layer_dense_model.onnx
│   │   │   │       └── othello_cnn.onnx
│   │   ├── python
│   │   │   ├── othello_dataset.csv
│   │   │   └── preprocess_parallel.py
│   │   └── resources
│   │       ├── data
│   │       │   └── othello_dataset.csv
│   │       └── othello
│   │           └── game-view.fxml
```
## Testing
```
│   └── test
│       └── java
│           ├── deeplearningjava
│           │   ├── DeepLearningIntegrationTest.java
│           │   ├── EdgeTest.java
│           │   ├── OnnxModelLoadTest.java
│           │   ├── OnnxModelLoaderTest.java
│           │   ├── core
│           │   │   └── activation
│           │   │       └── ActivationFunctionTest.java
│           │   ├── layer
│           │   │   └── LayerWeightTest.java
│           │   └── network
│           │       ├── AbstractNetworkTest.java
│           │       ├── ConvolutionalNetworkTest.java
│           │       ├── ConvolutionalNeuralNetworkTest.java
│           │       ├── DenseNetworkTensorTest.java
│           │       ├── DenseNetworkTest.java
│           │       └── FeedForwardNetworkTest.java
│           └── othello
│               └── gamelogic
│                   └── strategies
│                       ├── ExpectimaxStrategyTest.java
│                       ├── MCTSStrategyTest.java
│                       ├── MinimaxStrategyTest.java
│                       ├── NetworkWrapperTest.java
│                       ├── NeuralStrategyTest.java
│                       ├── OnnxStrategyTest.java
│                       ├── TensorNetworkWrapperTest.java
│                       └── TestUtils.java
```
