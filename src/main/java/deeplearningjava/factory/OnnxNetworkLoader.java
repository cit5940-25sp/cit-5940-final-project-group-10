package deeplearningjava.factory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import deeplearningjava.OnnxFileReader;
import deeplearningjava.api.Network;
import deeplearningjava.api.TensorNetwork;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.network.DenseNetwork;
import deeplearningjava.onnx.OnnxModelLoader;

/**
 * Utility class for loading neural networks from ONNX model files.
 * This provides integration between the deep learning framework and
 * pre-trained models saved in ONNX format.
 */
public class OnnxNetworkLoader {
    
    /**
     * Loads a vector-based neural network from an ONNX file.
     * 
     * @param modelPath Path to the ONNX model file
     * @return A Network instance with the architecture and weights extracted from the ONNX file
     * @throws IOException If an I/O error occurs during loading
     */
    public static Network loadNetwork(String modelPath) throws IOException {
        // First try using the advanced ONNX model loader to extract both architecture and weights
        try {
            OnnxModelLoader modelLoader = new OnnxModelLoader(modelPath);
            if (modelLoader.load()) {
                DenseNetwork network = modelLoader.createVectorNetwork();
                if (modelLoader.hasWeights()) {
                    System.out.println("Detected weights in ONNX file, but using network with extracted architecture only: " + modelPath);
                    return network;
                }
            }
        } catch (Exception e) {
            System.out.println("Advanced ONNX loading failed: " + e.getMessage());
            System.out.println("Falling back to basic ONNX parser...");
        }
        
        // Fallback to simple architecture extraction
        OnnxFileReader reader = new OnnxFileReader();
        if (!reader.parse(modelPath)) {
            throw new IOException("Failed to parse ONNX file: " + modelPath);
        }
        
        // Get the layer sizes from the ONNX model
        int[] layerSizes = reader.getLayerSizesArray();
        
        // Create a network with the extracted architecture
        // Note: This doesn't include weights, only the structure
        System.out.println("Created network with structure from ONNX file (no weights)");
        return new DenseNetwork(
            layerSizes,
            ActivationFunctions.relu(), // Common default for hidden layers
            ActivationFunctions.tanh(), // Common for regression outputs
            false // No softmax for regression
        );
    }
    
    /**
     * Loads a tensor-based neural network from an ONNX file.
     * 
     * @param modelPath Path to the ONNX model file
     * @param inputShape The shape of the input tensor [channels, height, width]
     * @return A TensorNetwork instance with the architecture and weights extracted from the ONNX file
     * @throws IOException If an I/O error occurs during loading
     */
    public static TensorNetwork loadTensorNetwork(String modelPath, int[] inputShape) throws IOException {
        // First try using the advanced ONNX model loader
        try {
            OnnxModelLoader modelLoader = new OnnxModelLoader(modelPath);
            if (modelLoader.load()) {
                DenseNetwork network = modelLoader.createTensorNetwork(inputShape);
                if (modelLoader.hasWeights()) {
                    System.out.println("Detected weights in ONNX file, but using tensor network with extracted architecture only: " + modelPath);
                    return network;
                }
            }
        } catch (Exception e) {
            System.out.println("Advanced ONNX loading failed: " + e.getMessage());
            System.out.println("Falling back to basic ONNX parser...");
        }
        
        // Fallback to simple architecture extraction
        OnnxFileReader reader = new OnnxFileReader();
        if (!reader.parse(modelPath)) {
            throw new IOException("Failed to parse ONNX file: " + modelPath);
        }
        
        // Get the layer sizes from the ONNX model
        int[] layerSizes = reader.getLayerSizesArray();
        
        // Extract only the hidden layer sizes (skip input and output)
        int[] hiddenLayerSizes = new int[layerSizes.length - 2];
        if (layerSizes.length > 2) {
            System.arraycopy(layerSizes, 1, hiddenLayerSizes, 0, layerSizes.length - 2);
        } else {
            // Default hidden layer sizes if ONNX doesn't have enough layers
            hiddenLayerSizes = new int[]{128, 64, 32};
        }
        
        // Create a tensor network for the board game
        System.out.println("Created tensor network with structure from ONNX file (no weights)");
        return DenseNetwork.createForBoardGame(
            inputShape,           // Use the provided input shape
            hiddenLayerSizes,     // Use extracted or default hidden layers
            1,                    // Output size for evaluation
            ActivationFunctions.relu(), // Common activation for hidden layers
            ActivationFunctions.tanh(), // Output activation for [-1,1] range
            false                 // No softmax for regression output
        );
    }
    
    /**
     * Loads a tensor-based neural network for Othello from an ONNX file.
     * 
     * @param modelPath Path to the ONNX model file
     * @param boardSize The size of the Othello board (typically 8)
     * @param channels The number of input channels (typically 1 or 3)
     * @return A TensorNetwork instance configured for Othello
     * @throws IOException If an I/O error occurs during loading
     */
    public static TensorNetwork loadOthelloNetwork(String modelPath, int boardSize, int channels) throws IOException {
        // Create input shape array for Othello board
        int[] inputShape = {channels, boardSize, boardSize};
        
        // Load the network using the tensor loader
        return loadTensorNetwork(modelPath, inputShape);
    }
    
    /**
     * Gets the path to the default Othello model file.
     * 
     * @return Path to the default Othello ONNX model
     */
    public static String getDefaultOthelloModelPath() {
        // First try to find the model in the standard location
        String projectDir = System.getProperty("user.dir");
        Path modelPath = Paths.get(projectDir, "models", "othello.onnx");
        
        if (modelPath.toFile().exists()) {
            return modelPath.toString();
        }
        
        // Fallback paths if the model is in other locations
        String[] possiblePaths = {
            Paths.get(projectDir, "othello.onnx").toString(),
            Paths.get(projectDir, "src", "main", "resources", "othello.onnx").toString()
        };
        
        for (String path : possiblePaths) {
            if (Paths.get(path).toFile().exists()) {
                return path;
            }
        }
        
        // Return the standard path even if it doesn't exist yet
        return modelPath.toString();
    }
}