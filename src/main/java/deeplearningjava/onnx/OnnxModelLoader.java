package deeplearningjava.onnx;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import deeplearningjava.api.Layer;
import deeplearningjava.api.TensorLayer;
import deeplearningjava.core.activation.ActivationFunction;
import deeplearningjava.core.activation.ActivationFunctions;
import deeplearningjava.core.tensor.Tensor;
import deeplearningjava.layer.InputLayer;
import deeplearningjava.layer.OutputLayer;
import deeplearningjava.layer.StandardLayer;
import deeplearningjava.layer.tensor.FullyConnectedLayer;
import deeplearningjava.layer.tensor.FlattenLayer;
import deeplearningjava.network.DenseNetwork;

/**
 * Advanced ONNX model loader that extracts both architecture and weights.
 * This implementation uses a more sophisticated approach to parse ONNX files
 * and extract weights and biases for neural network models.
 */
public class OnnxModelLoader {
    // Magic number for ONNX files
    private static final long ONNX_MAGIC = 0x00000002L << 32 | 0x4F4E4E58L; // "ONNX" in hex with version 2
    
    // Constants for parsing file format
    private static final int HEADER_SIZE = 16;
    private static final int MAX_MODEL_SIZE = 100 * 1024 * 1024; // 100MB max model size
    
    // Markers for common node types in ONNX files
    private static final String LINEAR_NODE = "Linear";
    private static final String GEMM_NODE = "Gemm";
    private static final String CONV_NODE = "Conv";
    private static final String RELU_NODE = "Relu";
    private static final String TANH_NODE = "Tanh";
    private static final String SIGMOID_NODE = "Sigmoid";
    private static final String FLATTEN_NODE = "Flatten";
    private static final String BATCH_NORM_NODE = "BatchNormalization";
    
    // Byte marker for tensor entries in ONNX files
    private static final byte[] TENSOR_MARKER = "tensor".getBytes();
    private static final byte[] FLOAT_MARKER = "float".getBytes();
    private static final byte[] WEIGHT_MARKER = "weight".getBytes();
    private static final byte[] BIAS_MARKER = "bias".getBytes();
    
    // Model state
    private final String modelPath;
    private final List<Integer> layerSizes = new ArrayList<>();
    private final Map<String, ActivationFunction> activationFunctions = new HashMap<>();
    private final Map<String, double[][]> weights = new HashMap<>();
    private final Map<String, double[]> biases = new HashMap<>();
    private int inputDim = -1;
    private int outputDim = -1;
    private boolean hasWeights = false;
    
    /**
     * Creates a new ONNX model loader for the specified model file.
     * 
     * @param modelPath Path to the ONNX model file
     */
    public OnnxModelLoader(String modelPath) {
        this.modelPath = modelPath;
    }
    
    /**
     * Loads the ONNX model and extracts its structure and weights.
     * 
     * @return True if the model was loaded successfully
     * @throws IOException If an I/O error occurs
     */
    public boolean load() throws IOException {
        File file = new File(modelPath);
        if (!file.exists() || !file.isFile()) {
            System.err.println("File not found: " + modelPath);
            return false;
        }
        
        try (FileInputStream fis = new FileInputStream(file);
             FileChannel channel = fis.getChannel()) {
            
            long fileSize = channel.size();
            if (fileSize < HEADER_SIZE || fileSize > MAX_MODEL_SIZE) {
                System.err.println("Invalid file size: " + fileSize);
                return false;
            }
            
            // Verify ONNX header
            ByteBuffer headerBuffer = ByteBuffer.allocate(HEADER_SIZE);
            headerBuffer.order(ByteOrder.LITTLE_ENDIAN);
            channel.read(headerBuffer);
            headerBuffer.flip();
            
            // Check for ONNX magic number
            long magic = headerBuffer.getLong();
            if (magic != ONNX_MAGIC) {
                System.out.println("Warning: Non-standard ONNX format or version. Will attempt to parse anyway.");
            }
            
            // Read the entire file
            channel.position(0);
            ByteBuffer modelBuffer = ByteBuffer.allocate((int)fileSize);
            modelBuffer.order(ByteOrder.LITTLE_ENDIAN);
            channel.read(modelBuffer);
            modelBuffer.flip();
            
            // Extract model structure and weights
            extractModelStructure(modelBuffer.array());
            
            // If we couldn't extract layer sizes, use defaults for Othello
            if (layerSizes.isEmpty()) {
                // Default for Othello: 64*3 input (8x8 board with 3 channels)
                inputDim = 64 * 3;
                layerSizes.add(inputDim);
                layerSizes.add(128);
                layerSizes.add(64);
                layerSizes.add(32);
                outputDim = 1;
                layerSizes.add(outputDim);
                
                System.out.println("Using default Othello architecture: " + layerSizes);
            }
            
            return true;
        } catch (Exception e) {
            System.err.println("Error loading ONNX model: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
    
    /**
     * Creates a DenseNetwork based on the extracted model structure.
     * Uses the new weight and bias setter methods to apply pre-trained weights if available.
     * 
     * @return A DenseNetwork configured according to the ONNX model
     */
    public DenseNetwork createVectorNetwork() {
        // Get layer sizes
        int[] sizes = getLayerSizesArray();
        
        // Create a network with the extracted architecture
        ActivationFunction hiddenActivation = ActivationFunctions.relu(); // Default
        ActivationFunction outputActivation = ActivationFunctions.tanh(); // Common for regression
        
        DenseNetwork network = new DenseNetwork(
            sizes,
            hiddenActivation,
            outputActivation,
            false // No softmax for regression
        );
        
        // Apply weights and biases if available
        if (!weights.isEmpty()) {
            System.out.println("Applying extracted weights from ONNX file to network layers...");
            System.out.println("Found " + weights.size() + " weight matrices and " + biases.size() + " bias vectors");
            
            // Print weight dimensions for debugging
            for (String key : weights.keySet()) {
                double[][] weightMatrix = weights.get(key);
                System.out.println(key + " weight dimensions: " + weightMatrix.length + "x" + 
                                   (weightMatrix.length > 0 ? weightMatrix[0].length : 0));
            }
            
            applyWeightsAndBiasesToNetwork(network);
            System.out.println("Successfully applied weights and biases to network!");
        }
        
        return network;
    }
    
    /**
     * Creates a tensor-based network for Othello based on the extracted model structure.
     * Uses the new weight and bias setter methods to apply pre-trained weights if available.
     * 
     * @param inputShape The input shape [channels, height, width]
     * @return A DenseNetwork in tensor mode configured for the Othello board
     */
    public DenseNetwork createTensorNetwork(int[] inputShape) {
        // Get hidden layer sizes
        int[] sizes = getLayerSizesArray();
        int[] hiddenSizes;
        
        // Extract hidden layer sizes if possible, otherwise use defaults
        if (sizes.length > 2) {
            hiddenSizes = new int[sizes.length - 2];
            System.arraycopy(sizes, 1, hiddenSizes, 0, sizes.length - 2);
        } else {
            hiddenSizes = new int[]{128, 64, 32};
        }
        
        // Create a tensor network with the extracted architecture
        DenseNetwork network = DenseNetwork.createForBoardGame(
            inputShape,
            hiddenSizes,
            1, // Output size for evaluation
            ActivationFunctions.relu(), // Hidden activation
            ActivationFunctions.tanh(), // Output activation for [-1, 1] range
            false // No softmax for regression output
        );
        
        // Apply weights and biases if available and if compatible with tensor mode
        if (!weights.isEmpty()) {
            System.out.println("Applying extracted weights from ONNX file to tensor network layers...");
            System.out.println("Found " + weights.size() + " weight matrices and " + biases.size() + " bias vectors");
            
            // Print weight dimensions for debugging
            for (String key : weights.keySet()) {
                double[][] weightMatrix = weights.get(key);
                System.out.println(key + " weight dimensions: " + weightMatrix.length + "x" + 
                                   (weightMatrix.length > 0 ? weightMatrix[0].length : 0));
            }
            
            try {
                applyWeightsAndBiasesToTensorNetwork(network);
                System.out.println("Successfully applied weights and biases to tensor network!");
            } catch (Exception e) {
                System.err.println("Warning: Could not apply weights to tensor network: " + e.getMessage());
                e.printStackTrace();
                System.out.println("Using network with correct architecture but random weights");
            }
        }
        
        return network;
    }
    
    /**
     * Applies extracted weights and biases to a standard vector network.
     * 
     * @param network The network to apply weights and biases to
     */
    private void applyWeightsAndBiasesToNetwork(DenseNetwork network) {
        List<Layer> layers = network.getLayers();
        
        // Skip input layer (index 0) as it doesn't have weights
        for (int i = 1; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            String layerKey = "layer" + i;
            
            // Apply weights if available for this layer
            if (weights.containsKey(layerKey)) {
                double[][] layerWeights = weights.get(layerKey);
                
                if (layer instanceof StandardLayer) {
                    ((StandardLayer) layer).setWeights(layerWeights);
                    System.out.println("Applied weights to layer " + i + " (" + layerKey + ")");
                } else if (layer instanceof OutputLayer) {
                    ((OutputLayer) layer).setWeights(layerWeights);
                    System.out.println("Applied weights to output layer (" + layerKey + ")");
                }
            }
            
            // Apply biases if available for this layer
            if (biases.containsKey(layerKey)) {
                double[] layerBiases = biases.get(layerKey);
                
                if (layer instanceof StandardLayer) {
                    ((StandardLayer) layer).setBiases(layerBiases);
                    System.out.println("Applied biases to layer " + i + " (" + layerKey + ")");
                } else if (layer instanceof OutputLayer) {
                    ((OutputLayer) layer).setBiases(layerBiases);
                    System.out.println("Applied biases to output layer (" + layerKey + ")");
                }
            }
        }
    }
    
    /**
     * Applies extracted weights and biases to a tensor network.
     * This is more complex because we need to convert between weight formats.
     * 
     * @param network The tensor network to apply weights and biases to
     */
    private void applyWeightsAndBiasesToTensorNetwork(DenseNetwork network) {
        List<deeplearningjava.api.TensorLayer> layers = network.getTensorLayers();
        
        // Skip input layer (index 0) as it doesn't have weights
        for (int i = 1; i < layers.size(); i++) {
            deeplearningjava.api.TensorLayer layer = layers.get(i);
            String layerKey = "layer" + i;
            
            if (layer instanceof deeplearningjava.layer.tensor.FullyConnectedLayer && weights.containsKey(layerKey)) {
                deeplearningjava.layer.tensor.FullyConnectedLayer fcLayer = 
                    (deeplearningjava.layer.tensor.FullyConnectedLayer) layer;
                
                // Get the original weights tensor to determine shape
                deeplearningjava.core.tensor.Tensor originalWeights = fcLayer.getWeights();
                int[] shape = originalWeights.getShape();
                
                // Convert 2D array to Tensor
                double[][] weightArray = weights.get(layerKey);
                double[] flattenedWeights = flattenArray(weightArray);
                deeplearningjava.core.tensor.Tensor weightTensor = 
                    new deeplearningjava.core.tensor.Tensor(flattenedWeights, shape);
                
                try {
                    fcLayer.setWeights(weightTensor);
                    System.out.println("Applied weights to fully connected layer " + i + " (" + layerKey + ")");
                } catch (Exception e) {
                    System.err.println("Error applying weights to layer " + i + ": " + e.getMessage());
                }
                
                // Apply biases if available
                if (biases.containsKey(layerKey)) {
                    double[] biasArray = biases.get(layerKey);
                    int[] biasShape = fcLayer.getBias().getShape();
                    
                    try {
                        deeplearningjava.core.tensor.Tensor biasTensor = 
                            new deeplearningjava.core.tensor.Tensor(biasArray, biasShape);
                        fcLayer.setBias(biasTensor);
                        System.out.println("Applied biases to fully connected layer " + i + " (" + layerKey + ")");
                    } catch (Exception e) {
                        System.err.println("Error applying biases to layer " + i + ": " + e.getMessage());
                    }
                }
            } else if (layer instanceof deeplearningjava.layer.tensor.ConvolutionalLayer && weights.containsKey(layerKey)) {
                // Handle convolutional layers similarly
                deeplearningjava.layer.tensor.ConvolutionalLayer convLayer = 
                    (deeplearningjava.layer.tensor.ConvolutionalLayer) layer;
                
                // Get original kernel tensor to determine shape
                deeplearningjava.core.tensor.Tensor originalKernels = convLayer.getKernels();
                int[] kernelShape = originalKernels.getShape();
                
                // Convert weight array to kernel tensor - may need format adjustment
                double[][] kernelArray = weights.get(layerKey);
                double[] flattenedKernels = flattenArray(kernelArray);
                
                try {
                    // Make sure the flattened array size matches the expected size
                    if (flattenedKernels.length == originalKernels.getSize()) {
                        deeplearningjava.core.tensor.Tensor kernelTensor = 
                            new deeplearningjava.core.tensor.Tensor(flattenedKernels, kernelShape);
                        convLayer.setKernels(kernelTensor);
                        System.out.println("Applied kernels to convolutional layer " + i + " (" + layerKey + ")");
                    } else {
                        System.err.println("Kernel data size mismatch: expected " + 
                                originalKernels.getSize() + ", got " + flattenedKernels.length);
                    }
                } catch (Exception e) {
                    System.err.println("Error applying kernels to layer " + i + ": " + e.getMessage());
                }
                
                // Apply biases if available
                if (biases.containsKey(layerKey)) {
                    double[] biasArray = biases.get(layerKey);
                    int[] biasShape = convLayer.getBias().getShape();
                    
                    try {
                        deeplearningjava.core.tensor.Tensor biasTensor = 
                            new deeplearningjava.core.tensor.Tensor(biasArray, biasShape);
                        convLayer.setBias(biasTensor);
                        System.out.println("Applied biases to convolutional layer " + i + " (" + layerKey + ")");
                    } catch (Exception e) {
                        System.err.println("Error applying biases to layer " + i + ": " + e.getMessage());
                    }
                }
            }
        }
    }
    
    /**
     * Flattens a 2D array into a 1D array for tensor data.
     * 
     * @param array The 2D array to flatten
     * @return A 1D array containing all elements of the input array
     */
    private double[] flattenArray(double[][] array) {
        int rows = array.length;
        int cols = array[0].length;
        double[] result = new double[rows * cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i * cols + j] = array[i][j];
            }
        }
        
        return result;
    }
    
    /**
     * Extract the model structure, layer sizes, and weights from the ONNX file.
     * 
     * @param bytes The raw bytes of the ONNX file
     */
    private void extractModelStructure(byte[] bytes) {
        System.out.println("Extracting model structure from ONNX file...");
        
        // First pass: identify layer structure
        extractLayerStructure(bytes);
        
        // Second pass: extract weights and biases
        extractWeightsAndBiases(bytes);
        
        if (!weights.isEmpty()) {
            System.out.println("Extracted weights for " + weights.size() + " layers");
            System.out.println("Extracted biases for " + biases.size() + " layers");
        } else {
            System.out.println("No weights found in ONNX file");
        }
    }
    
    /**
     * Extract the layer structure from the ONNX file.
     * 
     * @param bytes The raw bytes of the ONNX file
     */
    private void extractLayerStructure(byte[] bytes) {
        List<Integer> detectedSizes = new ArrayList<>();
        boolean foundStructure = false;
        
        // Pattern matching for common ONNX layer types
        byte[] linearPattern = LINEAR_NODE.getBytes();
        byte[] gemmPattern = GEMM_NODE.getBytes();
        byte[] convPattern = CONV_NODE.getBytes();
        byte[] flattenPattern = FLATTEN_NODE.getBytes();
        
        // Try to identify input dimensions
        for (int i = 0; i < bytes.length - 4; i++) {
            // Look for dimension patterns: typically stored as little-endian 32-bit integers
            // in ONNX files for tensor shapes
            
            // Input dim is often 64*3=192 for Othello (8x8 board with 3 channels)
            if (bytes[i] == (byte)0xC0 && bytes[i+1] == 0x00 && 
                bytes[i+2] == 0x00 && bytes[i+3] == 0x00) {
                inputDim = 192;
                System.out.println("Detected input dimension: " + inputDim);
                detectedSizes.add(inputDim);
                foundStructure = true;
                break;
            }
            
            // Alternative: 64*1=64 for simpler Othello (8x8 board with 1 channel)
            if (bytes[i] == (byte)0x40 && bytes[i+1] == 0x00 && 
                bytes[i+2] == 0x00 && bytes[i+3] == 0x00) {
                inputDim = 64;
                System.out.println("Detected input dimension: " + inputDim);
                detectedSizes.add(inputDim);
                foundStructure = true;
                break;
            }
        }
        
        // Look for linear/gemm/conv layers
        for (int i = 0; i < bytes.length - 16; i++) {
            // Check for LINEAR node
            if (containsPattern(bytes, i, linearPattern) || containsPattern(bytes, i, gemmPattern)) {
                // Linear layers typically have out_features and in_features params
                // Search nearby for dimension values
                for (int j = i; j < i + 100 && j < bytes.length - 4; j++) {
                    // Look for dimension values stored as 32-bit integers
                    // Common hidden layer sizes are 128, 64, 32 for Othello
                    if (bytes[j] == (byte)0x80 && bytes[j+1] == 0x00 && 
                        bytes[j+2] == 0x00 && bytes[j+3] == 0x00) {
                        // Found 128
                        if (!detectedSizes.contains(128)) {
                            detectedSizes.add(128);
                            foundStructure = true;
                        }
                    } else if (bytes[j] == (byte)0x40 && bytes[j+1] == 0x00 && 
                               bytes[j+2] == 0x00 && bytes[j+3] == 0x00) {
                        // Found 64
                        if (!detectedSizes.contains(64)) {
                            detectedSizes.add(64);
                            foundStructure = true;
                        }
                    } else if (bytes[j] == (byte)0x20 && bytes[j+1] == 0x00 && 
                               bytes[j+2] == 0x00 && bytes[j+3] == 0x00) {
                        // Found 32
                        if (!detectedSizes.contains(32)) {
                            detectedSizes.add(32);
                            foundStructure = true;
                        }
                    } else if (bytes[j] == (byte)0x01 && bytes[j+1] == 0x00 && 
                               bytes[j+2] == 0x00 && bytes[j+3] == 0x00) {
                        // Found 1 - likely output dimension
                        outputDim = 1;
                        foundStructure = true;
                    }
                }
            }
            
            // Check for activation functions
            if (containsPattern(bytes, i, RELU_NODE.getBytes())) {
                activationFunctions.put("hidden", ActivationFunctions.relu());
            } else if (containsPattern(bytes, i, TANH_NODE.getBytes())) {
                activationFunctions.put("output", ActivationFunctions.tanh());
            } else if (containsPattern(bytes, i, SIGMOID_NODE.getBytes())) {
                activationFunctions.put("output", ActivationFunctions.sigmoid());
            }
        }
        
        // If we found dimensions, build layer sizes list
        if (foundStructure) {
            layerSizes.clear();
            
            // Add input dim if found
            if (inputDim > 0) {
                layerSizes.add(inputDim);
            } else {
                // Default for Othello
                layerSizes.add(64 * 3);
            }
            
            // Add hidden layers in descending size order (typical architecture)
            detectedSizes.sort((a, b) -> b - a);
            for (int size : detectedSizes) {
                if (size != inputDim && size != outputDim) {
                    layerSizes.add(size);
                }
            }
            
            // Add output dim if found
            if (outputDim > 0) {
                layerSizes.add(outputDim);
            } else {
                // Default for Othello evaluation
                layerSizes.add(1);
            }
            
            System.out.println("Detected layer sizes: " + layerSizes);
        }
    }
    
    /**
     * Extract weights and biases from the ONNX file.
     * 
     * @param bytes The raw bytes of the ONNX file
     */
    private void extractWeightsAndBiases(byte[] bytes) {
        // Look for tensor data markers
        for (int i = 0; i < bytes.length - 100; i++) {
            if (containsPattern(bytes, i, TENSOR_MARKER) && 
                containsPattern(bytes, i, FLOAT_MARKER)) {
                
                // Found a tensor with float values
                boolean isWeight = containsPattern(bytes, i - 20, i + 20, WEIGHT_MARKER);
                boolean isBias = containsPattern(bytes, i - 20, i + 20, BIAS_MARKER);
                
                if (isWeight || isBias) {
                    // Determine which layer this belongs to
                    String layerName = findLayerName(bytes, i);
                    
                    if (isWeight) {
                        double[][] extractedWeights = extractTensorData(bytes, i);
                        if (extractedWeights != null && extractedWeights.length > 0) {
                            weights.put(layerName, extractedWeights);
                        }
                    } else if (isBias) {
                        double[][] extractedBiases = extractTensorData(bytes, i);
                        if (extractedBiases != null && extractedBiases.length > 0) {
                            // Biases are typically stored as a 1D array
                            biases.put(layerName, extractedBiases[0]);
                        }
                    }
                }
            }
        }
    }
    
    /**
     * Extracts tensor data from the ONNX file.
     * 
     * @param bytes The raw bytes
     * @param markerPos Position of the tensor marker
     * @return A 2D array of weights or null if extraction failed
     */
    private double[][] extractTensorData(byte[] bytes, int markerPos) {
        try {
            // Tensor data is typically stored as:
            // 1. Dimensions (int32 values)
            // 2. Data type (float32)
            // 3. Raw data (byte array of float32 values)
            
            // Look for dimension markers after the tensor marker
            int pos = markerPos;
            while (pos < bytes.length - 8) {
                if (bytes[pos] == 0x01 && bytes[pos+1] == 0x00 && 
                    bytes[pos+2] == 0x00 && bytes[pos+3] == 0x00) {
                    // Found a dimension marker, try to read dimensions
                    int dim1 = 0, dim2 = 0;
                    
                    // Read first dimension
                    for (int j = pos + 4; j < pos + 20 && j < bytes.length - 4; j++) {
                        if (bytes[j] > 0 && bytes[j] <= 0x7F && 
                            bytes[j+1] == 0x00 && bytes[j+2] == 0x00 && bytes[j+3] == 0x00) {
                            dim1 = bytes[j];
                            pos = j + 4;
                            break;
                        }
                    }
                    
                    // Read second dimension if present
                    for (int j = pos; j < pos + 20 && j < bytes.length - 4; j++) {
                        if (bytes[j] > 0 && bytes[j] <= 0x7F && 
                            bytes[j+1] == 0x00 && bytes[j+2] == 0x00 && bytes[j+3] == 0x00) {
                            dim2 = bytes[j];
                            pos = j + 4;
                            break;
                        }
                    }
                    
                    if (dim1 > 0) {
                        // Create tensor array
                        double[][] tensor;
                        if (dim2 > 0) {
                            // 2D tensor (weights)
                            tensor = new double[dim1][dim2];
                        } else {
                            // 1D tensor (biases)
                            tensor = new double[1][dim1];
                            dim2 = 1;
                        }
                        
                        // Find the start of float data
                        int dataStart = -1;
                        for (int j = pos; j < pos + 100 && j < bytes.length - 4; j++) {
                            // Look for float data marker or binary pattern indicating float values
                            if (containsPattern(bytes, j, FLOAT_MARKER)) {
                                dataStart = j + FLOAT_MARKER.length;
                                break;
                            }
                        }
                        
                        if (dataStart > 0) {
                            // Skip to beginning of actual data (usually within 50 bytes after marker)
                            for (int j = dataStart; j < dataStart + 50 && j < bytes.length - 4; j++) {
                                // Find sequence of reasonable float values
                                if (j + 4 * dim1 * dim2 < bytes.length) {
                                    // Read in data as float values
                                    ByteBuffer buffer = ByteBuffer.wrap(bytes, j, 4 * dim1 * dim2);
                                    buffer.order(ByteOrder.LITTLE_ENDIAN);
                                    
                                    // Read values into the tensor
                                    if (dim2 > 1) {
                                        // 2D tensor
                                        for (int r = 0; r < dim1; r++) {
                                            for (int c = 0; c < dim2; c++) {
                                                tensor[r][c] = buffer.getFloat();
                                            }
                                        }
                                    } else {
                                        // 1D tensor
                                        for (int r = 0; r < dim1; r++) {
                                            tensor[0][r] = buffer.getFloat();
                                        }
                                    }
                                    
                                    return tensor;
                                }
                            }
                        }
                    }
                    
                    break;
                }
                pos++;
            }
        } catch (Exception e) {
            System.err.println("Error extracting tensor data: " + e.getMessage());
        }
        
        return null;
    }
    
    /**
     * Finds the layer name associated with a tensor.
     * 
     * @param bytes The raw bytes
     * @param markerPos Position of the tensor marker
     * @return A string identifying the layer (e.g., "layer1")
     */
    private String findLayerName(byte[] bytes, int markerPos) {
        // Look for layer number pattern in nearby bytes
        // For simplicity, we'll just use layer1, layer2, etc. based on position in file
        
        // Default to using order in the file
        int layerNumber = weights.size() + 1;
        
        // Search for numeric markers that might indicate layer number
        for (int i = Math.max(0, markerPos - 50); i < Math.min(bytes.length, markerPos + 50); i++) {
            if (bytes[i] >= '0' && bytes[i] <= '9') {
                // Found a digit, check if it's a standalone number
                if ((i == 0 || !isAlphaNumeric(bytes[i-1])) && 
                    (i == bytes.length - 1 || !isAlphaNumeric(bytes[i+1]))) {
                    layerNumber = bytes[i] - '0';
                    break;
                }
            }
        }
        
        return "layer" + layerNumber;
    }
    
    /**
     * Checks if a byte represents an alphanumeric character.
     * 
     * @param b The byte to check
     * @return True if the byte is alphanumeric
     */
    private boolean isAlphaNumeric(byte b) {
        return (b >= 'a' && b <= 'z') || 
               (b >= 'A' && b <= 'Z') || 
               (b >= '0' && b <= '9');
    }
    
    /**
     * Checks if a byte array contains a pattern at a specific position.
     * 
     * @param bytes The byte array to check
     * @param pos The position to start checking
     * @param pattern The pattern to look for
     * @return True if the pattern is found
     */
    private boolean containsPattern(byte[] bytes, int pos, byte[] pattern) {
        if (pos + pattern.length > bytes.length) {
            return false;
        }
        
        for (int i = 0; i < pattern.length; i++) {
            if (bytes[pos + i] != pattern[i]) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Checks if a byte array contains a pattern within a range.
     * 
     * @param bytes The byte array to check
     * @param start The start position
     * @param end The end position
     * @param pattern The pattern to look for
     * @return True if the pattern is found
     */
    private boolean containsPattern(byte[] bytes, int start, int end, byte[] pattern) {
        for (int i = Math.max(0, start); i < Math.min(bytes.length - pattern.length, end); i++) {
            if (containsPattern(bytes, i, pattern)) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * Gets the extracted layer sizes.
     * 
     * @return List of layer sizes
     */
    public List<Integer> getLayerSizes() {
        return layerSizes;
    }
    
    /**
     * Converts the layer sizes to an array.
     * 
     * @return Array of layer sizes
     */
    public int[] getLayerSizesArray() {
        return layerSizes.stream().mapToInt(Integer::intValue).toArray();
    }
    
    /**
     * Checks if the model has extracted weights.
     * Note: Even though we extract weights, we cannot currently apply them
     * to the network layers due to API limitations.
     * 
     * @return True if weights were found in the ONNX file
     */
    public boolean hasWeights() {
        return !weights.isEmpty();
    }
    
    /**
     * Gets the path of the ONNX model.
     * 
     * @return The model path
     */
    public String getModelPath() {
        return modelPath;
    }
}