package deeplearningjava.onnx;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import deeplearningjava.api.Network;
import deeplearningjava.core.tensor.Tensor;
import deeplearningjava.factory.OnnxNetworkLoader;
import deeplearningjava.network.DenseNetwork;

/**
 * Advanced tests for OnnxModelLoader that improve code coverage.
 */
public class OnnxModelLoaderAdvancedTest {
    
    private OnnxModelLoader modelLoader;
    private String modelPath;
    
    @TempDir
    Path tempDir;
    
    @BeforeEach
    public void setUp() {
        // Get path to default ONNX model
        modelPath = OnnxNetworkLoader.getDefaultOthelloModelPath();
        
        // Initialize the loader but don't load yet
        modelLoader = new OnnxModelLoader(modelPath);
    }
    
    @Test
    public void testModelPathAccessor() {
        // Test the getModelPath accessor
        assertEquals(modelPath, modelLoader.getModelPath());
    }
    
    @Test
    public void testTensorNetworkWithDifferentInputShapes() throws IOException {
        // Load the model
        modelLoader.load();
        
        // Test with different input shapes
        int[] smallerShape = {1, 4, 4};
        DenseNetwork smallerNetwork = modelLoader.createTensorNetwork(smallerShape);
        assertNotNull(smallerNetwork, "Network should be created with smaller input shape");
        
        int[] largerShape = {2, 8, 8};
        DenseNetwork largerNetwork = modelLoader.createTensorNetwork(largerShape);
        assertNotNull(largerNetwork, "Network should be created with larger input shape");
        
        // Verify first layer accepts correct input shape
        assertEquals(smallerShape.length, smallerNetwork.getTensorLayers().get(0).getInputShape().length);
        assertEquals(largerShape.length, largerNetwork.getTensorLayers().get(0).getInputShape().length);
    }
    
    @Test
    public void testLayerSizesMethods() throws IOException {
        // Load the model
        modelLoader.load();
        
        // Test both List and array getters for layer sizes
        List<Integer> layerSizesList = modelLoader.getLayerSizes();
        int[] layerSizesArray = modelLoader.getLayerSizesArray();
        
        // Verify both methods return the same data
        assertNotNull(layerSizesList);
        assertNotNull(layerSizesArray);
        assertEquals(layerSizesList.size(), layerSizesArray.length);
        
        for (int i = 0; i < layerSizesList.size(); i++) {
            assertEquals(layerSizesList.get(i).intValue(), layerSizesArray[i]);
        }
    }
    
    @Test
    public void testNetworkPrediction() throws IOException {
        // Load the model
        modelLoader.load();
        
        // Create vector network
        DenseNetwork network = modelLoader.createVectorNetwork();
        
        // Test prediction with random input
        double[] input = new double[network.getInputSize()];
        Arrays.fill(input, 0.5); // Fill with test values
        
        // This should not throw exceptions
        double[] output = network.forward(input);
        
        // Output should be of the expected size (typically 1)
        assertEquals(network.getOutputSize(), output.length);
    }
    
    @Test
    public void testTensorNetworkPrediction() throws IOException {
        // Load the model
        modelLoader.load();
        
        // Create tensor network
        int[] inputShape = {1, 8, 8};
        DenseNetwork network = modelLoader.createTensorNetwork(inputShape);
        
        // Create input tensor
        Tensor inputTensor = new Tensor(inputShape);
        for (int i = 0; i < inputTensor.getSize(); i++) {
            inputTensor.getData()[i] = 0.5; // Fill with test values
        }
        
        // This should not throw exceptions
        Tensor output = network.forward(inputTensor);
        
        // Output should have expected shape
        int[] expectedOutputShape = network.getTensorLayers().get(network.getTensorLayers().size() - 1).getOutputShape();
        assertArrayEquals(expectedOutputShape, output.getShape());
    }
    
    @Test
    public void testLoadingNonExistentFile() throws IOException {
        // The OnnxModelLoader class's behavior for non-existent files seems to be returning false
        // rather than using a fallback, so we'll test this correctly
        OnnxModelLoader badLoader = new OnnxModelLoader("/path/to/nonexistent/model.onnx");
        
        try {
            // Attempt to load the non-existent file
            boolean result = badLoader.load();
            
            // Regardless of the result, the operation should not throw an exception
            // If result is true, verify layer sizes
            if (result) {
                int[] sizes = badLoader.getLayerSizesArray();
                assertTrue(sizes.length > 0, "Should have layer sizes if load returned true");
            }
        } catch (Exception e) {
            fail("Loading a non-existent file should not throw an exception, but gracefully handle the error");
        }
    }
    
    @Test
    public void testCreateMockOnnxFile() throws IOException {
        // Create a minimal mock ONNX file with the ONNX magic number and some patterns to test parsing
        File mockFile = tempDir.resolve("mock.onnx").toFile();
        
        try (FileOutputStream out = new FileOutputStream(mockFile)) {
            ByteBuffer buffer = ByteBuffer.allocate(1024);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            
            // ONNX magic number (from OnnxModelLoader.ONNX_MAGIC)
            buffer.putLong(0x00000002L << 32 | 0x4F4E4E58L);
            
            // Add some model header bytes
            buffer.put("ONNX_BINARY".getBytes());
            
            // Add some patterns that the parser might recognize
            buffer.put("Linear".getBytes());
            buffer.put("Relu".getBytes());
            
            // Add dimension markers (192 = 0xC0 in little endian)
            buffer.putInt(0xC0);
            
            // Add a tensor marker
            buffer.put("tensor".getBytes());
            buffer.put("float".getBytes());
            
            // Finalize the buffer
            buffer.flip();
            out.write(buffer.array(), 0, buffer.limit());
        }
        
        // Create a loader for this mock file
        OnnxModelLoader mockLoader = new OnnxModelLoader(mockFile.getAbsolutePath());
        boolean loaded = mockLoader.load();
        
        // Should load successfully
        assertTrue(loaded, "Mock ONNX file should load");
        
        // Should be able to create networks
        DenseNetwork vectorNetwork = mockLoader.createVectorNetwork();
        assertNotNull(vectorNetwork, "Should create vector network from mock file");
        
        DenseNetwork tensorNetwork = mockLoader.createTensorNetwork(new int[]{1, 8, 8});
        assertNotNull(tensorNetwork, "Should create tensor network from mock file");
    }
    
    @Test
    public void testContainsPatternDifferentMethods() throws Exception {
        // Use reflection to access private method for coverage
        java.lang.reflect.Method containsPattern = OnnxModelLoader.class.getDeclaredMethod(
            "containsPattern", byte[].class, int.class, byte[].class);
        containsPattern.setAccessible(true);
        
        java.lang.reflect.Method containsPatternRange = OnnxModelLoader.class.getDeclaredMethod(
            "containsPattern", byte[].class, int.class, int.class, byte[].class);
        containsPatternRange.setAccessible(true);
        
        // Test data
        byte[] bytes = "test pattern string".getBytes();
        byte[] pattern = "pattern".getBytes();
        
        // Test containsPattern
        boolean result1 = (boolean) containsPattern.invoke(modelLoader, bytes, 5, pattern);
        assertTrue(result1, "Should find pattern at position 5");
        
        boolean result2 = (boolean) containsPattern.invoke(modelLoader, bytes, 0, pattern);
        assertFalse(result2, "Should not find pattern at position 0");
        
        // Test containsPatternRange
        boolean result3 = (boolean) containsPatternRange.invoke(modelLoader, bytes, 0, bytes.length, pattern);
        assertTrue(result3, "Should find pattern in range");
        
        boolean result4 = (boolean) containsPatternRange.invoke(modelLoader, bytes, 0, 4, pattern);
        assertFalse(result4, "Should not find pattern in limited range");
    }
    
    @Test
    public void testIsAlphaNumeric() throws Exception {
        // Use reflection to access private method for coverage
        java.lang.reflect.Method isAlphaNumeric = OnnxModelLoader.class.getDeclaredMethod(
            "isAlphaNumeric", byte.class);
        isAlphaNumeric.setAccessible(true);
        
        // Test with different character types
        assertTrue((boolean) isAlphaNumeric.invoke(modelLoader, (byte) 'a'));
        assertTrue((boolean) isAlphaNumeric.invoke(modelLoader, (byte) 'Z'));
        assertTrue((boolean) isAlphaNumeric.invoke(modelLoader, (byte) '9'));
        assertFalse((boolean) isAlphaNumeric.invoke(modelLoader, (byte) ' '));
        assertFalse((boolean) isAlphaNumeric.invoke(modelLoader, (byte) '.'));
    }
    
    @Test
    public void testFlattenArray() throws Exception {
        // Use reflection to access private method for coverage
        java.lang.reflect.Method flattenArray = OnnxModelLoader.class.getDeclaredMethod(
            "flattenArray", double[][].class);
        flattenArray.setAccessible(true);
        
        // Test data
        double[][] array = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}
        };
        
        double[] expected = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        
        // Run the flattening
        double[] result = (double[]) flattenArray.invoke(modelLoader, (Object) array);
        
        // Verify result
        assertArrayEquals(expected, result, 0.0001);
    }
}