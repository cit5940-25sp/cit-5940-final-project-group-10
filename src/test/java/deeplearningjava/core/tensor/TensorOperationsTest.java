package deeplearningjava.core.tensor;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.Arrays;

/**
 * Tests for the TensorOperations utility class.
 */
public class TensorOperationsTest {
    
    private Tensor inputTensor;
    private Tensor kernelTensor;
    
    @BeforeEach
    public void setUp() {
        // Create a 4D input tensor [1, 1, 4, 4] (batch_size=1, channels=1, height=4, width=4)
        double[] inputData = {
            // Channel 0
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        };
        inputTensor = new Tensor(inputData, 1, 1, 4, 4);
        
        // Create a 4D kernel tensor [2, 1, 3, 3] (out_channels=2, in_channels=1, height=3, width=3)
        double[] kernelData = {
            // Out channel 0, in channel 0
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
            
            // Out channel 1, in channel 0
            0, 1, 0,
            1, -4, 1,
            0, 1, 0
        };
        kernelTensor = new Tensor(kernelData, 2, 1, 3, 3);
    }
    
    @Test
    public void testConvolve() {
        // Test convolution with padding
        Tensor output = TensorOperations.convolve(inputTensor, kernelTensor, new int[]{1, 1}, true);
        
        // Output shape should be [1, 2, 4, 4] (batch_size=1, out_channels=2, height=4, width=4)
        assertArrayEquals(new int[]{1, 2, 4, 4}, output.getShape());
        
        // Verify a few values from the first output channel (sum filter)
        assertEquals(sum3x3Region(inputTensor, 0, 0, 0, 0, 1, 1), output.get(0, 0, 0, 0), 0.001);
        assertEquals(sum3x3Region(inputTensor, 0, 0, 1, 1, 1, 1), output.get(0, 0, 1, 1), 0.001);
        
        // Verify a few values from the second output channel (edge detection filter)
        // Edge detection should give high values at edges and low values in uniform regions
        double centerValue = output.get(0, 1, 1, 1);
        double edgeValue = output.get(0, 1, 0, 1);
        assertTrue(Math.abs(edgeValue) > Math.abs(centerValue), 
                "Edge detection should highlight edges");
    }
    
    @Test
    public void testConvolveWithoutPadding() {
        // Test convolution without padding
        Tensor output = TensorOperations.convolve(inputTensor, kernelTensor, new int[]{1, 1}, false);
        
        // Output shape should be [1, 2, 2, 2] (batch_size=1, out_channels=2, height=2, width=2)
        assertArrayEquals(new int[]{1, 2, 2, 2}, output.getShape());
        
        // Verify center values (only valid convolutions without padding)
        assertEquals(sum3x3Region(inputTensor, 0, 0, 0, 0, 0, 0), output.get(0, 0, 0, 0), 0.001);
        assertEquals(sum3x3Region(inputTensor, 0, 0, 0, 1, 0, 0), output.get(0, 0, 0, 1), 0.001);
    }
    
    @Test
    public void testConvolveInvalidInput() {
        // Test with invalid input shapes
        Tensor invalidInput = new Tensor(new double[]{1, 2, 3, 4}, 2, 2);
        
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            TensorOperations.convolve(invalidInput, kernelTensor, new int[]{1, 1}, true);
        });
        
        assertTrue(exception.getMessage().contains("4D tensor"));
    }
    
    @Test
    public void testMaxPool() {
        // Test max pooling
        Tensor output = TensorOperations.maxPool(inputTensor, new int[]{2, 2}, new int[]{2, 2});
        
        // Output shape should be [1, 1, 2, 2]
        assertArrayEquals(new int[]{1, 1, 2, 2}, output.getShape());
        
        // Verify max pooled values
        assertEquals(6, output.get(0, 0, 0, 0)); // Max of [1,2,5,6]
        assertEquals(8, output.get(0, 0, 0, 1)); // Max of [3,4,7,8]
        assertEquals(14, output.get(0, 0, 1, 0)); // Max of [9,10,13,14]
        assertEquals(16, output.get(0, 0, 1, 1)); // Max of [11,12,15,16]
    }
    
    @Test
    public void testAvgPool() {
        // Test average pooling
        Tensor output = TensorOperations.avgPool(inputTensor, new int[]{2, 2}, new int[]{2, 2});
        
        // Output shape should be [1, 1, 2, 2]
        assertArrayEquals(new int[]{1, 1, 2, 2}, output.getShape());
        
        // Verify average pooled values
        assertEquals(3.5, output.get(0, 0, 0, 0)); // Avg of [1,2,5,6]
        assertEquals(5.5, output.get(0, 0, 0, 1)); // Avg of [3,4,7,8]
        assertEquals(11.5, output.get(0, 0, 1, 0)); // Avg of [9,10,13,14]
        assertEquals(13.5, output.get(0, 0, 1, 1)); // Avg of [11,12,15,16]
    }
    
    @Test
    public void testFlatten() {
        // Test flatten operation
        double[] flattened = TensorOperations.flatten(inputTensor);
        
        // The size should be the same
        assertEquals(inputTensor.getSize(), flattened.length);
        
        // The data should be the same
        assertArrayEquals(inputTensor.getData(), flattened);
    }
    
    @Test
    public void testReshape() {
        // Test reshape operation
        Tensor reshaped = TensorOperations.reshape(inputTensor, 2, 8);
        
        // Verify shape is changed but size remains the same
        assertArrayEquals(new int[]{2, 8}, reshaped.getShape());
        assertEquals(inputTensor.getSize(), reshaped.getSize());
        
        // The data should be preserved
        assertArrayEquals(inputTensor.getData(), reshaped.getData());
    }
    
    @Test
    public void testReshapeInvalidSize() {
        // Test reshape with incompatible size
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            TensorOperations.reshape(inputTensor, 5, 5);
        });
        
        assertTrue(exception.getMessage().contains("Cannot reshape"));
    }
    
    @Test
    public void testExpandDims() {
        // Start with a 3D tensor
        Tensor tensor3D = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, 1, 2, 3);
        
        // Add a dimension at position 1
        Tensor expanded = TensorOperations.expandDims(tensor3D, 1);
        
        // New shape should be [1, 1, 2, 3]
        assertArrayEquals(new int[]{1, 1, 2, 3}, expanded.getShape());
        assertEquals(tensor3D.getSize(), expanded.getSize());
        assertArrayEquals(tensor3D.getData(), expanded.getData());
    }
    
    @Test
    public void testAdd() {
        // Create two identical tensors
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, 2, 2);
        Tensor b = new Tensor(new double[]{5, 6, 7, 8}, 2, 2);
        
        // Add them
        Tensor result = TensorOperations.add(a, b);
        
        // Result should be element-wise addition
        assertArrayEquals(new int[]{2, 2}, result.getShape());
        assertEquals(6, result.get(0, 0)); // 1+5
        assertEquals(8, result.get(0, 1)); // 2+6
        assertEquals(10, result.get(1, 0)); // 3+7
        assertEquals(12, result.get(1, 1)); // 4+8
    }
    
    @Test
    public void testAddIncompatibleShapes() {
        // Create tensors with different shapes
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, 2, 2);
        Tensor b = new Tensor(new double[]{5, 6, 7, 8}, 4);
        
        // Adding these should throw exception
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            TensorOperations.add(a, b);
        });
        
        assertTrue(exception.getMessage().contains("shapes must match"));
    }
    
    @Test
    public void testTranspose() {
        // Create a 2D tensor (2x3)
        Tensor original = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        
        // Transpose dimensions (1,0) to get (3x2)
        Tensor transposed = TensorOperations.transpose(original, 1, 0);
        
        // Check shape
        assertArrayEquals(new int[]{3, 2}, transposed.getShape());
        
        // Check values
        assertEquals(1, transposed.get(0, 0)); // (0,0) -> (0,0)
        assertEquals(4, transposed.get(0, 1)); // (1,0) -> (0,1)
        assertEquals(2, transposed.get(1, 0)); // (0,1) -> (1,0)
        assertEquals(5, transposed.get(1, 1)); // (1,1) -> (1,1)
        assertEquals(3, transposed.get(2, 0)); // (0,2) -> (2,0)
        assertEquals(6, transposed.get(2, 1)); // (1,2) -> (2,1)
    }
    
    @Test
    public void testTransposeInvalidDimensions() {
        // Create a 2D tensor
        Tensor original = new Tensor(new double[]{1, 2, 3, 4}, 2, 2);
        
        // Invalid dimensions array
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            TensorOperations.transpose(original, 0, 0);
        });
        
        assertTrue(exception.getMessage().contains("Invalid dimensions"));
    }
    
    // Helper method to calculate the sum of a 3x3 region in the input tensor
    private double sum3x3Region(Tensor tensor, int batch, int channel, int startY, int startX, int padY, int padX) {
        double sum = 0;
        int[] shape = tensor.getShape();
        int height = shape[2];
        int width = shape[3];
        
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                int y = startY + ky - padY;
                int x = startX + kx - padX;
                
                if (y >= 0 && y < height && x >= 0 && x < width) {
                    sum += tensor.get(batch, channel, y, x);
                }
            }
        }
        
        return sum;
    }
}