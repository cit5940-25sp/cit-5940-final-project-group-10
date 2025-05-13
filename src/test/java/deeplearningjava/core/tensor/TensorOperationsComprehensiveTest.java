package deeplearningjava.core.tensor;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Arrays;
import java.util.stream.Stream;

/**
 * Comprehensive tests for the TensorOperations utility class.
 * This covers more complex scenarios and edge cases.
 */
public class TensorOperationsComprehensiveTest {
    
    private Tensor inputTensor4D;
    private Tensor inputTensor3D;
    private Tensor kernelTensor;
    
    @BeforeEach
    public void setUp() {
        // Create a 4D input tensor [2, 3, 4, 4] (batch_size=2, channels=3, height=4, width=4)
        double[] inputData4D = new double[2 * 3 * 4 * 4];
        // Fill with incremental values
        for (int i = 0; i < inputData4D.length; i++) {
            inputData4D[i] = i + 1;
        }
        inputTensor4D = new Tensor(inputData4D, 2, 3, 4, 4);
        
        // Create a 3D input tensor [3, 5, 5] (channels=3, height=5, width=5)
        double[] inputData3D = new double[3 * 5 * 5];
        for (int i = 0; i < inputData3D.length; i++) {
            inputData3D[i] = i + 1;
        }
        inputTensor3D = new Tensor(inputData3D, 3, 5, 5);
        
        // Create a 4D kernel tensor [4, 3, 3, 3] (out_channels=4, in_channels=3, height=3, width=3)
        double[] kernelData = new double[4 * 3 * 3 * 3];
        // Fill with specific patterns for each output channel
        for (int oc = 0; oc < 4; oc++) {
            for (int ic = 0; ic < 3; ic++) {
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int idx = ((oc * 3 + ic) * 3 + kh) * 3 + kw;
                        switch (oc) {
                            case 0: // Identity filter
                                kernelData[idx] = (kh == 1 && kw == 1) ? 1.0 : 0.0;
                                break;
                            case 1: // Edge detection (Laplacian)
                                kernelData[idx] = (kh == 1 && kw == 1) ? -4.0 : 
                                                 ((kh == 1 || kw == 1) && !(kh == 1 && kw == 1)) ? 1.0 : 0.0;
                                break;
                            case 2: // Blur filter
                                kernelData[idx] = 1.0 / 9.0;
                                break;
                            case 3: // Sharpen filter
                                kernelData[idx] = (kh == 1 && kw == 1) ? 5.0 : -1.0;
                                break;
                        }
                    }
                }
            }
        }
        kernelTensor = new Tensor(kernelData, 4, 3, 3, 3);
    }
    
    @Test
    public void testConvolveWithMultiChannelInput() {
        // Test convolution with multi-channel input (RGB-like)
        Tensor output = TensorOperations.convolve(inputTensor4D, kernelTensor, new int[]{1, 1}, true);
        
        // Check shape: [2, 4, 4, 4] (2 batches, 4 output channels, 4x4 output size with padding)
        assertArrayEquals(new int[]{2, 4, 4, 4}, output.getShape());
        
        // We can check specific locations to verify different filter behaviors
        // For example, edge filters should have higher values at edges
        double edgeValue = output.get(0, 1, 1, 2);  // Center-edge
        double centerValue = output.get(0, 1, 2, 2); // Center
        
        // In the Laplacian edge detection filter (output channel 1), 
        // edges should have higher values than centers
        assertTrue(Math.abs(edgeValue) != Math.abs(centerValue), 
                "Edge detection filter should differentiate between edges and centers");
        
        // Validate different portions of the output
        // For the blur filter (output channel 2), check that a region's value is the average
        // of surrounding values in the input (with appropriate cross-channel computation)
        double blurValue = output.get(0, 2, 2, 2);
        double expectedBlurValue = calculateBlurValue(inputTensor4D, 0, 1, 1);
        assertEquals(expectedBlurValue, blurValue, 0.001);
    }
    
    @Test
    public void testConvolveWithDifferentStrides() {
        // Test with stride 2,2
        Tensor output = TensorOperations.convolve(inputTensor4D, kernelTensor, new int[]{2, 2}, true);
        
        // Output shape should be [2, 4, 2, 2] with stride 2,2
        assertArrayEquals(new int[]{2, 4, 2, 2}, output.getShape());
        
        // Test with asymmetric stride 1,2
        Tensor outputAsymmetric = TensorOperations.convolve(inputTensor4D, kernelTensor, new int[]{1, 2}, true);
        
        // Output shape should be [2, 4, 4, 2] with stride 1,2
        assertArrayEquals(new int[]{2, 4, 4, 2}, outputAsymmetric.getShape());
    }
    
    @Test
    public void testConvolveEdgeCases() {
        // Create a 1x1 kernel (essentially a point-wise operation)
        Tensor smallKernel = new Tensor(new double[]{1, 2, 3}, 3, 1, 1, 1);
        
        // Test with 1x1 kernel
        Tensor output = TensorOperations.convolve(inputTensor4D, smallKernel, new int[]{1, 1}, false);
        
        // Output shape should maintain spatial dimensions
        assertArrayEquals(new int[]{2, 3, 4, 4}, output.getShape());
        
        // For the first channel, each value should be multiplied by 1
        for (int b = 0; b < 2; b++) {
            for (int h = 0; h < 4; h++) {
                for (int w = 0; w < 4; w++) {
                    assertEquals(inputTensor4D.get(b, 0, h, w), output.get(b, 0, h, w));
                }
            }
        }
        
        // For the second channel, each value should be multiplied by 2
        for (int b = 0; b < 2; b++) {
            for (int h = 0; h < 4; h++) {
                for (int w = 0; w < 4; w++) {
                    assertEquals(inputTensor4D.get(b, 1, h, w) * 2, output.get(b, 1, h, w));
                }
            }
        }
    }
    
    @Test
    public void testConvolveInvalidDimensionality() {
        // Create a 2D input tensor
        Tensor tensor2D = new Tensor(new double[]{1, 2, 3, 4}, 2, 2);
        
        // Should throw exception for non-4D input
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            TensorOperations.convolve(tensor2D, kernelTensor, new int[]{1, 1}, true);
        });
        assertTrue(exception.getMessage().contains("4D tensor"));
        
        // Create a 3D kernel tensor
        Tensor kernel3D = new Tensor(new double[]{1, 2, 3, 4}, 2, 2, 1);
        
        // Should throw exception for non-4D kernel
        exception = assertThrows(IllegalArgumentException.class, () -> {
            TensorOperations.convolve(inputTensor4D, kernel3D, new int[]{1, 1}, true);
        });
        assertTrue(exception.getMessage().contains("4D tensor"));
    }
    
    @Test
    public void testMaxPoolWithVariousSizesAndStrides() {
        // Test with 2x2 pool and 2x2 stride (standard pooling)
        Tensor output1 = TensorOperations.maxPool(inputTensor4D, new int[]{2, 2}, new int[]{2, 2});
        assertArrayEquals(new int[]{2, 3, 2, 2}, output1.getShape());
        
        // Test with 3x3 pool and 1x1 stride (overlapping pooling)
        Tensor output2 = TensorOperations.maxPool(inputTensor4D, new int[]{3, 3}, new int[]{1, 1});
        assertArrayEquals(new int[]{2, 3, 2, 2}, output2.getShape());
        
        // Test with asymmetric pool size (2x3)
        Tensor output3 = TensorOperations.maxPool(inputTensor4D, new int[]{2, 3}, new int[]{2, 2});
        assertArrayEquals(new int[]{2, 3, 2, 1}, output3.getShape());
        
        // Compare results - overlapping pooling should have higher values
        assertTrue(output2.get(0, 0, 0, 0) >= output1.get(0, 0, 0, 0), 
                "Overlapping pooling should capture more maximum values");
    }
    
    @Test
    public void testAvgPoolWithVariousSizesAndStrides() {
        // Test with 2x2 pool and 2x2 stride
        Tensor output1 = TensorOperations.avgPool(inputTensor4D, new int[]{2, 2}, new int[]{2, 2});
        assertArrayEquals(new int[]{2, 3, 2, 2}, output1.getShape());
        
        // Test with 3x3 pool and 1x1 stride
        Tensor output2 = TensorOperations.avgPool(inputTensor4D, new int[]{3, 3}, new int[]{1, 1});
        assertArrayEquals(new int[]{2, 3, 2, 2}, output2.getShape());
        
        // Verify averaging - for first 2x2 region in batch 0, channel 0
        double sum = inputTensor4D.get(0, 0, 0, 0) + inputTensor4D.get(0, 0, 0, 1) +
                     inputTensor4D.get(0, 0, 1, 0) + inputTensor4D.get(0, 0, 1, 1);
        double avg = sum / 4;
        assertEquals(avg, output1.get(0, 0, 0, 0), 0.001);
    }
    
    @Test
    public void testPoolingInvalidDimensionality() {
        // Create a 2D input tensor
        Tensor tensor2D = new Tensor(new double[]{1, 2, 3, 4}, 2, 2);
        
        // Should throw exception for non-4D input
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            TensorOperations.maxPool(tensor2D, new int[]{2, 2}, new int[]{2, 2});
        });
        assertTrue(exception.getMessage().contains("4D tensor"));
        
        exception = assertThrows(IllegalArgumentException.class, () -> {
            TensorOperations.avgPool(tensor2D, new int[]{2, 2}, new int[]{2, 2});
        });
        assertTrue(exception.getMessage().contains("4D tensor"));
    }
    
    @Test
    public void testExpandDimsWithMultipleExpansions() {
        // Start with a 2D tensor
        Tensor tensor2D = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        
        // Expand at beginning (position 0)
        Tensor expanded1 = TensorOperations.expandDims(tensor2D, 0);
        assertArrayEquals(new int[]{1, 2, 3}, expanded1.getShape());
        
        // Expand at end
        Tensor expanded2 = TensorOperations.expandDims(tensor2D, 2);
        assertArrayEquals(new int[]{2, 3, 1}, expanded2.getShape());
        
        // Double expansion (two operations in sequence)
        Tensor doubleExpanded = TensorOperations.expandDims(
                TensorOperations.expandDims(tensor2D, 0), 1);
        assertArrayEquals(new int[]{1, 1, 2, 3}, doubleExpanded.getShape());
        
        // Verify values are preserved
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                double original = tensor2D.get(i, j);
                assertEquals(original, expanded1.get(0, i, j));
                assertEquals(original, expanded2.get(i, j, 0));
                assertEquals(original, doubleExpanded.get(0, 0, i, j));
            }
        }
    }
    
    @Test
    public void testTransposeComprehensive() {
        // Create a 3D tensor [2, 3, 4]
        double[] data = new double[2 * 3 * 4];
        for (int i = 0; i < data.length; i++) {
            data[i] = i + 1;
        }
        Tensor tensor3D = new Tensor(data, 2, 3, 4);
        
        // Test different permutations
        
        // 1. Standard transpose [2,3,4] -> [4,3,2]
        Tensor transposed1 = TensorOperations.transpose(tensor3D, 2, 1, 0);
        assertArrayEquals(new int[]{4, 3, 2}, transposed1.getShape());
        
        // 2. Swap first two dimensions [2,3,4] -> [3,2,4]
        Tensor transposed2 = TensorOperations.transpose(tensor3D, 1, 0, 2);
        assertArrayEquals(new int[]{3, 2, 4}, transposed2.getShape());
        
        // 3. Cyclic permutation [2,3,4] -> [3,4,2]
        Tensor transposed3 = TensorOperations.transpose(tensor3D, 1, 2, 0);
        assertArrayEquals(new int[]{3, 4, 2}, transposed3.getShape());
        
        // Verify values for a specific position
        assertEquals(tensor3D.get(1, 2, 3), transposed1.get(3, 2, 1));
        assertEquals(tensor3D.get(1, 2, 3), transposed2.get(2, 1, 3));
        assertEquals(tensor3D.get(1, 2, 3), transposed3.get(2, 3, 1));
    }
    
    @Test
    public void testTransposeInvalidPermutations() {
        // Create a 3D tensor
        Tensor tensor3D = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, 1, 2, 3);
        
        // 1. Wrong number of dimensions
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            TensorOperations.transpose(tensor3D, 0, 1);
        });
        assertTrue(exception.getMessage().contains("Dimensions array must have the same length"));
        
        // 2. Repeated dimension
        exception = assertThrows(IllegalArgumentException.class, () -> {
            TensorOperations.transpose(tensor3D, 0, 0, 1);
        });
        assertTrue(exception.getMessage().contains("Invalid dimensions array"));
        
        // 3. Out of range dimension
        exception = assertThrows(IllegalArgumentException.class, () -> {
            TensorOperations.transpose(tensor3D, 0, 1, 3);
        });
        assertTrue(exception.getMessage().contains("Invalid dimensions array"));
    }
    
    @Test
    public void testAddAndElementWiseOperations() {
        // Test add with broadcasting
        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor b = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        
        // Standard addition
        Tensor result = TensorOperations.add(a, b);
        assertArrayEquals(new int[]{2, 3}, result.getShape());
        
        // Verify element-wise addition
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(a.get(i, j) + b.get(i, j), result.get(i, j));
            }
        }
    }
    
    @Test
    public void testReshapeComprehensive() {
        // Create a tensor with known data for reshape testing
        double[] data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Tensor original = new Tensor(data, 2, 3, 4);
        
        // 1. Reshape to 2D (6, 4)
        Tensor reshaped1 = TensorOperations.reshape(original, 6, 4);
        assertArrayEquals(new int[]{6, 4}, reshaped1.getShape());
        
        // 2. Reshape to 1D (24)
        Tensor reshaped2 = TensorOperations.reshape(original, 24);
        assertArrayEquals(new int[]{24}, reshaped2.getShape());
        
        // 3. Reshape to 4D (2, 2, 2, 3)
        Tensor reshaped3 = TensorOperations.reshape(original, 2, 2, 2, 3);
        assertArrayEquals(new int[]{2, 2, 2, 3}, reshaped3.getShape());
        
        // Verify data is preserved with correct order
        assertArrayEquals(data, reshaped1.getData());
        assertArrayEquals(data, reshaped2.getData());
        assertArrayEquals(data, reshaped3.getData());
        
        // Verify invalid reshape throws exception
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            TensorOperations.reshape(original, 5, 5);
        });
        assertTrue(exception.getMessage().contains("Cannot reshape tensor"));
    }
    
    /**
     * Tests performance with large tensors to ensure efficiency.
     * This test is optional and can be disabled if it takes too long.
     */
    @Test
    public void testLargeTensorOperations() {
        // Create large input and kernel
        int batchSize = 8;
        int inputChannels = 16;
        int height = 64;
        int width = 64;
        int outputChannels = 32;
        int kernelSize = 3;
        
        // Initialize large tensors
        double[] inputData = new double[batchSize * inputChannels * height * width];
        Arrays.fill(inputData, 1.0);
        Tensor largeInput = new Tensor(inputData, batchSize, inputChannels, height, width);
        
        double[] kernelData = new double[outputChannels * inputChannels * kernelSize * kernelSize];
        Arrays.fill(kernelData, 0.1);
        Tensor largeKernel = new Tensor(kernelData, outputChannels, inputChannels, kernelSize, kernelSize);
        
        // Measure convolution performance
        long startTime = System.currentTimeMillis();
        Tensor result = TensorOperations.convolve(largeInput, largeKernel, new int[]{1, 1}, true);
        long endTime = System.currentTimeMillis();
        
        // Check result shape
        assertArrayEquals(new int[]{batchSize, outputChannels, height, width}, result.getShape());
        
        // Log performance info
        System.out.println("Large tensor convolution time: " + (endTime - startTime) + "ms");
    }
    
    // Helper method to calculate expected blur value (average of 3x3 region across channels)
    private double calculateBlurValue(Tensor input, int batch, int centerY, int centerX) {
        double sum = 0;
        int count = 0;
        int channels = input.getShape()[1];
        
        for (int c = 0; c < channels; c++) {
            for (int y = centerY - 1; y <= centerY + 1; y++) {
                for (int x = centerX - 1; x <= centerX + 1; x++) {
                    try {
                        sum += input.get(batch, c, y, x) * (1.0 / 9.0); // 1/9 is the blur weight
                        count++;
                    } catch (IndexOutOfBoundsException e) {
                        // Skip if out of bounds
                    }
                }
            }
        }
        
        return sum;
    }
}