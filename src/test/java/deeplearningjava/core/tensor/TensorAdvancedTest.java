package deeplearningjava.core.tensor;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.Arrays;
import java.util.Random;

/**
 * Advanced tests for the Tensor class, focusing on edge cases and complex operations.
 */
public class TensorAdvancedTest {
    
    private Random random;
    
    @BeforeEach
    public void setUp() {
        random = new Random(12345); // Fixed seed for reproducibility
    }
    
    @Test
    public void testEmptyTensor() {
        // Test tensors with zero dimensions (empty tensors)
        Tensor emptyTensor = new Tensor(0);
        assertEquals(0, emptyTensor.getSize());
        assertArrayEquals(new int[]{0}, emptyTensor.getShape());
    }
    
    @Test
    public void testSingleElementTensor() {
        // Test tensors with a single element
        Tensor singleElementTensor = new Tensor(new double[]{42}, 1);
        assertEquals(1, singleElementTensor.getSize());
        assertEquals(42, singleElementTensor.get(0));
        
        // Also test single element in multiple dimensions
        Tensor tensor1x1x1 = new Tensor(new double[]{42}, 1, 1, 1);
        assertEquals(1, tensor1x1x1.getSize());
        assertEquals(42, tensor1x1x1.get(0, 0, 0));
    }
    
    @Test
    public void testHighDimensionalTensor() {
        // Test tensor with many dimensions (5D)
        int[] shape = {2, 3, 2, 2, 2};
        double[] data = new double[2 * 3 * 2 * 2 * 2];
        for (int i = 0; i < data.length; i++) {
            data[i] = i;
        }
        
        Tensor highDimTensor = new Tensor(data, shape);
        assertEquals(5, highDimTensor.getRank());
        assertEquals(48, highDimTensor.getSize());
        assertArrayEquals(shape, highDimTensor.getShape());
        
        // Test accessing elements in 5D space
        assertEquals(0, highDimTensor.get(0, 0, 0, 0, 0));
        assertEquals(47, highDimTensor.get(1, 2, 1, 1, 1));
    }
    
    @Test
    public void testIndexOutOfBounds() {
        // Test accessing indices outside of bounds
        Tensor tensor = new Tensor(new double[]{1, 2, 3, 4}, 2, 2);
        
        // Negative indices
        Exception exception = assertThrows(IndexOutOfBoundsException.class, () -> {
            tensor.get(-1, 0);
        });
        assertTrue(exception.getMessage().contains("out of bounds"));
        
        // Indices beyond shape
        exception = assertThrows(IndexOutOfBoundsException.class, () -> {
            tensor.get(0, 2);
        });
        assertTrue(exception.getMessage().contains("out of bounds"));
        
        // Setting out of bounds
        exception = assertThrows(IndexOutOfBoundsException.class, () -> {
            tensor.set(10, 2, 1);
        });
        assertTrue(exception.getMessage().contains("out of bounds"));
    }
    
    @Test
    public void testIncorrectDimensionAccess() {
        // Test providing wrong number of indices
        Tensor tensor3D = new Tensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2);
        
        // Too few dimensions
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            tensor3D.get(1, 1);
        });
        assertTrue(exception.getMessage().contains("doesn't match tensor dimensionality"));
        
        // Too many dimensions
        exception = assertThrows(IllegalArgumentException.class, () -> {
            tensor3D.get(0, 0, 0, 0);
        });
        assertTrue(exception.getMessage().contains("doesn't match tensor dimensionality"));
        
        // Null indices
        exception = assertThrows(NullPointerException.class, () -> {
            tensor3D.get((int[])null);
        });
        assertTrue(exception.getMessage().contains("must not be null"));
    }
    
    @Test
    public void testInvalidShapeCreation() {
        // Test creating a tensor with inconsistent data and shape
        double[] data = {1, 2, 3, 4};
        
        // Data length doesn't match shape product
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            new Tensor(data, 3, 2);
        });
        assertTrue(exception.getMessage().contains("Data length 4 doesn't match the shape size 6"));
    }
    
    @Test
    public void testLargeTensorCreation() {
        // Test creating and working with large tensors
        int size = 1000000; // 1 million elements
        Tensor largeTensor = new Tensor(size);
        
        assertEquals(size, largeTensor.getSize());
        assertEquals(1, largeTensor.getRank());
        
        // Test setting and getting values in large tensor
        largeTensor.set(42, 500000);
        assertEquals(42, largeTensor.get(500000));
    }
    
    @Test
    public void testStridesComputation() {
        // Test that strides are computed correctly
        Tensor tensor3D = new Tensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 2, 3, 2);
        
        int[] strides = tensor3D.getStrides();
        assertArrayEquals(new int[]{6, 2, 1}, strides);
        
        // Verify correct element access using strides
        assertEquals(1, tensor3D.get(0, 0, 0)); // offset = 0*6 + 0*2 + 0*1 = 0
        assertEquals(8, tensor3D.get(1, 0, 1)); // offset = 1*6 + 0*2 + 1*1 = 7
        assertEquals(11, tensor3D.get(1, 2, 0)); // offset = 1*6 + 2*2 + 0*1 = 10
    }
    
    @Test
    public void testTensorCopy() {
        // Test deep copy semantics
        Tensor original = new Tensor(new double[]{1, 2, 3, 4}, 2, 2);
        Tensor copy = original.copy();
        
        // Modify original, should not affect copy
        original.set(100, 0, 0);
        assertEquals(1, copy.get(0, 0));
        
        // Verify all data is copied correctly
        assertEquals(original.getSize(), copy.getSize());
        assertArrayEquals(original.getShape(), copy.getShape());
        
        // Complete independence of data
        original.fill(99);
        assertEquals(2, copy.get(0, 1));
        assertEquals(99, original.get(0, 1));
    }
    
    @Test
    public void testTensorFill() {
        // Test filling tensor with values
        Tensor tensor = new Tensor(3, 3);
        
        // Fill with a value
        tensor.fill(42);
        
        // Verify all elements are filled
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(42, tensor.get(i, j));
            }
        }
        
        // Test chaining with other operations
        Tensor filledAndMapped = tensor.fill(2).map(x -> x * 2);
        
        // Verify operations chained correctly
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(4, filledAndMapped.get(i, j));
            }
        }
    }
    
    @Test
    public void testTensorMap() {
        // Test applying functions to tensor elements
        Tensor tensor = new Tensor(new double[]{1, 2, 3, 4}, 2, 2);
        
        // Apply a square function
        Tensor squared = tensor.map(x -> x * x);
        
        // Verify function was applied correctly
        assertEquals(1, squared.get(0, 0)); // 1^2
        assertEquals(4, squared.get(0, 1)); // 2^2
        assertEquals(9, squared.get(1, 0)); // 3^2
        assertEquals(16, squared.get(1, 1)); // 4^2
        
        // Original tensor should be unchanged
        assertEquals(1, tensor.get(0, 0));
    }
    
    /**
     * Test creating tensors with different shapes
     */
    @Test
    public void testMultipleShapes() {
        // Test shape: [5]
        int[] shape1 = new int[]{5};
        Tensor tensor1 = new Tensor(shape1);
        assertEquals(5, tensor1.getSize());
        assertEquals(1, tensor1.getRank());
        assertArrayEquals(shape1, tensor1.getShape());
        
        // Test shape: [2, 3]
        int[] shape2 = new int[]{2, 3};
        Tensor tensor2 = new Tensor(shape2);
        assertEquals(6, tensor2.getSize());
        assertEquals(2, tensor2.getRank());
        assertArrayEquals(shape2, tensor2.getShape());
        
        // Test shape: [2, 2, 2]
        int[] shape3 = new int[]{2, 2, 2};
        Tensor tensor3 = new Tensor(shape3);
        assertEquals(8, tensor3.getSize());
        assertEquals(3, tensor3.getRank());
        assertArrayEquals(shape3, tensor3.getShape());
        
        // Test shape: [1, 3, 5, 7]
        int[] shape4 = new int[]{1, 3, 5, 7};
        Tensor tensor4 = new Tensor(shape4);
        assertEquals(105, tensor4.getSize());
        assertEquals(4, tensor4.getRank());
        assertArrayEquals(shape4, tensor4.getShape());
        
        // Test shape: [2, 2, 2, 2, 2]
        int[] shape5 = new int[]{2, 2, 2, 2, 2};
        Tensor tensor5 = new Tensor(shape5);
        assertEquals(32, tensor5.getSize());
        assertEquals(5, tensor5.getRank());
        assertArrayEquals(shape5, tensor5.getShape());
    }

    
    @Test
    public void testEdgeCaseShapesFailure() {
        // Test case: Negative dimension [-1]
        int[] shape1 = new int[]{-1};
        assertThrows(IllegalArgumentException.class, () -> {
            new Tensor(shape1);
        });
        
        // Test case: Too large dimension [Integer.MAX_VALUE]
        int[] shape2 = new int[]{Integer.MAX_VALUE};
        assertThrows(IllegalArgumentException.class, () -> {
            new Tensor(shape2);
        });
        
        // Test case: No dimensions []
        int[] shape3 = new int[]{};
        assertThrows(IllegalArgumentException.class, () -> {
            new Tensor(shape3);
        });
    }
    
    @Test
    public void testSpecialValuesInTensor() {
        // Test handling of special values like NaN and infinity
        double[] specialValues = {
            Double.NaN, 
            Double.POSITIVE_INFINITY, 
            Double.NEGATIVE_INFINITY,
            Double.MAX_VALUE,
            Double.MIN_VALUE
        };
        
        Tensor tensor = new Tensor(specialValues, 5);
        
        // Verify special values are preserved
        assertTrue(Double.isNaN(tensor.get(0)));
        assertTrue(Double.isInfinite(tensor.get(1)));
        assertTrue(Double.isInfinite(tensor.get(2)));
        assertEquals(Double.MAX_VALUE, tensor.get(3));
        assertEquals(Double.MIN_VALUE, tensor.get(4));
        
        // Test mapping function on special values
        Tensor mapped = tensor.map(x -> x * 2);
        assertTrue(Double.isNaN(mapped.get(0)));  // NaN * 2 = NaN
        assertTrue(Double.isInfinite(mapped.get(1))); // Inf * 2 = Inf
    }
    
    @Test
    public void testPerformanceOnLargeOperations() {
        // Create reasonably large tensors to test performance
        int size = 100;
        Tensor largeTensor = new Tensor(size, size);
        
        // Fill with random data
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                largeTensor.set(random.nextDouble(), i, j);
            }
        }
        
        // Measure performance of map operation
        long startTime = System.currentTimeMillis();
        Tensor result = largeTensor.map(Math::sin);
        long endTime = System.currentTimeMillis();
        
        System.out.println("Time to apply sin to " + (size*size) + 
                           " elements: " + (endTime - startTime) + "ms");
        
        // Basic validation
        assertEquals(size * size, result.getSize());
    }
    
    @Test
    public void testSerializationCompatibility() {
        // Test that getData() and the constructor form a serialization/deserialization pair
        double[] originalData = {1, 2, 3, 4, 5, 6};
        int[] shape = {2, 3};
        
        Tensor original = new Tensor(originalData, shape);
        
        // "Serialize"
        double[] serializedData = original.getData();
        int[] serializedShape = original.getShape();
        
        // "Deserialize"
        Tensor deserialized = new Tensor(serializedData, serializedShape);
        
        // Verify they're equivalent
        assertArrayEquals(original.getShape(), deserialized.getShape());
        assertArrayEquals(original.getData(), deserialized.getData());
        
        // Compare values at each position
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                assertEquals(original.get(i, j), deserialized.get(i, j));
            }
        }
    }
    
    @Test
    public void testTensorFunctionInterface() {
        // Test the TensorFunction interface
        Tensor tensor = new Tensor(new double[]{1, 2, 3, 4}, 2, 2);
        
        // Test with lambda
        Tensor result1 = tensor.map(x -> x + 1);
        
        // Test with anonymous inner class
        Tensor result2 = tensor.map(new Tensor.TensorFunction() {
            @Override
            public double apply(double value) {
                return value * 2;
            }
        });
        
        // Verify both approaches work
        assertEquals(2, result1.get(0, 0)); // 1+1
        assertEquals(2, result2.get(0, 0)); // 1*2
    }
    
    @Test
    public void testToStringMethod() {
        // Ensure toString contains useful debugging information
        Tensor tensor = new Tensor(new double[]{1, 2, 3, 4}, 2, 2);
        String str = tensor.toString();
        
        // Should contain shape and data
        assertTrue(str.contains("shape=[2, 2]"));
        assertTrue(str.contains("data=[1.0, 2.0, 3.0, 4.0]"));
    }
}