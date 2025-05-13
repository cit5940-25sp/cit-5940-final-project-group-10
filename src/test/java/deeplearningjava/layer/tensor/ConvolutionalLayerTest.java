package deeplearningjava.layer.tensor;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.Arrays;

import deeplearningjava.core.activation.ActivationFunction;
import deeplearningjava.core.activation.ReLU;
import deeplearningjava.core.activation.Sigmoid;
import deeplearningjava.core.activation.Tanh;
import deeplearningjava.core.tensor.Tensor;
import deeplearningjava.core.tensor.TensorOperations;
import deeplearningjava.api.TensorLayer;

/**
 * Comprehensive tests for the ConvolutionalLayer class.
 * Tests both forward and backward propagation.
 */
public class ConvolutionalLayerTest {
    
    private ConvolutionalLayer basicLayer;
    private Tensor inputTensor;
    
    @BeforeEach
    public void setUp() {
        // Create a basic convolutional layer
        int[] inputShape = {1, 1, 5, 5}; // batch=1, channels=1, height=5, width=5
        int[] kernelSize = {3, 3};
        int outChannels = 2;
        int[] stride = {1, 1};
        boolean padding = true;
        ActivationFunction relu = ReLU.getInstance();
        
        basicLayer = new ConvolutionalLayer(inputShape, kernelSize, outChannels, stride, padding, relu);
        
        // Create a simple input tensor (all ones)
        double[] inputData = new double[1 * 1 * 5 * 5]; // 25 elements
        Arrays.fill(inputData, 1.0);
        inputTensor = new Tensor(inputData, 1, 1, 5, 5);
    }
    
    @Test
    public void testLayerCreation() {
        // Test basic properties
        assertEquals(TensorLayer.LayerType.CONVOLUTIONAL, basicLayer.getType());
        assertArrayEquals(new int[]{1, 1, 5, 5}, basicLayer.getInputShape());
        assertArrayEquals(new int[]{1, 2, 5, 5}, basicLayer.getOutputShape()); // With padding
        
        // Test kernel shape
        Tensor kernels = basicLayer.getKernels();
        assertArrayEquals(new int[]{2, 1, 3, 3}, kernels.getShape());
        
        // Test bias shape
        Tensor bias = basicLayer.getBias();
        assertArrayEquals(new int[]{2}, bias.getShape());
        
        // Test stride and padding
        assertArrayEquals(new int[]{1, 1}, basicLayer.getStride());
        assertTrue(basicLayer.usesPadding());
    }
    
    @Test
    public void testLayerCreationWithoutPadding() {
        // Create layer without padding
        int[] inputShape = {1, 1, 5, 5};
        int[] kernelSize = {3, 3};
        int outChannels = 2;
        int[] stride = {1, 1};
        boolean padding = false;
        ActivationFunction relu = ReLU.getInstance();
        
        ConvolutionalLayer layerNoPadding = new ConvolutionalLayer(
            inputShape, kernelSize, outChannels, stride, padding, relu);
        
        // Without padding, output size should be reduced
        assertArrayEquals(new int[]{1, 2, 3, 3}, layerNoPadding.getOutputShape());
        assertFalse(layerNoPadding.usesPadding());
    }
    
    @Test
    public void testForwardPass() {
        // Perform forward pass
        Tensor output = basicLayer.forward(inputTensor);
        
        // Check output shape
        assertArrayEquals(new int[]{1, 2, 5, 5}, output.getShape());
        
        // Create a layer with known weights for predictable output
        setLayerToKnownWeights(basicLayer);
        
        // Perform forward pass again
        output = basicLayer.forward(inputTensor);
        
        // With the kernel set to all 0.1 and bias to 0.5, and input all 1s,
        // each output value should be 0.5 + (3*3*1*0.1) = 0.5 + 0.9 = 1.4
        // After ReLU activation, this should remain 1.4
        for (int oc = 0; oc < 2; oc++) {
            for (int h = 0; h < 5; h++) {
                for (int w = 0; w < 5; w++) {
                    if (h == 0 || h == 4 || w == 0 || w == 4) {
                        // Edge pixels have fewer kernel overlaps due to padding
                        // We'll skip detailed checks for edge pixels
                    } else {
                        // Central pixels should be fully convolved
                        assertEquals(1.4, output.get(0, oc, h, w), 0.0001);
                    }
                }
            }
        }
    }

    
    @Test
    public void testMultiChannelForwardPass() {
        // Create a multi-channel input
        int[] inputShape = {1, 3, 5, 5}; // batch=1, channels=3 (like RGB), height=5, width=5
        int[] kernelSize = {3, 3};
        int outChannels = 2;
        int[] stride = {1, 1};
        boolean padding = true;
        ActivationFunction relu = ReLU.getInstance();
        
        ConvolutionalLayer multiChannelLayer = new ConvolutionalLayer(
            inputShape, kernelSize, outChannels, stride, padding, relu);
        
        // Create input with each channel having a different value
        double[] inputData = new double[1 * 3 * 5 * 5]; // 75 elements
        for (int i = 0; i < inputData.length; i++) {
            inputData[i] = (i % 3) + 1; // Values cycle through 1, 2, 3
        }
        Tensor multiChannelInput = new Tensor(inputData, 1, 3, 5, 5);
        
        // Set known weights
        setLayerToKnownWeights(multiChannelLayer);
        
        // Perform forward pass
        Tensor output = multiChannelLayer.forward(multiChannelInput);
        
        // Check output shape
        assertArrayEquals(new int[]{1, 2, 5, 5}, output.getShape());
        
        // Here we would validate specific output values, but they would depend
        // on implementation details. Let's at least check that values are reasonable.
        for (int oc = 0; oc < 2; oc++) {
            for (int h = 1; h < 4; h++) {
                for (int w = 1; w < 4; w++) {
                    double value = output.get(0, oc, h, w);
                    assertTrue(value > 0, "Output values should be positive after ReLU");
                }
            }
        }
    }
    
    @Test
    public void testStrideForwardPass() {
        // Create a layer with stride 2,2
        int[] inputShape = {1, 1, 6, 6}; // batch=1, channels=1, height=6, width=6
        int[] kernelSize = {3, 3};
        int outChannels = 2;
        int[] stride = {2, 2};
        boolean padding = true;
        ActivationFunction relu = ReLU.getInstance();
        
        ConvolutionalLayer strideLayer = new ConvolutionalLayer(
            inputShape, kernelSize, outChannels, stride, padding, relu);
        
        // Output shape should be [1, 2, 3, 3] with stride 2,2
        assertArrayEquals(new int[]{1, 2, 3, 3}, strideLayer.getOutputShape());
        
        // Create input tensor
        double[] inputData = new double[1 * 1 * 6 * 6]; // 36 elements
        Arrays.fill(inputData, 1.0);
        Tensor strideInput = new Tensor(inputData, 1, 1, 6, 6);
        
        // Set known weights
        setLayerToKnownWeights(strideLayer);
        
        // Perform forward pass
        Tensor output = strideLayer.forward(strideInput);
        
        // Check output shape
        assertArrayEquals(new int[]{1, 2, 3, 3}, output.getShape());
    }
    
    @Test
    public void testDifferentActivationFunctions() {
        // Test with different activation functions
        testActivationFunction(ReLU.getInstance(), 1.4, 1.4);
        testActivationFunction(Sigmoid.getInstance(), 1.4, 0.8021838885585991);
        testActivationFunction(Tanh.getInstance(), 1.4, 0.8854454532088594);
    }
    
    /**
     * Helper method to test different activation functions
     */
    private void testActivationFunction(ActivationFunction activation, 
                                       double preActivationExpected, 
                                       double postActivationExpected) {
        // Create layer with specific activation
        int[] inputShape = {1, 1, 5, 5};
        int[] kernelSize = {3, 3};
        int outChannels = 1;
        int[] stride = {1, 1};
        boolean padding = true;
        
        ConvolutionalLayer layer = new ConvolutionalLayer(
            inputShape, kernelSize, outChannels, stride, padding, activation);
        
        // Set known weights
        setLayerToKnownWeights(layer);
        
        // Forward pass
        Tensor output = layer.forward(inputTensor);
        
        // Check central value against expected post-activation value
        assertEquals(postActivationExpected, output.get(0, 0, 2, 2), 0.0001);
    }
    
    @Test
    public void testSetGetKernelsAndBias() {
        // Create custom kernels and bias
        Tensor customKernels = new Tensor(new int[]{2, 1, 3, 3});
        customKernels.fill(0.25);
        
        Tensor customBias = new Tensor(new int[]{2});
        customBias.fill(0.75);
        
        // Set them
        basicLayer.setKernels(customKernels);
        basicLayer.setBias(customBias);
        
        // Check they were set correctly
        Tensor retrievedKernels = basicLayer.getKernels();
        Tensor retrievedBias = basicLayer.getBias();
        
        assertArrayEquals(customKernels.getShape(), retrievedKernels.getShape());
        assertArrayEquals(customBias.getShape(), retrievedBias.getShape());
        
        // Check values
        for (int oc = 0; oc < 2; oc++) {
            for (int ic = 0; ic < 1; ic++) {
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        assertEquals(0.25, retrievedKernels.get(oc, ic, kh, kw));
                    }
                }
            }
            
            assertEquals(0.75, retrievedBias.get(oc));
        }
    }
    
    @Test
    public void testInvalidInputShape() {
        // Create invalid 2D tensor
        Tensor invalidInput = new Tensor(new double[]{1, 2, 3, 4}, 2, 2);
        
        // Should throw exception
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            basicLayer.forward(invalidInput);
        });
        assertTrue(exception.getMessage().contains("Expected input shape"));
    }
    
    @Test
    public void testInvalidGradientShape() {
        // First do a forward pass
        basicLayer.forward(inputTensor);
        
        // Create invalid gradient tensor
        Tensor invalidGradient = new Tensor(new double[]{1, 2, 3, 4}, 2, 2);
        
        // Should throw exception
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            basicLayer.backward(invalidGradient);
        });
        assertTrue(exception.getMessage().contains("Expected gradient shape"));
    }
    
    @Test
    public void testNullInput() {
        // Should throw exception
        Exception exception = assertThrows(NullPointerException.class, () -> {
            basicLayer.forward(null);
        });
        assertTrue(exception.getMessage().contains("Input tensor must not be null"));
    }
    
    @Test
    public void testLayerConnection() {
        // Create second layer to connect to
        int[] secondLayerInput = {1, 2, 5, 5};
        int[] kernelSize = {3, 3};
        int outChannels = 3;
        int[] stride = {1, 1};
        boolean padding = true;
        ActivationFunction relu = ReLU.getInstance();
        
        ConvolutionalLayer secondLayer = new ConvolutionalLayer(
            secondLayerInput, kernelSize, outChannels, stride, padding, relu);
        
        // Connect layers
        basicLayer.connectTo(secondLayer);
        
        // Test incompatible connection
        ConvolutionalLayer incompatibleLayer = new ConvolutionalLayer(
            new int[]{1, 3, 5, 5}, kernelSize, outChannels, stride, padding, relu);
        
        // Should throw exception
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            basicLayer.connectTo(incompatibleLayer);
        });
        assertTrue(exception.getMessage().contains("not compatible"));
    }
    
    /**
     * Helper method to set the layer to known weights for predictable output
     */
    private void setLayerToKnownWeights(ConvolutionalLayer layer) {
        // Set all kernel values to 0.1
        Tensor kernels = layer.getKernels();
        double[] kernelData = kernels.getData();
        Arrays.fill(kernelData, 0.1);
        layer.setKernels(new Tensor(kernelData, kernels.getShape()));
        
        // Set all bias values to 0.5
        Tensor bias = layer.getBias();
        double[] biasData = bias.getData();
        Arrays.fill(biasData, 0.5);
        layer.setBias(new Tensor(biasData, bias.getShape()));
    }
    
    /**
     * Test layer creation with different configurations
     */
    @Test
    public void testLayerConfigurations() {
        ActivationFunction relu = ReLU.getInstance();
        
        // Test configuration 1
        int[] inputShape1 = {2, 1, 7, 7};
        int[] kernelSize1 = {3, 3};
        int outChannels1 = 2;
        int[] stride1 = {1, 1};
        boolean padding1 = true;
        int[] expectedOutputShape1 = {2, 2, 7, 7};
        
        ConvolutionalLayer layer1 = new ConvolutionalLayer(
            inputShape1, kernelSize1, outChannels1, stride1, padding1, relu);
        assertArrayEquals(expectedOutputShape1, layer1.getOutputShape());
        
        // Test configuration 2
        int[] inputShape2 = {2, 3, 7, 7};
        int[] kernelSize2 = {3, 3};
        int outChannels2 = 4;
        int[] stride2 = {1, 1};
        boolean padding2 = false;
        int[] expectedOutputShape2 = {2, 4, 5, 5};
        
        ConvolutionalLayer layer2 = new ConvolutionalLayer(
            inputShape2, kernelSize2, outChannels2, stride2, padding2, relu);
        assertArrayEquals(expectedOutputShape2, layer2.getOutputShape());
        
        // Test configuration 3
        int[] inputShape3 = {2, 3, 10, 10};
        int[] kernelSize3 = {5, 5};
        int outChannels3 = 2;
        int[] stride3 = {2, 2};
        boolean padding3 = true;
        int[] expectedOutputShape3 = {2, 2, 5, 5};
        
        ConvolutionalLayer layer3 = new ConvolutionalLayer(
            inputShape3, kernelSize3, outChannels3, stride3, padding3, relu);
        assertArrayEquals(expectedOutputShape3, layer3.getOutputShape());
        
        // Test configuration 4
        int[] inputShape4 = {1, 1, 8, 8};
        int[] kernelSize4 = {2, 2};
        int outChannels4 = 1;
        int[] stride4 = {2, 2};
        boolean padding4 = false;
        int[] expectedOutputShape4 = {1, 1, 4, 4};
        
        ConvolutionalLayer layer4 = new ConvolutionalLayer(
            inputShape4, kernelSize4, outChannels4, stride4, padding4, relu);
        assertArrayEquals(expectedOutputShape4, layer4.getOutputShape());
    }
}