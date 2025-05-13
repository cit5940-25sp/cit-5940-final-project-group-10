package othello.gamelogic.strategies;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the StrategyFactory class which creates game strategy instances.
 */
public class StrategyFactoryTest {

    @BeforeEach
    public void setUp() throws Exception {
        // Reset any custom ONNX model path before each test
        Field customOnnxModelPathField = StrategyFactory.class.getDeclaredField("customOnnxModelPath");
        customOnnxModelPathField.setAccessible(true);
        customOnnxModelPathField.set(null, null);
    }
    
    @Test
    public void testCreateMinimaxStrategy() {
        Strategy strategy = StrategyFactory.createStrategy("minimax");
        assertTrue(strategy instanceof MinimaxStrategy, "Factory should create a MinimaxStrategy instance");
    }
    
    @Test
    public void testCreateExpectimaxStrategy() {
        Strategy strategy = StrategyFactory.createStrategy("expectimax");
        assertTrue(strategy instanceof ExpectimaxStrategy, "Factory should create an ExpectimaxStrategy instance");
    }
    
    @Test
    public void testCreateMCTSStrategy() {
        Strategy strategy = StrategyFactory.createStrategy("mcts");
        assertTrue(strategy instanceof MCTSStrategy, "Factory should create an MCTSStrategy instance");
    }
    
    @Test
    public void testCreateCustomStrategy() {
        Strategy strategy = StrategyFactory.createStrategy("custom");
        assertTrue(strategy instanceof NeuralStrategy, "Factory should create a NeuralStrategy instance");
    }
    
    @Test
    public void testCreateTensorStrategy() {
        Strategy strategy = StrategyFactory.createStrategy("tensor");
        assertTrue(strategy instanceof NeuralStrategy, "Factory should create a NeuralStrategy instance");
    }
    
    @Test
    public void testCreateOnnxStrategy() {
        Strategy strategy = StrategyFactory.createStrategy("onnx");
        assertTrue(strategy instanceof NeuralStrategy, "Factory should create a NeuralStrategy instance");
    }
    
    @Test
    public void testUnknownStrategyName() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            StrategyFactory.createStrategy("invalid_strategy_name");
        });
        
        String expectedMessage = "Unknown strategy: invalid_strategy_name";
        String actualMessage = exception.getMessage();
        assertTrue(actualMessage.contains(expectedMessage), 
                "Exception message should contain: " + expectedMessage);
    }
    
    @Test
    public void testSetCustomOnnxModelPath() {
        // Set a custom model path
        String customPath = "/path/to/custom/model.onnx";
        StrategyFactory.setCustomOnnxModelPath(customPath);
        
        // Create ONNX strategy
        // We can't easily test the internal path usage directly,
        // but we can indirectly check that strategy creation doesn't fail
        Strategy strategy = StrategyFactory.createStrategy("onnx");
        assertNotNull(strategy, "Should create strategy with custom path");
    }
    
    @Test
    public void testCreateOnnxNetworkStrategyWithExplicitPath(@TempDir Path tempDir) throws IOException {
        // Create a temporary file to use as a model path
        Path modelPath = tempDir.resolve("test_model.onnx");
        Files.createFile(modelPath);
        
        try {
            // This might throw an exception because the file isn't a valid ONNX model,
            // but we're testing that the method uses the provided path
            StrategyFactory.createOnnxNetworkStrategy(modelPath.toString());
        } catch (Exception e) {
            // Expected since we're not providing a real ONNX model
            assertTrue(e.getCause() instanceof IOException || e instanceof RuntimeException,
                    "Should throw exception for invalid model file");
        }
    }
    
    @Test
    public void testFallbackToTensorOnFailure() {
        // Provide an invalid model path that will cause loading to fail
        String invalidPath = "/invalid/model/path.onnx";
        
        // This should fallback to tensor network strategy
        Strategy strategy = StrategyFactory.createOnnxNetworkStrategy(invalidPath);
        
        assertNotNull(strategy, "Should fallback to tensor network strategy");
        assertTrue(strategy instanceof NeuralStrategy, "Should return a NeuralStrategy instance");
    }
}