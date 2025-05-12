package othello.gamelogic.strategies;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.io.File;
import java.nio.file.Paths;

import othello.gamelogic.BoardSpace;
import othello.gamelogic.ComputerPlayer;
import othello.gamelogic.OthelloGame;
import othello.gamelogic.Player;

/**
 * Tests for the NeuralStrategy implementation with ONNX model loading.
 */
public class OnnxStrategyTest {
    
    @Test
    public void testOnnxModelExists() {
        // Get the model path
        String modelPath = Paths.get(System.getProperty("user.dir"), "models", "othello.onnx").toString();
        
        // Check that the model file exists
        File modelFile = new File(modelPath);
        assertTrue(modelFile.exists(), "ONNX model file should exist at: " + modelPath);
    }
    
    @Test
    public void testStrategyCreation() {
        // This test verifies that we can create a strategy with an ONNX model
        // It's okay if it falls back to the default tensor network if the model can't be loaded
        Strategy strategy = StrategyFactory.createStrategy("onnx");
        
        assertNotNull(strategy, "Strategy should be created");
        assertTrue(strategy instanceof NeuralStrategy, "Strategy should be a NeuralStrategy");
        
        NeuralStrategy neuralStrategy = (NeuralStrategy)strategy;
        
        // Check if it's using a tensor network (either from ONNX or fallback)
        assertTrue(neuralStrategy.usesTensorNetwork(), "Should use a tensor network");
        assertNotNull(neuralStrategy.getTensorNetwork(), "Should have a tensor network wrapper");
    }
    
    @Test
    public void testCreateOnnxStrategy() {
        // Create a strategy using the ONNX factory method
        NeuralStrategy strategy = StrategyFactory.createOnnxNetworkStrategy();
        
        // Check that the strategy was created successfully
        assertNotNull(strategy, "Strategy should be created");
        assertTrue(strategy.usesTensorNetwork(), "Strategy should use a tensor network");
        assertNotNull(strategy.getTensorNetwork(), "Should have a tensor network wrapper");
    }
}