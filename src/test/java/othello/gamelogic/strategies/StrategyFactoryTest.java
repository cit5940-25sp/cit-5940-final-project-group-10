package othello.gamelogic.strategies;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Assertions;
// Removed imports that were only used for ONNX tests

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the StrategyFactory class which creates game strategy instances.
 */
public class StrategyFactoryTest {

    @BeforeEach
    public void setUp() {
        // No setup needed after ONNX tests were removed
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
    public void testUnknownStrategyName() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            StrategyFactory.createStrategy("invalid_strategy_name");
        });
        
        String expectedMessage = "Unknown strategy: invalid_strategy_name";
        String actualMessage = exception.getMessage();
        assertTrue(actualMessage.contains(expectedMessage), 
                "Exception message should contain: " + expectedMessage);
    }
    
    // ONNX-related tests have been removed
}