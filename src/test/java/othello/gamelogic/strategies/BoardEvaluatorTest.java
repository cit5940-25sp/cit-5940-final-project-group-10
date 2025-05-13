package othello.gamelogic.strategies;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import othello.Constants;
import othello.gamelogic.BoardSpace;
import othello.gamelogic.OthelloGame;
import othello.gamelogic.Player;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the BoardEvaluator interface and implementations.
 */
public class BoardEvaluatorTest {

    private BoardSpace[][] board;
    private TestPlayer player;
    private TestPlayer opponent;
    
    /**
     * Simple test implementation of Player class
     */
    private static class TestPlayer extends Player {
        public TestPlayer(BoardSpace.SpaceType color) {
            setColor(color);
        }
    }
    
    /**
     * Simple implementation of BoardEvaluator for testing
     */
    private static class TestEvaluator implements BoardEvaluator {
        @Override
        public double evaluate(BoardSpace[][] board, Player player, Player opponent) {
            // Simple piece counting evaluator
            int playerPieces = 0;
            int opponentPieces = 0;
            
            for (BoardSpace[] row : board) {
                for (BoardSpace space : row) {
                    if (space.getType() == player.getColor()) {
                        playerPieces++;
                    } else if (space.getType() == opponent.getColor()) {
                        opponentPieces++;
                    }
                }
            }
            
            return playerPieces - opponentPieces;
        }
    }

    @BeforeEach
    public void setUp() {
        board = new BoardSpace[OthelloGame.GAME_BOARD_SIZE][OthelloGame.GAME_BOARD_SIZE];
        
        // Initialize an empty board
        for (int i = 0; i < OthelloGame.GAME_BOARD_SIZE; i++) {
            for (int j = 0; j < OthelloGame.GAME_BOARD_SIZE; j++) {
                board[i][j] = new BoardSpace(i, j, BoardSpace.SpaceType.EMPTY);
            }
        }
        
        // Create players
        player = new TestPlayer(BoardSpace.SpaceType.BLACK);
        opponent = new TestPlayer(BoardSpace.SpaceType.WHITE);
    }

    @Test
    public void testCustomEvaluator() {
        // Create a custom evaluator
        BoardEvaluator evaluator = new TestEvaluator();
        
        // Empty board should have a score of 0
        double emptyScore = evaluator.evaluate(board, player, opponent);
        assertEquals(0.0, emptyScore, "Empty board should have score of 0");
        
        // Add player pieces
        board[3][3].setType(BoardSpace.SpaceType.BLACK);
        board[3][4].setType(BoardSpace.SpaceType.BLACK);
        board[4][4].setType(BoardSpace.SpaceType.BLACK);
        
        // Add opponent pieces
        board[4][3].setType(BoardSpace.SpaceType.WHITE);
        
        // Score should be player pieces - opponent pieces
        double score = evaluator.evaluate(board, player, opponent);
        assertEquals(2.0, score, "Score should be player pieces - opponent pieces");
        
        // Test with reversed player roles
        double reversedScore = evaluator.evaluate(board, opponent, player);
        assertEquals(-2.0, reversedScore, "Reversed score should be opponent pieces - player pieces");
    }

    @Test
    public void testWeightedEvaluator() {
        // Test that the WeightedEvaluator implements BoardEvaluator correctly
        BoardEvaluator weightedEvaluator = new WeightedEvaluator();
        
        // Place pieces on the board at positions with different weights
        board[0][0].setType(BoardSpace.SpaceType.BLACK); // Player in corner (high value: 200)
        board[1][1].setType(BoardSpace.SpaceType.WHITE); // Opponent in poor position (negative value: -100)
        
        // Evaluate the board
        double score = weightedEvaluator.evaluate(board, player, opponent);
        
        // Expected score based on the weights
        double expectedScore = Constants.BOARD_WEIGHTS[0][0] - Constants.BOARD_WEIGHTS[1][1];
        assertEquals(expectedScore, score, "WeightedEvaluator should use board weights");
        assertTrue(score > 0, "Player should have positive score due to better position");
    }
    
    @Test
    public void testMultipleEvaluators() {
        // Test that multiple evaluators can be used interchangeably
        BoardEvaluator simpleEvaluator = new TestEvaluator();
        BoardEvaluator weightedEvaluator = new WeightedEvaluator();
        
        // Set up a board with more player pieces but in worse positions
        board[1][1].setType(BoardSpace.SpaceType.BLACK); // Player in bad position (-100)
        board[1][2].setType(BoardSpace.SpaceType.BLACK); // Player in bad position (-10)
        board[2][1].setType(BoardSpace.SpaceType.BLACK); // Player in bad position (-10)
        
        board[0][0].setType(BoardSpace.SpaceType.WHITE); // Opponent in corner (high value: 200)
        
        // Simple evaluator just counts pieces
        double simpleScore = simpleEvaluator.evaluate(board, player, opponent);
        assertEquals(2.0, simpleScore, "Simple evaluator should count pieces: 3 player - 1 opponent = 2");
        
        // Weighted evaluator takes position values into account
        double weightedScore = weightedEvaluator.evaluate(board, player, opponent);
        
        // Expected weighted score should be negative because the opponent's corner is worth more
        // than the player's three pieces in bad positions
        double expectedWeighted = Constants.BOARD_WEIGHTS[1][1] + Constants.BOARD_WEIGHTS[1][2] + 
                                 Constants.BOARD_WEIGHTS[2][1] - Constants.BOARD_WEIGHTS[0][0];
        assertEquals(expectedWeighted, weightedScore, "Weighted evaluator should use position values");
        
        // The sign of the scores should be different
        assertTrue(simpleScore > 0 && weightedScore < 0, 
                "Simple evaluator should be positive, weighted evaluator should be negative");
    }
}