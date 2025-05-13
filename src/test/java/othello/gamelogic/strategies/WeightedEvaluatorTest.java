package othello.gamelogic.strategies;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import othello.Constants;
import othello.gamelogic.BoardSpace;
import othello.gamelogic.Player;
import othello.gamelogic.OthelloGame;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the WeightedEvaluator class which evaluates Othello board positions
 * based on positional weights.
 */
public class WeightedEvaluatorTest {

    private WeightedEvaluator evaluator;
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

    @BeforeEach
    public void setUp() {
        evaluator = new WeightedEvaluator();
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
    public void testEmptyBoardEvaluation() {
        // An empty board should have a score of 0
        double score = evaluator.evaluate(board, player, opponent);
        assertEquals(0.0, score, "Empty board should have a score of 0");
    }

    @Test
    public void testCornerPositions() {
        // Test corner positions which have high weight
        board[0][0].setType(BoardSpace.SpaceType.BLACK); // Player in top-left corner
        double scoreWithOneCorner = evaluator.evaluate(board, player, opponent);
        assertEquals(Constants.BOARD_WEIGHTS[0][0], scoreWithOneCorner, 
                "Score should match corner weight");
        
        // Add opponent in another corner
        board[7][7].setType(BoardSpace.SpaceType.WHITE); // Opponent in bottom-right corner
        double scoreWithTwoCorners = evaluator.evaluate(board, player, opponent);
        assertEquals(Constants.BOARD_WEIGHTS[0][0] - Constants.BOARD_WEIGHTS[7][7], 
                scoreWithTwoCorners, "Score should account for both corners");
    }

    @Test
    public void testNegativePositions() {
        // Test positions with negative weights (bad positions)
        board[0][1].setType(BoardSpace.SpaceType.BLACK); // Player in a bad position
        double score = evaluator.evaluate(board, player, opponent);
        assertEquals(Constants.BOARD_WEIGHTS[0][1], score, 
                "Score should be negative for bad positions");
    }

    @Test
    public void testMixedPositions() {
        // Place player pieces in various positions
        board[0][0].setType(BoardSpace.SpaceType.BLACK); // Corner (high value)
        board[1][1].setType(BoardSpace.SpaceType.BLACK); // Bad position
        board[2][2].setType(BoardSpace.SpaceType.BLACK); // Neutral position
        
        // Place opponent pieces
        board[7][7].setType(BoardSpace.SpaceType.WHITE); // Corner (high value)
        board[6][6].setType(BoardSpace.SpaceType.WHITE); // Bad position
        
        double expectedScore = 
            Constants.BOARD_WEIGHTS[0][0] + 
            Constants.BOARD_WEIGHTS[1][1] + 
            Constants.BOARD_WEIGHTS[2][2] - 
            Constants.BOARD_WEIGHTS[7][7] - 
            Constants.BOARD_WEIGHTS[6][6];
        
        double actualScore = evaluator.evaluate(board, player, opponent);
        assertEquals(expectedScore, actualScore, 
                "Score should correctly sum weights for all pieces");
    }

    @Test
    public void testFullBoard() {
        // Setup a full board with alternating pieces
        double expectedScore = 0;
        
        for (int i = 0; i < OthelloGame.GAME_BOARD_SIZE; i++) {
            for (int j = 0; j < OthelloGame.GAME_BOARD_SIZE; j++) {
                if ((i + j) % 2 == 0) {
                    board[i][j].setType(BoardSpace.SpaceType.BLACK);
                    expectedScore += Constants.BOARD_WEIGHTS[i][j];
                } else {
                    board[i][j].setType(BoardSpace.SpaceType.WHITE);
                    expectedScore -= Constants.BOARD_WEIGHTS[i][j];
                }
            }
        }
        
        double actualScore = evaluator.evaluate(board, player, opponent);
        assertEquals(expectedScore, actualScore, "Full board evaluation should match expected weights");
    }
}