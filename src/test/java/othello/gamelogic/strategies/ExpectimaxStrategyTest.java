package othello.gamelogic.strategies;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import othello.gamelogic.*;
import othello.gamelogic.strategies.TestUtils.TestOthelloGame;
import othello.gamelogic.strategies.TestUtils.TestPlayer;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test for the ExpectimaxStrategy class.
 */
public class ExpectimaxStrategyTest {

    private ExpectimaxStrategy expectimaxStrategy;
    private TestOthelloGame testGame;
    private TestPlayer testCurrentPlayer;
    private TestPlayer testOpponent;
    private BoardSpace[][] testBoard;

    @BeforeEach
    public void setUp() {
        // Initialize the strategy
        expectimaxStrategy = new ExpectimaxStrategy();
        
        // Create test objects
        testCurrentPlayer = new TestPlayer(BoardSpace.SpaceType.BLACK);
        testOpponent = new TestPlayer(BoardSpace.SpaceType.WHITE);
        
        // Create a test board with standard setup
        testBoard = TestUtils.createStandardBoard();
        
        // Create the test game
        testGame = new TestOthelloGame(testBoard);
    }

    @Test
    public void testGetBestMoveWithNoAvailableMoves() {
        // Set up the game to return no available moves
        Map<BoardSpace, List<BoardSpace>> emptyMoves = new HashMap<>();
        testGame.setAvailableMoves(emptyMoves);
        
        // Test that the strategy returns null when there are no available moves
        BoardSpace result = expectimaxStrategy.getBestMove(testGame, testCurrentPlayer, testOpponent);
        
        assertNull(result, "Should return null when there are no available moves");
    }

    @Test
    public void testGetBestMoveWithAvailableMoves() {
        // Create a map of available moves
        Map<BoardSpace, List<BoardSpace>> availableMoves = new HashMap<>();
        BoardSpace move1 = new BoardSpace(2, 3, BoardSpace.SpaceType.EMPTY);
        BoardSpace move2 = new BoardSpace(3, 2, BoardSpace.SpaceType.EMPTY);
        
        List<BoardSpace> flippedPieces1 = new ArrayList<>();
        flippedPieces1.add(new BoardSpace(3, 3, BoardSpace.SpaceType.WHITE));
        
        List<BoardSpace> flippedPieces2 = new ArrayList<>();
        flippedPieces2.add(new BoardSpace(3, 3, BoardSpace.SpaceType.WHITE));
        
        availableMoves.put(move1, flippedPieces1);
        availableMoves.put(move2, flippedPieces2);
        
        // Set up the test game to return the available moves
        testGame.setAvailableMoves(availableMoves);
        
        // Test that the strategy returns a valid move
        BoardSpace result = expectimaxStrategy.getBestMove(testGame, testCurrentPlayer, testOpponent);
        
        // The result should be one of the available moves
        boolean isValidMove = result != null && 
                             (result.equals(move1) || result.equals(move2));
        
        assertTrue(isValidMove, "Should return one of the available moves");
    }

    @Test
    public void testGetBestMoveConsistency() {
        // Create a map of available moves
        Map<BoardSpace, List<BoardSpace>> availableMoves = new HashMap<>();
        BoardSpace move1 = new BoardSpace(2, 3, BoardSpace.SpaceType.EMPTY);
        
        List<BoardSpace> flippedPieces1 = new ArrayList<>();
        flippedPieces1.add(new BoardSpace(3, 3, BoardSpace.SpaceType.WHITE));
        
        availableMoves.put(move1, flippedPieces1);
        
        // Set up the test game to return the available moves
        testGame.setAvailableMoves(availableMoves);
        
        // When there's only one available move, the strategy should consistently return it
        BoardSpace result = expectimaxStrategy.getBestMove(testGame, testCurrentPlayer, testOpponent);
        
        assertEquals(move1, result, "Should consistently return the only available move");
    }
    
    @Test
    public void testExpectimaxVsMinimaxBehavior() {
        // Create a specific board state where Expectimax might behave differently than Minimax
        // In Expectimax, the chance nodes (opponent's moves) are treated as having equal probability
        
        // Create a map of available moves for the current player
        Map<BoardSpace, List<BoardSpace>> availableMoves = new HashMap<>();
        BoardSpace move1 = new BoardSpace(2, 3, BoardSpace.SpaceType.EMPTY);
        BoardSpace move2 = new BoardSpace(3, 2, BoardSpace.SpaceType.EMPTY);
        
        List<BoardSpace> flippedPieces1 = new ArrayList<>();
        flippedPieces1.add(new BoardSpace(3, 3, BoardSpace.SpaceType.WHITE));
        
        List<BoardSpace> flippedPieces2 = new ArrayList<>();
        flippedPieces2.add(new BoardSpace(3, 3, BoardSpace.SpaceType.WHITE));
        
        availableMoves.put(move1, flippedPieces1);
        availableMoves.put(move2, flippedPieces2);
        
        // Set up the test game to return the available moves
        testGame.setAvailableMoves(availableMoves);
        
        // Set up opponent's available moves
        Map<BoardSpace, List<BoardSpace>> opponentMoves = new HashMap<>();
        BoardSpace opponentMove = new BoardSpace(1, 2, BoardSpace.SpaceType.EMPTY);
        List<BoardSpace> opponentFlippedPieces = new ArrayList<>();
        opponentFlippedPieces.add(new BoardSpace(2, 3, BoardSpace.SpaceType.BLACK));
        opponentMoves.put(opponentMove, opponentFlippedPieces);
        testOpponent.setAvailableMoves(opponentMoves);
        
        // Test that the strategy returns a valid move
        BoardSpace result = expectimaxStrategy.getBestMove(testGame, testCurrentPlayer, testOpponent);
        
        // The result should be one of the available moves
        boolean isValidMove = result != null && 
                             (result.equals(move1) || result.equals(move2));
        
        assertTrue(isValidMove, "Should return one of the available moves");
    }
    
    @Test
    public void testExpectimaxWithDifferentEvaluationScores() {
        // Create a map of available moves
        Map<BoardSpace, List<BoardSpace>> availableMoves = new HashMap<>();
        BoardSpace move1 = new BoardSpace(2, 3, BoardSpace.SpaceType.EMPTY);
        BoardSpace move2 = new BoardSpace(3, 2, BoardSpace.SpaceType.EMPTY);
        
        List<BoardSpace> flippedPieces1 = new ArrayList<>();
        flippedPieces1.add(new BoardSpace(3, 3, BoardSpace.SpaceType.WHITE));
        
        List<BoardSpace> flippedPieces2 = new ArrayList<>();
        flippedPieces2.add(new BoardSpace(3, 3, BoardSpace.SpaceType.WHITE));
        
        availableMoves.put(move1, flippedPieces1);
        availableMoves.put(move2, flippedPieces2);
        
        // Set up the test game to return the available moves
        testGame.setAvailableMoves(availableMoves);
        
        // Run the test
        BoardSpace result = expectimaxStrategy.getBestMove(testGame, testCurrentPlayer, testOpponent);
        
        // Verify that a valid move is returned
        boolean isValidMove = result != null && 
                             (result.equals(move1) || result.equals(move2));
        
        assertTrue(isValidMove, "Should return one of the available moves");
    }
}
