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
 * Test for the MCTSStrategy class.
 */
public class MCTSStrategyTest {

    private MCTSStrategy mctsStrategy;
    private TestOthelloGame testGame;
    private TestPlayer testCurrentPlayer;
    private TestPlayer testOpponent;
    private BoardSpace[][] testBoard;

    @BeforeEach
    public void setUp() {
        // Initialize the strategy
        mctsStrategy = new MCTSStrategy();
        
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
        BoardSpace result = mctsStrategy.getBestMove(testGame, testCurrentPlayer, testOpponent);
        
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
        BoardSpace result = mctsStrategy.getBestMove(testGame, testCurrentPlayer, testOpponent);
        
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
        BoardSpace result = mctsStrategy.getBestMove(testGame, testCurrentPlayer, testOpponent);
        
        assertEquals(move1, result, "Should consistently return the only available move");
    }
}
