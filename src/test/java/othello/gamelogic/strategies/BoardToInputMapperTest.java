package othello.gamelogic.strategies;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import othello.gamelogic.BoardSpace;
import othello.gamelogic.OthelloGame;
import othello.gamelogic.Player;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the BoardToInputMapper class which converts Othello board states
 * to neural network inputs.
 */
public class BoardToInputMapperTest {

    private BoardSpace[][] board;
    private TestPlayer player;
    private int boardSize;
    
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
        boardSize = OthelloGame.GAME_BOARD_SIZE;
        board = new BoardSpace[boardSize][boardSize];
        
        // Initialize an empty board
        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                board[i][j] = new BoardSpace(i, j, BoardSpace.SpaceType.EMPTY);
            }
        }
        
        // Create player
        player = new TestPlayer(BoardSpace.SpaceType.BLACK);
    }

    @Test
    public void testMapToInputEmptyBoard() {
        // Test with completely empty board
        double[] input = BoardToInputMapper.mapToInput(board, player);
        
        // Expected size is boardSize * boardSize * 3 (channels)
        assertEquals(boardSize * boardSize * 3, input.length, "Input should have correct length");
        
        // Check that all values in empty channel are 1 and others are 0
        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                int playerIndex = i * boardSize + j;
                int opponentIndex = boardSize * boardSize + i * boardSize + j;
                int emptyIndex = 2 * boardSize * boardSize + i * boardSize + j;
                
                assertEquals(0.0, input[playerIndex], "Player channel should be 0 for empty space");
                assertEquals(0.0, input[opponentIndex], "Opponent channel should be 0 for empty space");
                assertEquals(1.0, input[emptyIndex], "Empty channel should be 1 for empty space");
            }
        }
    }

    @Test
    public void testMapToInputWithPieces() {
        // Place some pieces on the board
        board[3][3].setType(BoardSpace.SpaceType.WHITE); // Opponent piece
        board[3][4].setType(BoardSpace.SpaceType.BLACK); // Player piece
        board[4][3].setType(BoardSpace.SpaceType.BLACK); // Player piece
        board[4][4].setType(BoardSpace.SpaceType.WHITE); // Opponent piece
        
        double[] input = BoardToInputMapper.mapToInput(board, player);
        
        // Check player pieces
        int playerIndex1 = 3 * boardSize + 4;
        int playerIndex2 = 4 * boardSize + 3;
        assertEquals(1.0, input[playerIndex1], "Player channel should be 1 for player piece");
        assertEquals(1.0, input[playerIndex2], "Player channel should be 1 for player piece");
        
        // Check opponent pieces
        int opponentIndex1 = boardSize * boardSize + 3 * boardSize + 3;
        int opponentIndex2 = boardSize * boardSize + 4 * boardSize + 4;
        assertEquals(1.0, input[opponentIndex1], "Opponent channel should be 1 for opponent piece");
        assertEquals(1.0, input[opponentIndex2], "Opponent channel should be 1 for opponent piece");
        
        // Check empty channel for these positions
        int emptyIndex1 = 2 * boardSize * boardSize + 3 * boardSize + 4;
        int emptyIndex2 = 2 * boardSize * boardSize + 4 * boardSize + 3;
        assertEquals(0.0, input[emptyIndex1], "Empty channel should be 0 for player piece");
        assertEquals(0.0, input[emptyIndex2], "Empty channel should be 0 for player piece");
    }

    @Test
    public void testMapToInputWithWhitePlayer() {
        // Test with player as WHITE
        player = new TestPlayer(BoardSpace.SpaceType.WHITE);
        
        // Setup some pieces
        board[3][3].setType(BoardSpace.SpaceType.WHITE); // Player piece
        board[3][4].setType(BoardSpace.SpaceType.BLACK); // Opponent piece
        
        double[] input = BoardToInputMapper.mapToInput(board, player);
        
        // Check player piece (WHITE)
        int playerIndex = 3 * boardSize + 3;
        assertEquals(1.0, input[playerIndex], "Player channel should be 1 for player piece");
        
        // Check opponent piece (BLACK)
        int opponentIndex = boardSize * boardSize + 3 * boardSize + 4;
        assertEquals(1.0, input[opponentIndex], "Opponent channel should be 1 for opponent piece");
    }

    @Test
    public void testMapToMove() {
        // Create output array with probabilities for each position
        double[] output = new double[boardSize * boardSize];
        
        // Set highest probability for position (2,3)
        int bestMoveIndex = 2 * boardSize + 3;
        output[bestMoveIndex] = 0.9;
        
        // Set other probabilities
        output[1 * boardSize + 2] = 0.5;
        output[3 * boardSize + 4] = 0.3;
        
        // Create map of available moves
        Map<BoardSpace, List<BoardSpace>> availableMoves = new HashMap<>();
        
        // Add moves to the map
        BoardSpace move1 = new BoardSpace(2, 3, BoardSpace.SpaceType.EMPTY);
        BoardSpace move2 = new BoardSpace(1, 2, BoardSpace.SpaceType.EMPTY);
        BoardSpace move3 = new BoardSpace(3, 4, BoardSpace.SpaceType.EMPTY);
        
        availableMoves.put(move1, List.of(new BoardSpace(2, 4, BoardSpace.SpaceType.BLACK)));
        availableMoves.put(move2, List.of(new BoardSpace(1, 3, BoardSpace.SpaceType.BLACK)));
        availableMoves.put(move3, List.of(new BoardSpace(3, 5, BoardSpace.SpaceType.BLACK)));
        
        // Test map to move
        BoardSpace selected = BoardToInputMapper.mapToMove(output, availableMoves);
        
        // Verify highest probability move was selected
        assertEquals(2, selected.getX(), "Should select move with highest probability (X)");
        assertEquals(3, selected.getY(), "Should select move with highest probability (Y)");
    }

    @Test
    public void testMapToMoveWithUnavailableHighestProbability() {
        // Create output array with probabilities for each position
        double[] output = new double[boardSize * boardSize];
        
        // Set highest probability for position (2,3) - NOT in available moves
        output[2 * boardSize + 3] = 0.9;
        
        // Set probabilities for available moves
        output[1 * boardSize + 2] = 0.5; // This should be selected as highest available
        output[3 * boardSize + 4] = 0.3;
        
        // Create map of available moves (excluding the highest probability move)
        Map<BoardSpace, List<BoardSpace>> availableMoves = new HashMap<>();
        
        // Add moves to the map
        BoardSpace move1 = new BoardSpace(1, 2, BoardSpace.SpaceType.EMPTY);
        BoardSpace move2 = new BoardSpace(3, 4, BoardSpace.SpaceType.EMPTY);
        
        availableMoves.put(move1, List.of(new BoardSpace(1, 3, BoardSpace.SpaceType.BLACK)));
        availableMoves.put(move2, List.of(new BoardSpace(3, 5, BoardSpace.SpaceType.BLACK)));
        
        // Test map to move
        BoardSpace selected = BoardToInputMapper.mapToMove(output, availableMoves);
        
        // Verify best available move was selected
        assertEquals(1, selected.getX(), "Should select move with highest available probability (X)");
        assertEquals(2, selected.getY(), "Should select move with highest available probability (Y)");
    }
}