package othello.gamelogic.strategies;

import othello.gamelogic.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Utility classes for testing the strategy implementations.
 */
public class TestUtils {

    /**
     * Test implementation of OthelloGame for testing
     */
    public static class TestOthelloGame extends OthelloGame {
        private BoardSpace[][] board;
        private Map<BoardSpace, List<BoardSpace>> availableMoves;
        
        public TestOthelloGame(BoardSpace[][] board) {
            super(null, null); // Pass null for players since we'll override methods
            this.board = board;
            this.availableMoves = new HashMap<>();
        }
        
        public void setAvailableMoves(Map<BoardSpace, List<BoardSpace>> moves) {
            this.availableMoves = moves;
        }
        
        @Override
        public BoardSpace[][] getBoard() {
            return board;
        }
        
        @Override
        public Map<BoardSpace, List<BoardSpace>> getAvailableMoves(Player player) {
            return availableMoves;
        }
    }

    /**
     * Test implementation of Player for testing
     */
    public static class TestPlayer extends Player {
        private Map<BoardSpace, List<BoardSpace>> availableMoves;

        public TestPlayer(BoardSpace.SpaceType color) {
            this.availableMoves = new HashMap<>();
            this.setColor(color);
        }
        
        public void setAvailableMoves(Map<BoardSpace, List<BoardSpace>> moves) {
            this.availableMoves = moves;
        }
        
        @Override
        public Map<BoardSpace, List<BoardSpace>> getAvailableMoves(BoardSpace[][] board) {
            return availableMoves;
        }
    }
    
    /**
     * Creates a standard 8x8 Othello board with the initial setup
     */
    public static BoardSpace[][] createStandardBoard() {
        BoardSpace[][] board = new BoardSpace[8][8];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                board[i][j] = new BoardSpace(i, j, BoardSpace.SpaceType.EMPTY);
            }
        }
        
        // Set up standard starting position
        board[3][3].setType(BoardSpace.SpaceType.WHITE);
        board[4][4].setType(BoardSpace.SpaceType.WHITE);
        board[3][4].setType(BoardSpace.SpaceType.BLACK);
        board[4][3].setType(BoardSpace.SpaceType.BLACK);
        
        return board;
    }
}
