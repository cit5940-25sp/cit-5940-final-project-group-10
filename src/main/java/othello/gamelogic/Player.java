package othello.gamelogic;

import java.util.*;

/**
 * Abstract Player class for representing a player within the game.
 * All types of Players have a color and a set of owned spaces on the game board.
 */
public abstract class Player {
    private final List<BoardSpace> playerOwnedSpaces = new ArrayList<>();
    public List<BoardSpace> getPlayerOwnedSpacesSpaces() {
        return playerOwnedSpaces;
    }

    private BoardSpace.SpaceType color;
    public void setColor(BoardSpace.SpaceType color) {
        this.color = color;
    }
    public BoardSpace.SpaceType getColor() {
        return color;
    }

    /**
     * PART 1
     * TODO: Implement this method
     * Gets the available moves for this player given a certain board state.
     * This method will find destinations, empty spaces that are valid moves,
     * and map them to a list of origins that can traverse to those destinations.
     * @param board the board that will be evaluated for possible moves for this player
     * @return a map with a destination BoardSpace mapped to a List of origin BoardSpaces.
     */
    // IDEA 1 : Use DFS to find the empty spaces adjacent of the contiguous pieces
    private void edgeTrace(int row, int col, BoardSpace[][] board, boolean[][] visited) {
        // base case, move is outside grid
        if (row >= board.length || row < 0 || col < 0 || col >= board[0].length) {
            return;
        }

        // base case, already visited
        if (visited[row][col]) {
            return;
        }

        visited[row][col] = true;

        // found empty space, check if it is a valid move
        if (board[row][col].getType() == BoardSpace.SpaceType.EMPTY) {
            validCheck(row, col, board);
        }

        // traverse right and left
        edgeTrace(row - 1, col, board, visited);
        edgeTrace(row + 1, col, board, visited);

        //traverse up and down
        edgeTrace(row, col + 1, board, visited);
        edgeTrace(row, col - 1, board, visited);
    }

    private void validCheck(int row, int col, BoardSpace[][] board) {
        int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (int[] direction : directions) {
            // if any of the adjacent four cells to an empty is the other player's color
            // begin looking for a cell of player's own color in the spaces along the same axis
            int rowCheck = row + direction[0];
            int colCheck = col + direction[1];
            if (rowCheck >= board.length || rowCheck < 0 || colCheck < 0 || colCheck >= board[0].length) {
                continue;
            }

            BoardSpace nextSpace = board[row][col];
            BoardSpace.SpaceType nextType = nextSpace.getType();
            while (nextType != BoardSpace.SpaceType.EMPTY && nextType != color) {
                // while the space is the opposing players color, check along the axis
                rowCheck += direction[0];
                colCheck += direction[1];

            }
        }
    }

    public Map<BoardSpace, List<BoardSpace>> getAvailableMoves(BoardSpace[][] board) {
//        int[][] directions = {{0,1}, {0,-1}, {1,0}, {-1,0}};
//
//        Map<BoardSpace, List<BoardSpace>> validMoves = new HashMap<>();
//        Queue<BoardSpace> sourcePieces = new ArrayDeque<>();
//
//        // get player owned spaces
//        for (BoardSpace[] space_row : board) {
//            for (BoardSpace space : space_row) {
//                if (space.getType() == color) {
//                    // populate list of owned spaces and BFS queue
//                    playerOwnedSpaces.add(space);
//                    sourcePieces.add(space);
//                }
//            }
//        }
//        // BFS using player spaces
//        boolean[][] visited = new boolean[board.length][board[0].length];
//        while (!sourcePieces.isEmpty()) {
//            BoardSpace begin = sourcePieces.poll();
//            for (int[] direction : directions) {
//                int row = begin.getX() + direction[0];
//                int col = begin.getY() + direction[1];
//                BoardSpace next;
//
//                // if the piece is
//                if (row >= 0 && row < board.length && col >= 0 && col < board.length
//                        && !visited[row][col]
//                        && board[row][col]) {
//                    visited[row][col] = true;
//                    }
//                }
//
//
//
//            }
//        }
        return null;
    }

}
