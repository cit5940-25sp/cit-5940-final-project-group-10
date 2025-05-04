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

    public Map<BoardSpace, List<BoardSpace>> getAvailableMoves(BoardSpace[][] board) {
        // init, get the player color, uses ternary (?) to define the other player
        Map<BoardSpace, List<BoardSpace>> moves = new HashMap<>();
        BoardSpace.SpaceType myPlayer = getColor();
        BoardSpace.SpaceType opponent = (myPlayer == BoardSpace.SpaceType.BLACK)
                ? BoardSpace.SpaceType.WHITE
                : BoardSpace.SpaceType.BLACK;
        int rows = board.length;
        int cols = board[0].length;
        // sequence of xy coordinate dimensions to travel along
        int[] dx = {-1, -1, -1, 0, 0, 1, 1, 1};
        int[] dy = {-1,  0,  1,-1, 1,-1, 0, 1};
        // go row by row
        for (int x = 0; x < rows; x++) {
            for (int y = 0; y < cols; y++) {
                // skip non-empty spaces, not valid moves
                if (board[x][y].getType() != BoardSpace.SpaceType.EMPTY) {
                    continue;
                }

                List<BoardSpace> origins = new ArrayList<>();
                // check all possible spaces for opponent's color
                for (int direction = 0; direction < 8; direction++) {
                    int i = x + dx[direction];
                    int j = y + dy[direction];
                    boolean hasOpponent = false;
                    // if we find an opponents color, keep traversing along that dimension dx,dy
                    while (OthelloGame.inBounds(i,j)
                            && board[i][j].getType() == opponent) {
                        hasOpponent = true;
                        i += dx[direction];
                        j += dy[direction];
                    }
                    // if we encountered some amount of opponent pieces and then one of our own
                    if (hasOpponent
                            && OthelloGame.inBounds(i,j)
                            && board[i][j].getType() == myPlayer) {
                        // add this player piece as an origin for a move
                        origins.add(board[i][j]);
                    }
                }

                // if we found any possible origins for the move, it is valid
                if (!origins.isEmpty()) {
                    moves.put(board[x][y], origins);
                }
            }
        }
        return moves;
    }

}
