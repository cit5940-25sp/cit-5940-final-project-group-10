package othello.gamelogic;

import java.util.*;

/**
 * Models a board of Othello.
 * Includes methods to get available moves and take spaces.
 */
public class OthelloGame {
    public static final int GAME_BOARD_SIZE = 8;

    private BoardSpace[][] board;
    private final Player playerOne;
    private final Player playerTwo;

    public OthelloGame(Player playerOne, Player playerTwo) {
        this.playerOne = playerOne;
        this.playerTwo = playerTwo;
        initBoard();
    }

    public BoardSpace[][] getBoard() {
        return board;
    }

    public Player getPlayerOne() {
        return playerOne;
    }

    public Player getPlayerTwo() {
        return  playerTwo;
    }

    /**
     * Returns the available moves for a player.
     * Used by the GUI to get available moves each turn.
     * @param player player to get moves for
     * @return the map of available moves,that maps destination to list of origins
     */
    public Map<BoardSpace, List<BoardSpace>> getAvailableMoves(Player player) {
        return player.getAvailableMoves(board);
    }

    /**
     * Initializes the board at the start of the game with all EMPTY spaces.
     */
    public void initBoard() {
        board = new BoardSpace[GAME_BOARD_SIZE][GAME_BOARD_SIZE];
        for (int i = 0; i < GAME_BOARD_SIZE; i++) {
            for (int j = 0; j < GAME_BOARD_SIZE; j++) {
                board[i][j] = new BoardSpace(i, j, BoardSpace.SpaceType.EMPTY);
            }
        }
        //p1 is black
        board[4][3].setType(BoardSpace.SpaceType.BLACK);
        playerOne.getPlayerOwnedSpaces().add(board[4][3]);
        board[3][4].setType(BoardSpace.SpaceType.BLACK);
        playerOne.getPlayerOwnedSpaces().add(board[3][4]);

        //p2 is while
        board[3][3].setType(BoardSpace.SpaceType.WHITE);
        playerTwo.getPlayerOwnedSpaces().add(board[3][3]);
        board[4][4].setType(BoardSpace.SpaceType.WHITE);
        playerTwo.getPlayerOwnedSpaces().add(board[4][4]);
    }

    // useful for less repetitive bounds checking booleans
    public static boolean inBounds(int row, int col) {
        return (
                row >= 0 && row < GAME_BOARD_SIZE &&
                        col >= 0 && col < GAME_BOARD_SIZE
        );
    }

    /**
     * PART 1
     * TODO: Implement this method
     * Claims the specified space for the acting player.
     * Should also check if the space being taken is already owned by the acting player,
     * should not claim anything if acting player already owns space at (x,y)
     * @param actingPlayer the player that will claim the space at (x,y)
     * @param opponent the opposing player, will lose a space if their space is at (x,y)
     * @param x the x-coordinate of the space to claim
     * @param y the y-coordinate of the space to claim
     */
    public void takeSpace(Player actingPlayer, Player opponent, int x, int y) {
        BoardSpace space = board[x][y];
        if (!actingPlayer.getPlayerOwnedSpaces().contains(board[x][y])) {
            // update board state
            // take from opponent, give to player
            opponent.getPlayerOwnedSpaces().remove(space);

            if (!actingPlayer.getPlayerOwnedSpaces().contains(space)) {
                actingPlayer.getPlayerOwnedSpaces().add(board[x][y]);
            }

            space.setType(actingPlayer.getColor());
        }
    }

    /**
     * PART 1
     * TODO: Implement this method
     * Claims spaces from all origins that lead to a specified destination.
     * This is called when a player, human or computer, selects a valid destination.
     * @param actingPlayer the player that will claim spaces
     * @param opponent the opposing player, that may lose spaces
     * @param availableMoves map of the available moves, that maps destination to list of origins
     * @param selectedDestination the specific destination that a HUMAN player selected
     */
    public void takeSpaces(Player actingPlayer, Player opponent, Map<BoardSpace, List<BoardSpace>> availableMoves, BoardSpace selectedDestination) {
        // cardinal directions
        int[] dx = {-1, -1, -1, 0, 0, 1, 1, 1};
        int[] dy = {-1,  0,  1,-1, 1,-1, 0, 1};

        List<BoardSpace> origins = availableMoves.get(selectedDestination);

        // check which direction the origin is coming from
        for (BoardSpace origin : origins) {
            int xdist = origin.getX() - selectedDestination.getX();
            int ydist = origin.getY() - selectedDestination.getY();


            for (int direction = 0; direction < 8; direction++) {
                // look for direction with the same cardinality as the difference between dest and origin
                if (dx[direction] == Integer.signum(xdist) && dy[direction] == Integer.signum(ydist)) {
                    int i = selectedDestination.getX();
                    int j = selectedDestination.getY();

                    // fill spaces once found
                    while ((i != origin.getX() || j != origin.getY())) {
                        if (inBounds(i,j)) {
                            takeSpace(actingPlayer, opponent, i, j);
                        }
                       if (i != origin.getX()) {
                           i += dx[direction];
                       }
                       if (j != origin.getY()) {
                           j += dy[direction];
                       }
                    }
                }
            }
        }
    }

    /**
     * PART 2
     * TODO: Implement this method
     * Gets the computer decision for its turn.
     * Should call a method within the ComputerPlayer class that returns a BoardSpace using a specific strategy.
     * @param computer computer player that is deciding their move for their turn
     * @return the BoardSpace that was decided upon
     */
    public BoardSpace computerDecision(ComputerPlayer computer) {
        BoardSpace.SpaceType computerColor = computer.getColor();
        if (playerOne.getColor() == computerColor) {
            return computer.getBestMove(this, playerTwo);
        } else {
            return computer.getBestMove(this, playerOne);
        }
     }

}
