import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import math
import concurrent.futures
import signal
import sys
import h5py

## convert chess notation to 64x1 vector indx
def chess_to_1d_index(move):
    col = move[0].lower()
    row = move[1]
    col_index = ord(col) - ord('a')
    row_index = int(row) - 1
    return row_index * 8 + col_index

def chess_to_2d_index(move):
    col = move[0].lower()
    row = move[1]
    col_index = ord(col) - ord('a')
    row_index = int(row) - 1
    return (row_index, col_index)

def print_board(arr):
    print("    a b c d e f g h")
    print("  +-----------------+")
    for i in range(8):
        row = arr[i * 8:(i + 1) * 8]
        row_str = " ".join(["B" if cell == 1 else "W" if cell == -1 else "X" if cell == 99 else "." for cell in row])
        print(f"{8 - i} | {row_str} |")


def get_legal_moves(board_vec, black_player):
    directions = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],         [0, 1],
        [1, -1], [1, 0], [1, 1]
    ])

    current_player = 1 if black_player else -1
    legal_moves = {}

    # Find all opponent pieces
    opponent = -current_player
    opponent_indices = np.where(board_vec == opponent)[0]
    if opponent_indices.size == 0:
        return {}

    # Check each empty cell
    empty_indices = np.where(board_vec == 0)[0]
    for idx in empty_indices:
        row, col = divmod(idx, 8)
        for dx, dy in directions:
            x, y = row + dx, col + dy
            found_opponent = False
            while 0 <= x < 8 and 0 <= y < 8:
                neighbor_index = x * 8 + y
                piece = board_vec[neighbor_index]
                if piece == opponent:
                    found_opponent = True
                elif piece == current_player and found_opponent:
                    origin_piece = x * 8 + y
                    legal_moves.setdefault(idx, []).append(origin_piece)
                    break
                else:
                    break
                x += dx
                y += dy

    return legal_moves

def take_spaces(move, origin, board_vec, black_player):
    dest_row, dest_col = divmod(move, 8)
    origin_row, origin_col = divmod(origin, 8)

    player = 1 if black_player else -1
    # Calculate direction explicitly
    dx = (origin_row - dest_row) // abs(origin_row - dest_row) if origin_row != dest_row else 0
    dy = (origin_col - dest_col) // abs(origin_col - dest_col) if origin_col != dest_col else 0

    # Move one step towards the origin to avoid skipping the first piece
    x, y = dest_row, dest_col

    # Flip all pieces until the origin
    while x != (origin_row) or y != (origin_col):
        if 0 <= x < 8 and 0 <= y < 8:
            index = x * 8 + y
            if board_vec[index] != player:
                board_vec[index] = player
        else:
            break
        if x != origin_row:
            x += dx
        if y != origin_col:
            y += dy


## convert gamestring into training samples
def game_to_samples(movestring : str, outcome : int):
    sample_set = []
    board_vec = np.zeros(64, dtype=int)
    board_vec[27] = -1
    board_vec[28] = 1
    board_vec[35] = 1
    board_vec[36] = -1
    black_player = True


    for i in range(0, len(movestring), 2):
        move = movestring[i:i+2]
        vec_index = chess_to_1d_index(move)
        map = get_legal_moves(board_vec, black_player)
        if not map:
            black_player = not black_player
            continue

        origins = map.get(vec_index)
        
        if not origins:
            print("---- Bad move ----")
            print(f"Move : {move}")
            print(f"1d index : {vec_index}")
            print(f"2d index : {divmod(vec_index, 8)}")
            print(f"current player : {'B' if black_player else 'W'}")
            print(f"Valid moves : {map}")
            print("------------------")

            X_board = board_vec.copy()
            X_board[vec_index] = 99
            board_vec = X_board
            break

        ## next move probabilities as calculated by MCTS
        move_probabilities = mcts_probabilities(board_vec, black_player)

        ## append tuple of current board, MCTS move probabilites, game outcome
        sample_set.append((board_vec.copy(), move_probabilities, outcome))

        ## update board with move just made
        for origin in origins:
            take_spaces(vec_index, origin, board_vec, black_player)
        
        black_player = not black_player
        
    
    black = np.sum(board_vec == 1)
    white = np.sum(board_vec == -1)

    # print(f"Black : {black}\nWhite: {white}\nWinner : {outcome}")

    print_board(board_vec)
    print("  +-----------------+")
    return sample_set
        
class MCTSNode:
    def __init__(self, board_vec, player : bool, parent=None, move=None):
        self.board_vec = board_vec.copy()
        self.player = player
        self.parent = parent
        self.move = move
        self.visits = 0
        self.wins = 0
        self.children = []
        self.origin_map = get_legal_moves(self.board_vec, self.player)
        self.move_list = list(self.origin_map.keys())

    def is_fully_expanded(self):
        return len(self.move_list) == 0

    def best_child(self, c=math.sqrt(2)):
        return max(self.children, key=lambda child: (child.wins / child.visits) + 
                   c * math.sqrt(math.log(self.visits) / child.visits))

    def expand(self):
        move = self.move_list.pop(random.randint(0, len(self.move_list) - 1))
        move_origins = self.origin_map.get(move)
        new_board_vec = self.board_vec.copy()
        for origin in move_origins:
            take_spaces(move, origin, new_board_vec, self.player)
        child_node = MCTSNode(new_board_vec, player = not self.player, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result

    def backpropagate(self, result):
        node = self
        while node is not None:
            node.update(result)
            node = node.parent

def mcts_probabilities(board, player, num_simulations=1000):
    root = MCTSNode(board, player)

    for _ in range(num_simulations):
        # Selection
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        # Expansion
        if not node.is_fully_expanded():
            node = node.expand()

        # Simulation
        current_board = node.board_vec.copy()
        current_player = 1 if node.player else -1
        empty_indices = np.where(current_board == 0)[0]

        # Randomly simulate moves until the board is full
        np.random.shuffle(empty_indices)
        moves_to_fill = len(empty_indices)
        current_board[empty_indices[:moves_to_fill // 2]] = current_player
        current_board[empty_indices[moves_to_fill // 2:]] = -current_player

        # Backpropagation
        winner = 1 if np.sum(current_board == 1) > np.sum(current_board == -1) else -1
        result = 1 if winner == (current_player) else 0
        node.backpropagate(result)

    # Convert visit counts to probabilities
    move_probs = np.zeros(64)
    for child in root.children:
        move_probs[child.move] = child.visits / root.visits

    return move_probs



def process_game(args):
    movestring, outcome = args
    return game_to_samples(movestring, outcome)

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

## main processing loop
if __name__ == "__main__":
    raw_df = pd.read_csv('othello_dataset.csv', header=None)
  
    samples = []
    count = 1
    
    game_args = [(row[2], row[1]) for row in raw_df.itertuples(index=False, name=None)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
            # Set up the pool with a SIGINT-safe initializer
        executor = concurrent.futures.ProcessPoolExecutor(initializer=init_worker)
        try:
            for sample_set in tqdm(executor.map(process_game, game_args), total=len(game_args)):
                samples.extend(sample_set)
        except KeyboardInterrupt:
            print("\nReceived interrupt. Shutting down workers...")
            executor.shutdown(cancel_futures=True)
            sys.exit(1)
        else:
            executor.shutdown()

    with h5py.File("training_samples.h5", "w") as f:
        for i, (board, probs, outcome) in enumerate(samples):
            group = f.create_group(f"sample_{i}")
            group.create_dataset("board", data=board, compression="gzip")
            group.create_dataset("probs", data=probs, compression="gzip")
            group.attrs["outcome"] = outcome

    
