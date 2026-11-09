import numpy as np

# Constants
EMPTY = 0
BLACK = 1
WHITE = 2
SIZE = 19

class GoBoard:
    def __init__(self, size=SIZE):
        self.size = size
        # 0: Empty, 1: Black, 2: White
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.next_player = BLACK

    def reset(self):
        self.board.fill(EMPTY)
        self.next_player = BLACK

    def is_on_board(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def get_neighbors(self, x, y):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_on_board(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def get_group_and_liberties(self, x, y):
        """
        Get the group of stones connected to (x, y) and their liberties.
        Returns: (group_points_set, liberties_count)
        """
        color = self.board[x, y]
        if color == EMPTY:
            return set(), 0

        group = set()
        liberties = set()
        queue = [(x, y)]
        group.add((x, y))

        idx = 0
        while idx < len(queue):
            cx, cy = queue[idx]
            idx += 1
            
            for nx, ny in self.get_neighbors(cx, cy):
                neighbor_color = self.board[nx, ny]
                if neighbor_color == color:
                    if (nx, ny) not in group:
                        group.add((nx, ny))
                        queue.append((nx, ny))
                elif neighbor_color == EMPTY:
                    liberties.add((nx, ny))
        
        return group, len(liberties)

    def play(self, x, y, color):
        """
        Place a stone.
        x, y: 0-based coordinates
        color: 1 (Black) or 2 (White)
        Returns: number of captured stones, or -1 if invalid move.
        """
        if not self.is_on_board(x, y) or self.board[x, y] != EMPTY:
            return -1

        self.board[x, y] = color
        
        opponent = WHITE if color == BLACK else BLACK
        captured_count = 0
        
        # Check for captures
        neighbors = self.get_neighbors(x, y)
        checked_groups = set()

        for nx, ny in neighbors:
            if self.board[nx, ny] == opponent and (nx, ny) not in checked_groups:
                group, liberties = self.get_group_and_liberties(nx, ny)
                checked_groups.update(group)
                
                if liberties == 0:
                    # Capture
                    for gx, gy in group:
                        self.board[gx, gy] = EMPTY
                    captured_count += len(group)

        # Suicide check (usually forbidden, but we handle it gracefully)
        # In a real game engine, we might return error. 
        # For data replay, we assume moves are valid or standard rules (suicide forbidden unless it captures).
        if captured_count == 0:
            _, liberties = self.get_group_and_liberties(x, y)
            if liberties == 0:
                # Suicide move. In standard rules, this is invalid.
                # However, some datasets might contain weird moves or we might want to just allow it (stone dies immediately).
                # But typically we shouldn't see this in pro games.
                # We will keep the stone there (it has 0 liberties), assuming subsequent logic handles it or it's a valid "suicide" variant?
                # Actually, standard Go rules: suicide is forbidden.
                # Let's trust the dataset.
                pass

        self.next_player = opponent
        return captured_count

    def get_features(self):
        """
        Generate input features for the neural network.
        Shape: (3, SIZE, SIZE)
        Channel 0: Stones of current player (next_player)
        Channel 1: Stones of opponent
        Channel 2: All 1s (to help detect board edges) or Empty spots
        """
        features = np.zeros((3, self.size, self.size), dtype=np.float32)
        
        my_color = self.next_player
        opp_color = WHITE if my_color == BLACK else BLACK
        
        features[0] = (self.board == my_color).astype(np.float32)
        features[1] = (self.board == opp_color).astype(np.float32)
        features[2] = 1.0 # Bias plane
        
        return features

    def print_board(self):
        print("   " + " ".join([f"{i%10}" for i in range(self.size)]))
        for i in range(self.size):
            row_str = f"{i:2d} "
            for j in range(self.size):
                if self.board[i, j] == EMPTY:
                    row_str += ". "
                elif self.board[i, j] == BLACK:
                    row_str += "X "
                else:
                    row_str += "O "
            print(row_str)
