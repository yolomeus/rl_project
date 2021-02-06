from collections import defaultdict

import numpy as np


class Board:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.grid = defaultdict(lambda pos: None)

    def get_piece(self, coordinates):
        return self.grid[coordinates]

    def place_piece(self, piece, coordinates):
        assert self.grid[coordinates] is None, 'can only place piece on empty field'
        self.grid[coordinates] = piece

    def clear(self):
        self.grid = defaultdict(lambda pos: None)

    def to_binary(self):
        board = np.zeros(self.grid_size)
        return board
