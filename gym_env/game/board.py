import itertools
from collections import defaultdict

import numpy as np

from gym_env.game.buildings import Empty


class Board:
    def __init__(self, grid_size, game):
        self.grid_size = grid_size
        self.game = game
        self.type_to_id = {t: i for i, t in enumerate(game.building_types)}
        # shared empty field
        self.empty_field = Empty(None, self)
        self.grid = None
        self.reset_grid()

        self.one_hot_dim = game.n_players * len(game.building_types)
        self._one_hots = np.eye(self.one_hot_dim)

    def get_piece(self, coordinates):
        return self.grid[coordinates]

    def place_piece(self, piece, coordinates):
        assert self.is_legal_position(coordinates), 'the coordinates provided are off the grid'
        # assert self.grid[coordinates] is self.empty_field, 'can only place piece on empty field'
        self.grid[coordinates] = piece

    def reset_grid(self):
        self.grid = defaultdict(lambda: self.empty_field)

    def is_legal_position(self, position):
        """Check whether position is a legal position on the board.

        :param position: the position to check
        :return: bool
        """
        for i, x in enumerate(position):
            if x < 0 or x >= self.grid_size[i]:
                return False

        return True

    def piece_to_one_hot(self, piece):
        return self._one_hots[self.type_to_id[type(piece)]]

    def to_one_hot(self):
        one_hot_grid = np.zeros((*self.grid_size, self.one_hot_dim))
        ranges = (range(x) for x in self.grid_size)
        for vec in itertools.product(*ranges):
            one_hot_grid[vec] = self.piece_to_one_hot(self.grid[vec])

        return one_hot_grid
