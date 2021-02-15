import itertools
from collections import defaultdict

import numpy as np

from gym_env.game.pieces import Empty


class Board:
    def __init__(self, grid_size, game):
        self.grid_size = grid_size
        self.game = game
        self.type_to_id = {t: i for i, t in enumerate(game.piece_types)}
        # shared empty field
        self.empty_field = Empty(None, self)
        self.grid = None
        self.reset_grid()

        self.one_hot_dim = game.n_players * len(game.piece_types)
        self._one_hots = np.eye(self.one_hot_dim)

    def get_piece(self, coordinates):
        """Get the piece at the specified coordinates.

        :param coordinates: coordinate vector to access the board at.
        :return: piece or Empty piece.
        """
        if self.is_within_grid(coordinates):
            return self.grid[coordinates]
        return self.empty_field

    def place_piece(self, piece, coordinates):
        """Place a piece on the board at given coordinates.

        :param piece: the piece to place.
        :param coordinates: coordinate vector for where to place the piece.
        :return: whether the placing was a success or not
        :rtype: bool
        """
        if self.is_within_grid(coordinates) and not self.is_occupied(coordinates):
            self.grid[coordinates] = piece
            return True
        return False

    def reset_grid(self):
        self.grid = defaultdict(lambda: self.empty_field)

    def is_within_grid(self, position):
        """Check whether position is a legal position on the board.

        :param position: the position to check
        :return: bool
        """
        for i, x in enumerate(position):
            if x < 0 or x >= self.grid_size[i]:
                return False

        return True

    def is_occupied(self, position):
        return self.grid[position] is not self.empty_field

    def piece_to_one_hot(self, piece):
        return self._one_hots[self.type_to_id[type(piece)]]

    def to_one_hot(self):
        one_hot_grid = np.zeros((*self.grid_size, self.one_hot_dim))
        ranges = (range(x) for x in self.grid_size)
        for vec in itertools.product(*ranges):
            one_hot_grid[vec] = self.piece_to_one_hot(self.grid[vec])

        return one_hot_grid
