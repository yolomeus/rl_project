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

        self.one_hot_dim = 1 + game.n_players * (len(game.piece_types) - 1)
        self._n_piece_types = len(self.type_to_id) - 1
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

    @property
    def all_positions(self):
        return itertools.product(*(range(x) for x in self.grid_size))

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

    def piece_to_one_hot(self, piece, observing_player_id):
        piece_type_id = self.type_to_id[type(piece)]
        # empty piece doesn't belong to a player
        if piece == self.empty_field:
            return self._one_hots[0]

        piece_player_id = piece.player.player_id
        # swap player 0 and observing player
        if observing_player_id != 0:
            if piece_player_id == 0:
                player_id = observing_player_id
            elif observing_player_id == piece_player_id:
                player_id = 0
            else:
                player_id = piece_player_id
        else:
            player_id = piece_player_id

        # lookup one-hot
        one_hot_id = piece_type_id + self._n_piece_types * player_id
        assert one_hot_id != 0
        return self._one_hots[one_hot_id]

    def to_one_hot(self, observing_player_id=0):
        """

        :param observing_player_id: id of the player that observes. This player's pieces will be encoded as if
        the player was player 0.
        :return: one-hot encoding of the board from observing player's perspective.
        """
        one_hot_grid = np.zeros((*self.grid_size, self.one_hot_dim))
        for vec in self.all_positions:
            one_hot_grid[vec] = self.piece_to_one_hot(self.grid[vec], observing_player_id)

        return one_hot_grid

    def is_full(self):
        return all([self.is_occupied(pos) for pos in self.all_positions])
