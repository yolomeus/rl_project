import itertools
from collections import defaultdict

import numpy as np

from gym_env.game.pieces import Empty


class Board:
    """Board representation for the Expando Game.
    """

    def __init__(self, grid_size, game):
        """
        :param grid_size: dimensions of the board.
        :param game: game which the board belongs to.
        """
        self.grid_size = grid_size
        self.name_to_id = game.name_to_id
        # shared empty field
        self.empty_field = Empty(None, self)
        self.grid = None
        self.reset_grid()

        self.one_hot_dim = 1 + game.n_players * (len(game.name_to_id) - 1)
        self._n_piece_types = len(self.name_to_id) - 1
        self._one_hots = np.eye(self.one_hot_dim)

    @property
    def all_positions(self):
        """Iterate over all possible positions on the board.
        :return: an iterable of positions.
        """
        return itertools.product(*(range(x) for x in self.grid_size))

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
        """Clear the grid.
        """
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
        """Check whether a piece is placed at a position.

        :param position: the position to check on the board.
        :return: bool whether position is occupied
        """
        return self.grid[position] is not self.empty_field

    def is_full(self):
        """Check whether all fields on the board are occupied.

        :return: True if board is filled with pieces.
        """
        return all([self.is_occupied(pos) for pos in self.all_positions])

    def _piece_to_one_hot(self, piece, observing_player_id):
        """Encode a piece as one-hot depending relative to the observing player, i.e. the player sees herself as player
        0.

        :param piece: the piece to encode.
        :param observing_player_id: the player_id of the player that observes the piece
        :return: numpy array as one-hot encoding
        """
        piece_type_id = self.name_to_id[piece.name]
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
        """Get a one-hot representation of the grid.

        :param observing_player_id: id of the player that observes. This player's pieces will be encoded as if
        the player was player 0.
        :return: one-hot encoding of the board from observing player's perspective.
        """
        one_hot_grid = np.zeros((*self.grid_size, self.one_hot_dim))
        for vec in self.all_positions:
            one_hot_grid[vec] = self._piece_to_one_hot(self.grid[vec], observing_player_id)

        return one_hot_grid
