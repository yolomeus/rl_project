import itertools
from abc import ABC, abstractmethod

import numpy as np


class Piece(ABC):
    def __init__(self, player, board, position=None):
        self.age = 0
        self.player = player
        self.board = board
        self.position = position

    @abstractmethod
    def turn_reward(self):
        """Compute the current reward for the piece"""

    def at_placement(self):
        """Perform any effects that placement of this piece has."""
        pass

    def __str__(self):
        return f'Player {self.player.player_id}: {self.__class__.__name__}'


class Empty(Piece):

    def turn_reward(self):
        return 0


class City(Piece):
    room_capacity = 1

    def turn_reward(self):
        return 0

    def at_placement(self):
        self.player.room += self.room_capacity


class Farm(Piece):
    reward_size = 1
    reward_delay = 0

    def turn_reward(self):
        # only produces a reward if adjacent to a city
        if self.age >= self.reward_delay:
            # get adjacent pieces
            n_dims = len(self.position)
            adjacent_directions = itertools.permutations((-1, 0, 1), n_dims)
            for direction in adjacent_directions:
                adj_pos = tuple(self.position + np.array(direction))
                piece = self.board.get_piece(adj_pos)
                if isinstance(piece, City) and piece.player == self.player:
                    return self.reward_size
        return 0

    def at_placement(self):
        self.player.population += 1
