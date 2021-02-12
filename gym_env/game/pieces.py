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

    def compute_reward(self):
        return 0


class City(Piece):
    # TODO implement
    def compute_reward(self):
        return 1


class Farm(Piece):
    # TODO implement
    def compute_reward(self):
        return 1
