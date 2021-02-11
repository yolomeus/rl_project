from abc import ABC, abstractmethod


class Piece(ABC):
    def __init__(self, player, board):
        self.age = 0
        self.player = player
        self.board = board

    @abstractmethod
    def compute_reward(self):
        """Compute the current reward for the piece"""


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
