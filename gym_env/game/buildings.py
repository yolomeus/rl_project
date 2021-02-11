from abc import ABC, abstractmethod


class Building(ABC):
    def __init__(self, player, board):
        self.age = 0
        self.player = player
        self.board = board

    @abstractmethod
    def compute_reward(self):
        """Compute the reward of the building"""


class Empty(Building):

    def compute_reward(self):
        return 0


class City(Building):
    # TODO implement
    def compute_reward(self):
        return 1


class Farm(Building):
    # TODO implement
    def compute_reward(self):
        return 1
