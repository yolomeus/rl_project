from abc import ABC, abstractmethod


class Building(ABC):
    def __init__(self, player, board):
        self.age = 0
        self.player = player
        self.board = board

    @abstractmethod
    def compute_reward(self):
        pass
