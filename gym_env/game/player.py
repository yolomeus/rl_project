import numpy as np

from gym_env.game.buildings import Building


class Player:
    def __init__(self, player_id, board):
        """

        :param player_id:
        :param board:
        """
        self.player_id = player_id
        self.buildings = []
        self.board = board
        self.cursor = np.zeros(len(self.board.grid_size))

    def move_cursor(self, direction):
        if self.board.is_legal_position(self.cursor + direction):
            self.cursor += direction

    def register_building(self, building: Building):
        self.buildings.append(building)

    def get_observation(self):
        # this is where we control observability
        return self.board.to_one_hot(), self.cursor

    @property
    def current_reward(self):
        building_reward = sum([building.compute_reward() for building in self.buildings])
        return building_reward
