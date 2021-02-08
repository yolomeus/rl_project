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
        """

        :param direction:
        :return:
        """
        assert direction >= 0
        assert 0 <= direction <= 2 * len(self.cursor), 'we can move in two directions for each axis or not move at all'

        cur_pos = self.cursor.copy()
        n_dimensions = len(self.cursor)
        if 0 < direction <= n_dimensions:
            # 1 to n_dimensions
            cur_pos[direction - 1] += 1
        elif len(self.cursor) < direction <= 2 * len(self.cursor):
            # n_dimensions to 2 * n_dimensions
            cur_pos[(direction + 1) % n_dimensions] -= 1
        # else do nothing...

        is_legal_move = self.is_legal(cur_pos)
        if is_legal_move:
            self.cursor = cur_pos

        return is_legal_move

    def is_legal(self, position):
        """

        :param position:
        :return:
        """
        for i, x in enumerate(position):
            if x < 0 or x >= self.board.grid_size[i]:
                return False

        return True

    def register_building(self, building: Building):
        self.buildings.append(building)

    def get_observation(self):
        # this is where we control observability
        return self.board.to_binary()

    @property
    def current_reward(self):
        building_reward = sum([building.compute_reward() for building in self.buildings])
        return building_reward
