from abc import ABCMeta

import numpy as np
from numpy.random import default_rng

from gym_env.game.board import Board
from gym_env.game.pieces import Empty
from gym_env.game.player import Player


class ExpandoGame:
    def __init__(self, grid_size, n_players, piece_types, seed=None):
        self.np_random = default_rng(seed)

        self.piece_types = piece_types
        self.grid_size = grid_size
        self.n_dims = len(grid_size)

        self.n_players = n_players
        self.n_turns = 0

        self.board = Board(grid_size, self)
        self.players = [Player(i, self.board) for i in range(n_players)]

        self._init_player_positions()

    def _init_player_positions(self):
        cursors = set()
        while len(cursors) < self.n_players:
            cursor = tuple(self.np_random.integers(0, self.grid_size))
            cursors.add(cursor)

        for c, p in zip(cursors, self.players):
            p.cursor = np.array(c, dtype=np.int64)

    def take_turn(self, action, player_id):
        cursor_move, place_action = action
        move_direction: np.ndarray = self._decode_action(cursor_move, 'cursor_move')
        piece_type: ABCMeta = self._decode_action(place_action, 'place_action')

        cur_player = self.players[player_id]
        cur_player.move_cursor(move_direction)
        if not piece_type == Empty:
            cur_player.place_piece(piece_type(cur_player, self.board))

        reward = self.players[player_id].current_reward
        self.n_turns += 1
        return reward

    def reset(self):
        self.n_turns = 0
        self.board.reset_grid()

    def get_observation(self, player_id):
        return self.players[player_id].get_observation()

    @property
    def is_done(self):
        return False

    def _decode_action(self, action, action_type):
        if action_type == 'cursor_move':
            return self._decode_cursor_move(action)
        elif action_type == 'place_action':
            return self._decode_place_action(action)
        else:
            raise NotImplementedError()

    def _decode_cursor_move(self, action: int):
        """Take an integer and return the corresponding direction vector for moving the cursor. The integer encoding
        for a cursor move is within range [0, 2*n_dims] where 0 means no change in direction, 1 <= k <= n_dims means +1
        along the k-th dimension and n_dims+1 <= i <= 2*n_dims means -1 along the (i-n_dims)-th dimension.

        :param action: integer representing cursor move
        :return: direction vector to move to as a numpy array
        """
        assert action >= 0
        assert 0 <= action <= 2 * self.n_dims, 'we can move in two directions for each axis or not move at all'

        direction = np.zeros((self.n_dims,), dtype=np.int64)
        if 0 < action <= self.n_dims:
            # 1 to n_dimensions
            direction[action - 1] += 1
        elif self.n_dims < action <= 2 * self.n_dims:
            # n_dimensions to 2 * n_dimensions
            direction[(action + 1) % self.n_dims] -= 1
        # else action=0, do nothing...

        return direction

    def _decode_place_action(self, action):
        piece_type = self.piece_types[action]
        return piece_type
