import itertools
from copy import copy

import numpy as np
from hydra.utils import instantiate
from numpy.random import default_rng

from gym_env.game.board import Board
from gym_env.game.player import Player


class ExpandoGame:
    """Expando is a multiplayer, turn based, strategy game that takes place on a d_0 x ... x d_n grid.
    Each turn, a player receives reward based on the constellation of all the pieces that she has placed on the board
    so far. Every player is associated with a cursor which can be moved along all of the board's axis but only for 1
    step per turn. In addition, a player may choose to set a piece after moving or choosing not to move.
    The player with the largest total reward at the game's end, wins and receives a large reward, the loser wins the
    same amount but as penalty.
    """

    def __init__(self, grid_size, n_players, max_turns, final_reward, piece_types, seed=None):
        """

        :param grid_size: the dimensions of the board.
        :param n_players: number of players participating in the game.
        :param max_turns: the maximum number of turns that a game is allowed to last. Each player's turn is counted.
        :param piece_types: list of sub-classes of Piece, that can be used in the game
        :param final_reward: the amount of reward that is either granted for winning or used as penalty for loosing
        :param seed: used to seed any random number generators
        """
        self.np_random = default_rng(seed)

        self.name_to_id = {t: i for i, t in enumerate(piece_types.keys())}

        self.grid_size = grid_size
        self.n_dims = len(grid_size)

        self.final_reward = final_reward
        self.n_players = n_players
        self.max_turns = max_turns
        self.n_turns = 0

        self.board = Board(grid_size, self)
        self.players = [Player(i, self.board) for i in range(n_players)]

        self._init_player_positions()
        self._action_pairs = None
        self._id_to_piece = {i: instantiate(piece, player=None, board=None) for i, piece in
                             enumerate(piece_types.values())}

    def _init_player_positions(self):
        """Place each player's cursor at a random position.
        """
        cursors = set()
        while len(cursors) < self.n_players:
            cursor = tuple(self.np_random.integers(0, self.grid_size))
            cursors.add(cursor)

        for c, p in zip(cursors, self.players):
            p.cursor = np.array(c, dtype=np.int64)

    def take_turn(self, action, player_id):
        """Perform a player's turn given an action.

        :param action: the action the paler should take, encoded as integer.
        :param player_id: the player_id of the player that should perform the action.
        :return: the player's reward after performing the action.
        """
        if not isinstance(action, list):
            action = self._discrete_to_multidiscrete(action)

        cursor_move, place_action = action
        move_direction: np.ndarray = self._decode_action(cursor_move, 'cursor_move')
        piece_id = self._decode_action(place_action, 'piece_type')

        cur_player = self.players[player_id]
        cur_player.move_cursor(move_direction)
        if piece_id is not None:
            piece = self._get_piece(piece_id, cur_player)
            cur_player.place_piece(piece)

        reward = self.players[player_id].current_reward
        self.players[player_id].total_reward += reward

        # increase  counters
        for piece in self.all_pieces:
            piece.age += 1
        self.n_turns += 1

        if self.is_done:
            # add the final reward or penalty, depending on whether the player did win or lose
            other_players = [self.players[i] for i in range(len(self.players)) if i != player_id]
            if all([cur_player.total_reward > p.total_reward for p in other_players]):
                # player has won
                reward += self.final_reward
            else:
                # player did loose
                reward -= self.final_reward
        return reward

    def reset(self):
        """Reset the game's state and place the player's cursors at random positions.
        """
        self.n_turns = 0
        self.board.reset_grid()
        self.players = [Player(i, self.board) for i in range(self.n_players)]
        self._init_player_positions()

    def get_observation(self, player_id, formatting):
        """Return an observation from the perspective of a player, i.e. treating her as player 0.

        :param player_id: player_id of the player from who's perspective the game is observed.
        :param formatting: 'flat' or 'grid' representation of the game. Where flat is a k-dimensional vector, and grid a
        d_0 x ... x d_n dimensional tensor.
        :return: the observation of the player encoded as numpy array.
        """
        return self.players[player_id].get_observation(formatting)

    @property
    def is_done(self):
        """Whether the game has reached a terminal state.

        :return: bool, true if game has ended
        """
        return self.board.is_full() or self.n_turns > self.max_turns

    @property
    def all_pieces(self):
        """Get a list of all pieces that are currently placed on the board.

        :return: list of all pieces.
        """
        all_pieces = []
        for player in self.players:
            all_pieces.extend(player.pieces)
        return all_pieces

    def _decode_action(self, action, action_type):
        """Factory method for decoding integer encoded actions based on the type of action.

        :param action: integer encoding an action
        :param action_type: either 'cursor_move' ir 'piece_type'.
        :return: a decoded action.
        """
        if action_type == 'cursor_move':
            return self._decode_cursor_move(action)
        elif action_type == 'piece_type':
            return self._decode_piece_type(action)
        else:
            raise NotImplementedError()

    def _decode_cursor_move(self, action: int):
        """Take an integer and return the corresponding direction vector for moving the cursor. The integer encoding
        for a cursor move is within range [0, 2*n_dims] where 0 means no change in direction, 1 <= k <= n_dims means
        +1 along the k-th dimension and n_dims+1 <= i <= 2*n_dims means -1 along the (i-n_dims)-th dimension.

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

    @staticmethod
    def _decode_piece_type(action):
        """Decode the correct piece type from an integer id.

        :param action: integer id representing a piece type
        :return: a piece object
        """

        return None if action is 0 else action

    def _discrete_to_multidiscrete(self, action):
        """Transform a discrete action into a multidiscrete action by looking up a corresponding action pair.

        :param action: integer representing a multidiscrete action.
        :return: a pair of integers, each representing an action.
        """
        if self._action_pairs is None:
            move_directions = range(2 * self.n_dims + 1)
            piece_types = range(len(self.name_to_id))

            self._action_pairs = [(a_0, a_1) for a_0, a_1 in itertools.product(move_directions, piece_types)]

        return self._action_pairs[action]

    def seed(self, seed=None):
        """Seed any random number generators. Note that pseudo random actions are performed at initialization, so in
        order to seed these actions as well you need to pass a seed to the constructor.

        :param seed: the seed to set.
        """
        self.np_random = default_rng(seed)

    def _get_piece(self, piece_id, player):
        piece = copy(self._id_to_piece[piece_id])
        piece.player = player
        piece.board = self.board
        return piece
