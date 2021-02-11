import numpy as np
from gym import Env
from gym.spaces import MultiDiscrete, Box, Tuple

from gym_env.game.pieces import Farm, City, Empty
from gym_env.game.game import ExpandoGame
from spaces import OneHot


class Expando(Env):
    def __init__(self, grid_size: tuple, n_players: int = 2):
        self.piece_types = (Empty, Farm, City)
        n_piece_types = len(self.piece_types)

        # actions: (cursor move direction, piece_type)
        # where (cursor move direction) encodes +1 or -1 movement along an axis and 0 for no movement.
        n_move_directions = 1 + 2 * len(grid_size)
        self.action_space = MultiDiscrete([n_move_directions, n_piece_types])

        # observation space: (d_0 x d_1 x ... x d_n x piece_type x player)
        obs_dims = grid_size + (n_piece_types * n_players,)
        grid_space = OneHot(obs_dims)
        cursor_space = Box(low=0, high=np.array(grid_size), dtype=np.uint)
        self.observation_space = Tuple((grid_space, cursor_space))

        self.game = ExpandoGame(grid_size, n_players, self.piece_types)

    def step(self, actions):
        assert len(actions) == self.game.n_players, 'only one action per player is allowed'
        rewards = [self.game.take_turn(action, i) for i, action in enumerate(actions)]

        obs0 = self.game.get_observation(player_id=0)
        reward_0 = rewards[0]

        return obs0, reward_0, self.game.is_done, {}

    def seed(self, seed=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def reset(self):
        self.game.reset()

    def render(self, mode='human'):
        pass
