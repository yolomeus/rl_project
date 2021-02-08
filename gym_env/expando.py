import numpy as np
from gym import Env
from gym.spaces import MultiDiscrete, Box, Tuple

from gym_env.game.game import ExpandoGame
from spaces import OneHot


class Expando(Env):
    def __init__(self, grid_size: tuple, n_building_types: int = 2, n_players: int = 2):
        # actions: (cursor move direction, building_type), +1 no-op building
        n_directions = 1 + 2 * len(grid_size)
        self.action_space = MultiDiscrete([n_directions, n_building_types + 1])

        # observation space: (d_0 x d_1 x ... x d_n x building_type x player)
        obs_dims = grid_size + ((n_building_types + 1) * n_players,)
        grid_space = OneHot(obs_dims)
        cursor_space = Box(low=0, high=np.array(grid_size), dtype=np.uint)
        self.observation_space = Tuple((grid_space, cursor_space))

        self.game = ExpandoGame(grid_size, n_players)

    def step(self, actions):
        assert len(actions) == self.game.n_players, 'only one action per player is allowed'
        rewards = [self.game.take_turn(action, i) for i, action in enumerate(actions)]

        obs0 = self.game.get_observation(player_id=0)
        reward_0 = rewards[0]

        return obs0, reward_0, self.game.is_done, {}

    def reset(self):
        self.game.reset()

    def render(self, mode='human'):
        pass
