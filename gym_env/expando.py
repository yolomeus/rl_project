from gym import Env
from gym.spaces import MultiDiscrete

from gym_env.game.game import ExpandoGame
from spaces import OneHot


class Expando(Env):
    def __init__(self, grid_size: tuple, n_building_types: int = 2, n_players: int = 2):
        self.action_space = MultiDiscrete([4, n_building_types + 1])
        obs_dims = grid_size + ((n_building_types + 1) * n_players,)  # +1 for empty position
        self.observation_space = OneHot(obs_dims)

        self.game = ExpandoGame(grid_size, n_players)

    def step(self, actions):
        assert len(actions) == self.game.n_players, 'only one action per player is allowed'
        rewards = [self.game.take_turn(action) for action in actions]

        obs0 = self.game.get_observation(0)
        reward_0 = rewards[0]

        return obs0, reward_0, self.game.is_done, {}

    def reset(self):
        self.game.reset()

    def render(self, mode='human'):
        pass
