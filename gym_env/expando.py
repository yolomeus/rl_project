import numpy as np
from gym import Env
from gym.spaces import MultiDiscrete, Box, Tuple

from gym_env.game.game import ExpandoGame
from gym_env.game.pieces import Farm, City, Empty
from spaces import OneHot


class Expando(Env):
    def __init__(self, grid_size: tuple, n_players: int = 2, max_turns=100, observe_all=False):
        self.observe_all = observe_all
        self.piece_types = (Empty, Farm, City)
        n_piece_types = len(self.piece_types)

        # actions: (cursor move direction, piece_type)
        # where (cursor move direction) encodes +1 or -1 movement along an axis and 0 for no movement.
        n_move_directions = 1 + 2 * len(grid_size)
        self.action_space = MultiDiscrete([n_move_directions, n_piece_types])

        # observation space: (d_0 x d_1 x ... x d_n x piece_type x player)
        obs_dims = grid_size + (n_piece_types * n_players,)
        self.observation_space = OneHotBox(OneHot(obs_dims), Box(0.0, 1.0, shape=(4,)), flatten=True)

        self.game = ExpandoGame(grid_size, n_players, max_turns, self.piece_types, final_reward=100)

    def step(self, actions):
        assert len(actions) == self.game.n_players, 'only one action per player is allowed'
        rewards = [self.game.take_turn(action, i) for i, action in enumerate(actions)]

        obs0 = self.game.get_observation(player_id=0, formatting='flat')
        reward_0 = rewards[0]
        info = {}
        if self.observe_all:
            other_obs = [self.game.get_observation(i, 'flat') for i in range(1, len(actions))]
            info = {'rewards_other': rewards[1:], 'obs_other': other_obs}

        done = self.game.is_done

        if done:
            self.game.reset()

        return obs0, reward_0, done, info

    def seed(self, seed=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def reset(self):
        self.game.reset()

    def render(self, mode='human'):
        pass
