from gym import Env
from gym.spaces import MultiDiscrete, Box

from gym_env.game.game import ExpandoGame
from gym_env.game.pieces import Farm, City, Empty
from spaces import OneHot, OneHotBox


class Expando(Env):
    def __init__(self, grid_size: tuple, n_players: int = 2, max_turns=100, observe_all=False):
        self.n_players = n_players
        self.observe_all = observe_all
        self.piece_types = (Empty, Farm, City)
        n_piece_types = len(self.piece_types)

        # actions: (cursor move direction, piece_type)
        # where (cursor move direction) encodes +1 or -1 movement along an axis and 0 for no movement.
        n_move_directions = 1 + 2 * len(grid_size)
        self.action_space = MultiDiscrete([n_move_directions, n_piece_types])

        # observation space:
        # (d_0 * ... * d_n * piece_type * player
        # + cursor_d_0 + ... + cursor_d_n + population + room)
        obs_dims = grid_size + (n_piece_types * n_players,)
        self.observation_space = OneHotBox(OneHot(obs_dims), Box(0.0, 1.0, shape=(4,)), flatten=True)

        self.game = ExpandoGame(grid_size, n_players, max_turns, self.piece_types, final_reward=100)

    def step(self, action, other_actions=None):
        """Perform each player's turn.

        :param action: action to take as player 0
        :param other_actions: optional list of actions to take for the other players. Will be sampled from actions_space
        if not provided.
        :return: obs_0, reward_0, done, info
        """

        reward_0 = self.game.take_turn(action, player_id=0)
        if other_actions is not None:
            assert len(other_actions) + 1 == self.n_players, 'please provide an action for each player'
            rewards_other = [self.game.take_turn(action, i) for i, action in enumerate(other_actions, start=1)]

        else:
            rewards_other = [self.game.take_turn(self.action_space.sample(), i) for i in range(1, self.n_players)]

        obs_0 = self.game.get_observation(player_id=0, formatting='flat')
        info = {}
        if self.observe_all:
            other_obs = [self.game.get_observation(i, 'flat') for i in range(1, self.n_players)]
            info = {'rewards_other': rewards_other, 'obs_other': other_obs}

        done = self.game.is_done

        if done:
            self.game.reset()

        return obs_0, reward_0, done, info

    def seed(self, seed=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def reset(self):
        self.game.reset()
        return self.game.get_observation(0, 'flat')

    def render(self, mode='human'):
        pass
