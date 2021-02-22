from gym import Env
from gym.spaces import MultiDiscrete, Box, Discrete

from gym_env.game.game import ExpandoGame
from gym_env.game.pieces import Farm, City, Empty
from gym_env.rendering import GameRenderer
from spaces import OneHot, OneHotBox


class Expando(Env):
    def __init__(self,
                 grid_size: tuple,
                 n_players: int = 2,
                 max_turns=100,
                 observe_all=False,
                 multi_discrete_actions=False,
                 flat_observations=False,
                 render=False,
                 cell_size=50,
                 padding=5,
                 ui_font_size=14):
        self.n_players = n_players
        self.observe_all = observe_all
        self.piece_types = (Empty, Farm, City)
        n_piece_types = len(self.piece_types)

        # actions: (cursor move direction, piece_type)
        # where (cursor move direction) encodes +1 or -1 movement along an axis and 0 for no movement.
        n_move_directions = 1 + 2 * len(grid_size)
        if multi_discrete_actions:
            self.action_space = MultiDiscrete([n_move_directions, n_piece_types])
        else:
            self.action_space = Discrete(n_move_directions * n_piece_types)

        # observation space:
        # (d_0 * ... * d_n * piece_type * player
        # + cursor_d_0 + ... + cursor_d_n + population + room)
        k_cursor_features = 2 if flat_observations else 1
        obs_dims = grid_size + (1 + (n_piece_types - 1) * n_players,)
        self.observation_space = OneHotBox(OneHot(obs_dims),
                                           Box(0.0, 1.0, shape=(2 + k_cursor_features,)),
                                           flatten=flat_observations)

        self.game = ExpandoGame(grid_size, n_players, max_turns, self.piece_types, final_reward=100)
        self.observation_format = 'flat' if flat_observations else 'grid'
        self.do_render = render
        if self.do_render:
            self.renderer = GameRenderer(self.game, cell_size, padding, ui_font_size)

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

    def reset(self, player_id=0):
        """

        :param player_id: id of the player to get the first observation from.
        :return:
        """
        self.game.reset()
        if self.observe_all:
            return [self.game.get_observation(i, self.observation_format) for i in range(self.n_players)]
        return self.game.get_observation(0, self.observation_format)

    def render(self, mode='human'):
        if self.do_render:
            self.renderer.step()
