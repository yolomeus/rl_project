from gym import Env
from gym.spaces import MultiDiscrete, Box, Discrete

from gym_env.game.game import ExpandoGame
from gym_env.rendering import GameRenderer
from gym_env.spaces import OneHot, OneHotBox


class Expando(Env):
    """Gym environment wrapping the expando game. For details on the game, check the ExpandoGame class.

    Action-space:
        Multidiscrete: (move_direction, action_type) with move_direction in {0, ..., 2 * n_axis}, where 0 - n_axis
        represent movement along an axis in the positive direction and n_axis - 2 * n_axis in negative direction. And
        action_type is in {piece_type_0, ..., piece_type_n}, ie.e. there is a placement action for each type of piece.

        If `multi_discrete_actions` is set to False, the discrete action space over all items in the cartesian product
        of move_direction and action_type will be used to get a single discrete action space over
        {0,..., n_axis * piece_type_n}. The order of action pairs then is the same as returned by `itertools.product()`.

    Observation-space description:
        A Box space where each observation has dimensions (axis_0 x axis_1 ... x axis_n x n_one_hot x n_scores)
        where axis_k is the length of the k-th axis of the game board grid,
        n_one_hot = 1 + n_players * (n_piece_types - 1) the dimension of the piece's one-hot encodings and n_scores = 3
        is the number of additional normalized features regarding the player: is_cursor_position, room, population.
        Note that n_one_hot accounts for the empty piece_type which doesn't belong to a player.

        If `flat_observations` is set to True, the box observations are going to be
        (axis_0 * axis_1 ... * axis_n * n_one_hot + n_scores) dimensional vectors, where n_scores = 3 + n_axis, since
        the cursor's position is on longer represented as bit, but as normalized (x, y, ...) coordinates.
    """

    def __init__(self,
                 grid_size: tuple,
                 piece_types: tuple,
                 n_players: int = 2,
                 max_turns=100,
                 final_reward=100,
                 policies_other=None,
                 observe_all=False,
                 multi_discrete_actions=False,
                 flat_observations=False,
                 render=False,
                 cell_size=50,
                 padding=5,
                 ui_font_size=14,
                 seed=None):

        if policies_other is not None:
            assert n_players - 1 == len(policies_other), 'please provide a policy for each opponent.'

        self.n_players = n_players
        self.policies_other = policies_other
        self.observe_all = observe_all
        self.piece_types = piece_types
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

        self.game = ExpandoGame(grid_size, n_players, max_turns, final_reward=final_reward,
                                piece_types=piece_types,
                                seed=seed)
        self.observation_format = 'flat' if flat_observations else 'grid'
        self.do_render = render
        if self.do_render:
            self.renderer = GameRenderer(self.game, cell_size, padding, ui_font_size)

        self.seed(seed)

    def step(self, action, other_actions=None):
        """Perform each player's turn.

        :param action: action to take as player 0
        :param other_actions: optional list of actions to take for the other players. Will be sampled from actions_space
        if not provided.
        :return: obs_0, reward_0, done, info
        """
        if self.policies_other is not None:
            assert other_actions is None, 'other actions are already defined by the policies passed at initialization'

        # other player actions passed as argument
        if other_actions is not None:
            assert len(other_actions) + 1 == self.n_players, 'please provide an action for each player'
            rewards_other = [self.game.take_turn(action, i) for i, action in enumerate(other_actions, start=1)]
        # other player actions defined by policies passed to constructor
        elif self.policies_other is not None:
            other_obs = [self.game.get_observation(i, self.observation_format) for i in range(1, self.n_players)]
            actions_other = [policy.predict(obs)[0][0] for obs, policy in zip(other_obs, self.policies_other)]
            rewards_other = [self.game.take_turn(a, i) for i, a in enumerate(actions_other, start=1)]
        # no other player actions provided: sample
        else:
            rewards_other = [self.game.take_turn(self.action_space.sample(), i) for i in range(1, self.n_players)]

        info = {}
        if self.observe_all:
            other_obs_new = [self.game.get_observation(i, self.observation_format) for i in range(1, self.n_players)]
            info = {'rewards_other': rewards_other, 'obs_other': other_obs_new}

        reward_0 = self.game.take_turn(action, player_id=0)
        obs_0 = self.game.get_observation(player_id=0, formatting=self.observation_format)
        done = self.game.is_done

        if done:
            self.game.reset()

        return obs_0, reward_0, done, info

    def seed(self, seed=None):
        """Set seeds of all random number generators. Note that pseudo random actions are performed at initialization,
        so in order to seed these actions as well you need to pass a seed to the constructor.

        :param seed: seed to set
        """
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        self.game.seed(seed)

    def reset(self, player_id=0):
        """Reset the environment.

        :param player_id: id of the player to get the first observation from.
        :return: observation of player with player_id or a list of all observations if `observe_all` was set.
        """
        self.game.reset()
        if self.observe_all:
            return [self.game.get_observation(i, self.observation_format) for i in range(self.n_players)]
        return self.game.get_observation(player_id, self.observation_format)

    def render(self, mode='human'):
        """Render a pyglet visualization. Only works with 2D grids.
        """
        assert len(self.game.grid_size) < 3, 'Only 2D grids are supported for rendering at the moment.'
        if self.do_render:
            self.renderer.step()
