from abc import ABC, abstractmethod
from collections import defaultdict

from gym import Env
from gym.spaces import Discrete

from spaces import OneHot


class Expando(Env):
    def __init__(self, grid_size: tuple, n_actions: int, n_building_types: int = 2, n_players: int = 2):
        # TODO compute n_buildings_types internally
        self.action_space = Discrete(n_actions)
        obs_dims = grid_size + ((n_building_types + 1) * n_players,)  # +1 for empty position
        self.observation_space = OneHot(obs_dims)

        self.players = [Player(i) for i in range(n_players)]

    def step(self, actions):
        assert len(actions) == len(self.players)
        for player, action in zip(actions, self.players):
            player.act(action)

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def current_reward(self, player_id):
        all_rewards = []
        player = self.players[player_id]
        for building in player.buildings:
            all_rewards.append(building.compute_reward())
        return sum(all_rewards)


class Player:
    def __init__(self, player_id):
        self.player_id = player_id
        self.buildings = []

    def act(self, action):
        pass

    def register_building(self, building):
        self.buildings.append(building)
