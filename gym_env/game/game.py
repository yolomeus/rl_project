from gym_env.game.board import Board
from gym_env.game.player import Player


class ExpandoGame:
    def __init__(self, grid_size, n_players):
        self.n_players = n_players
        self.board = Board(grid_size)
        self.players = [Player(i, self.board) for i in range(n_players)]
        self.n_turns = 0

    def take_turn(self, action):
        cur_player = self.players[self.n_turns % self.n_players]
        cur_player.act(action)
        self.n_turns += 1

    def reset(self):
        self.n_turns = 0
        self.board.clear()

    def get_observation(self, player_id):
        return self.players[player_id].get_observation()

    @property
    def is_done(self):
        return False
