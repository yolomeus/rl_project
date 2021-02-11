import numpy as np


class Player:
    def __init__(self, player_id, board):
        """

        :param player_id:
        :param board:
        """
        self.player_id = player_id
        self.pieces = []
        self.board = board
        self.cursor = np.zeros(len(self.board.grid_size))

    def move_cursor(self, direction):
        if self.board.is_legal_position(self.cursor + direction):
            self.cursor += direction

    def place_piece(self, piece):
        self.pieces.append(piece)
        self.board.place_piece(piece, tuple(self.cursor))

    def get_observation(self):
        # this is where we control observability
        return self.board.to_one_hot(), self.cursor

    @property
    def current_reward(self):
        return sum([piece.compute_reward() for piece in self.pieces])
