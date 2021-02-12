import numpy as np


class Player:
    """A Player can have a list of associated pieces, set on a board. To set pieces, the player has a cursor that can
    be moved and indicates where to place a piece.
    """

    def __init__(self, player_id, board):
        """

        :param player_id: integer identifying the player
        :param board: board object that the player interacts with
        """
        self.player_id = player_id
        self.pieces = []
        self.board = board
        self.cursor = np.zeros(len(self.board.grid_size))

        # game stats
        self.room = 0
        self.population = 0

    def __str__(self):
        player_id = f'Player {self.player_id}'
        population_info = f'population/room: {self.population}/{self.room}'
        happiness = f'happiness penalty: {self.happiness_penalty}'
        turn_reward = f'current turn reward: {self.current_reward}'

        sep = "-" * 50

        return '\n'.join([sep, player_id, population_info, happiness, turn_reward, sep])

    def move_cursor(self, direction):
        """Move the player's cursor by adding a direction vector.

        :param direction: an offset vector that is added to the current cursor position if it describes a legal move.
        """
        if self.board.is_within_grid(self.cursor + direction):
            self.cursor += direction

    def place_piece(self, piece):
        """Place a piece on the board at the current cursor position of the player.

        :param piece: the piece to place.
        :return: whether the placing the piece was a success, i.e. the move was valid.
        :rtype: bool
        """
        piece.position = self.cursor.copy()
        success = self.board.place_piece(piece, tuple(self.cursor))
        if success:
            self.pieces.append(piece)
            piece.at_placement()
        return success

    def get_observation(self):
        # this is where we control observability
        return self.board.to_one_hot(), self.cursor

    @property
    def happiness_penalty(self):
        """Compute the happiness penalty which is added to the reward.

        :return: negative scalar or 0 which is the happiness penalty.
        """
        return min(self.room - self.population, 0)

    @property
    def current_reward(self):
        turn_rewards = sum([piece.turn_reward() for piece in self.pieces])
        return turn_rewards + self.happiness_penalty
