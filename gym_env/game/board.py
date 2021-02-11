from collections import defaultdict


class Board:
    def __init__(self, grid_size, game):
        self.game = game
        self.type_to_id = {t: i for i, t in enumerate(game.building_types)}
        self.grid_size = grid_size
        self.grid = None
        self.reset_grid()

    def get_piece(self, coordinates):
        return self.grid[coordinates]

    def place_piece(self, piece, coordinates):
        assert self.is_legal_position(coordinates), 'the coordinates provided are off the grid'
        assert self.grid[coordinates] is None, 'can only place piece on empty field'
        self.grid[coordinates] = piece

    def reset_grid(self):
        self.grid = defaultdict(lambda: None)

    def is_legal_position(self, position):
        """Check whether position is a legal position on the board.

        :param position: the position to check
        :return: bool
        """
        for i, x in enumerate(position):
            if x < 0 or x >= self.grid_size[i]:
                return False

        return True

    def to_one_hot(self):
        # TODO implement
        return 0
