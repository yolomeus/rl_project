import itertools
from abc import ABC, abstractmethod

import numpy as np
from pyglet.shapes import Rectangle


class Piece(ABC):
    """Base-class for pieces. A piece can generate reward based on its and other pieces positions on the board that it
    holds a reference to.
    """

    def __init__(self, player, board, position=None):
        """

        :param player: player that the piece belongs to
        :param board: board that the piece is placed on
        :param position: position on the board
        """
        self.age = 0
        self.player = player
        self.board = board
        self.position = position

    @abstractmethod
    def turn_reward(self):
        """Compute the current reward that the piece generates
        :return: a numeric reward
        """

    def at_placement(self):
        """Perform any side-effects that result from placing this piece. This is completely optional.
        """
        pass

    def __str__(self):
        return f'Player {self.player.player_id}: {self.__class__.__name__}'

    def to_drawable(self, x, y, batch, square_size, color):
        """Return a drawable 2D representation of the piece as a pyglet supported object. Defaults to a square.

        :param x: x position that the piece should be drawn at.
        :param y: y position that the piece should be drawn at.
        :param batch: pyglet Batch that the drawable should be added to.
        :param square_size: size of a square on the grid that the piece will be drawn on.
        :param color: color the piece should have.
        :return: a drawable pyglet object that holds a reference/was added to `batch`.
        """
        r = Rectangle(x, y,
                      square_size, square_size,
                      color=color,
                      batch=batch)
        return r


class Empty(Piece):
    """Piece that represents an Empty field.
    """

    def turn_reward(self):
        return 0

    def to_drawable(self, x, y, batch, square_size, color):
        r = Rectangle(x, y,
                      square_size, square_size,
                      color=(10, 10, 10),
                      batch=batch)
        return r


class City(Piece):
    """Cities increase a players capacity for population, but do not generate rewards themselves. However, other pieces
    will depend on cities for generating rewards.
    """

    # the capacity a city will grand the player
    room_capacity = 0.5

    def turn_reward(self):
        """A city doesn't actively produce reward.
        :return: zero reward
        """
        return 0

    def at_placement(self):
        """Placing a city grants the player additional room for population.
        """
        self.player.room += self.room_capacity

    def to_drawable(self, x, y, batch, square_size, color):
        """A city is represented as square containing a smaller square.
        """
        shapes = []
        r = super().to_drawable(x, y, batch, square_size, color)
        shapes.append(r)

        w = h = square_size * .4
        offset = square_size * .5
        r_inner = Rectangle(x + offset, y + offset,
                            w, h,
                            color=(0, 0, 0),
                            batch=batch)
        r_inner.anchor_position = (w * .5, h * .5)
        r_inner.opacity = 128
        shapes.append(r_inner)

        return shapes


class Farm(Piece):
    """Farms increase the overall population count of a player, but also generate reward when placed adjacent to a city.
    """

    population_increase = 1
    reward_size = 0.1
    reward_delay = 0

    # whether to ignore adjacent cities that are diagonal w.r.t self
    ignore_diagonal = True

    def __init__(self, player, board):
        super().__init__(player, board)
        self.generates_reward = False
        self._adjacent_directions = None

    def turn_reward(self):
        """Reward is only generated when the farm is placed adjacent to a city. Once next to a city, we assume it can no
        longer be moved and will generate reward for the rest of te object's lifetime.

        :return:
        """
        # if generating reward once, it can't be blocked
        if self.generates_reward:
            return self.reward_size

        # only produces a reward if adjacent to a city
        if self.age >= self.reward_delay:
            # check if there's a city in adjacent positions
            for direction in self._get_adjacent_directions(self.ignore_diagonal):
                adj_pos = tuple(self.position + np.array(direction))
                if self._is_owned_city(adj_pos):
                    self.generates_reward = True
                    return self.reward_size
        return 0

    def _is_owned_city(self, pos):
        piece = self.board.get_piece(pos)
        return isinstance(piece, City) and piece.player == self.player

    def _get_adjacent_directions(self, ignore_diagonal):
        """Get all directional vectors that land on adjacent pieces of self.

        :param ignore_diagonal: whether to ignore diagonal directions or not
        :return: iterable of directions as numpy arrays
        """
        # use cached directions
        if self._adjacent_directions is not None:
            return self._adjacent_directions

        # compute directions 1st time
        n_dims = len(self.position)
        if ignore_diagonal:
            adjacent_directions = []
            for i, val in itertools.product(range(n_dims), [-1, 1]):
                vec = np.zeros((n_dims,))
                vec[i] = val
                adjacent_directions.append(vec)
            self._adjacent_directions = adjacent_directions
        else:
            self._adjacent_directions = itertools.product((-1, 0, 1), repeat=n_dims)

        return self._adjacent_directions

    def at_placement(self):
        """Placing a farm causes an increase in population for the player.
        """
        self.player.population += self.population_increase

    def to_drawable(self, x, y, batch, square_size, color):
        r = super().to_drawable(x, y, batch, square_size, color)
        if not self.generates_reward:
            r.opacity = 100
        return r
