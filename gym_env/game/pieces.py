import itertools
from abc import ABC, abstractmethod

import numpy as np
from pyglet.shapes import Rectangle


class Piece(ABC):
    def __init__(self, player, board, position=None):
        self.age = 0
        self.player = player
        self.board = board
        self.position = position

    @abstractmethod
    def turn_reward(self):
        """Compute the current reward for the piece"""

    def at_placement(self):
        """Perform any effects that placement of this piece has."""
        pass

    def __str__(self):
        return f'Player {self.player.player_id}: {self.__class__.__name__}'

    def to_drawable(self, x, y, batch, square_size, color):
        r = Rectangle(x, y,
                      square_size, square_size,
                      color=color,
                      batch=batch)
        return r


class Empty(Piece):

    def turn_reward(self):
        return 0

    def to_drawable(self, x, y, batch, square_size, color):
        r = Rectangle(x, y,
                      square_size, square_size,
                      color=(10, 10, 10),
                      batch=batch)
        return r


class City(Piece):
    room_capacity = 0.5

    def turn_reward(self):
        return 0

    def at_placement(self):
        self.player.room += self.room_capacity

    def to_drawable(self, x, y, batch, square_size, color):
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
    reward_size = 0.1
    reward_delay = 0
    # whether to ignore adjacent cities that are diagonal w.r.t self
    ignore_diagonal = True

    def __init__(self, player, board):
        super().__init__(player, board)
        self.generates_reward = False
        self._adjacent_directions = None

    def turn_reward(self):
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
        self.player.population += 1

    def to_drawable(self, x, y, batch, square_size, color):
        r = super().to_drawable(x, y, batch, square_size, color)
        if not self.generates_reward:
            r.opacity = 100
        return r
