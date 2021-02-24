import pyglet
from pyglet.graphics import Batch
from pyglet.shapes import Rectangle
from pyglet.text import Label
from pyglet.window import Window

from gym_env.game.game import ExpandoGame


class GameRenderer(Window):
    """Takes a 2D ExpandoGame and draws it, everytime `step()` is called.
    """

    def __init__(self, game: ExpandoGame, cell_size=50, padding=10, ui_font_size=12):
        """
        :param game: The game to render
        :param cell_size: the length of each square in pixels
        :param padding: the padding between cells in pixels
        :param ui_font_size: size of the font, used to show player statistics.
        """
        self.player_colors = [(84, 22, 180),
                              (255, 106, 0),
                              (204, 255, 0),
                              (244, 147, 242)]
        self.padding = padding
        self.cell_size = cell_size
        self.square_size = self.cell_size - self.padding
        self.game = game
        self.board = self.game.board
        self.font_height = ui_font_size * 0.75

        assert len(self.board.grid_size) == 2, 'only 2d grids can be rendered at the moment'

        h, w = self.game.grid_size
        window_height = h * cell_size + padding
        # extra room for displaying scores
        self.window_height = window_height + 2 * self.font_height * (game.n_players + 1) + self.padding
        self.window_width = w * cell_size + padding

        super().__init__(width=self.window_width, height=int(self.window_height))
        self.batch = Batch()

    @staticmethod
    def step():
        """Render a single frame.
        """
        pyglet.clock.tick()

        for window in pyglet.app.windows:
            window.switch_to()
            window.dispatch_events()
            window.dispatch_event('on_draw')
            window.flip()

    def on_draw(self):
        """triggered by pyglet to draw everything.
        """
        self.clear()
        self.draw_grid()
        self.draw_cursors()
        self.draw_scores()

    def draw_grid(self):
        """Draws the board's grid.
        """
        h, w = self.board.grid_size

        pieces = []
        for i in range(w):
            for j in range(h):
                piece = self.board.get_piece((j, i))
                x, y = self._get_canvas_pos(i, j)
                r = piece.to_drawable(x, y, self.batch, self.square_size, self._get_piece_color(piece))
                pieces.append(r)

        self.batch.draw()

    def draw_cursors(self):
        """Draw the cursors of each player.
        """

        rects = []
        for player in self.game.players:
            cursor = tuple(player.cursor)
            x, y = self._get_canvas_pos(*cursor)
            color = self._get_player_color(player)
            color = self.brighten(color, 50)
            r = Rectangle(y, x,
                          self.square_size, self.square_size,
                          color=color,
                          batch=self.batch)
            rects.append(r)

        self.batch.draw()

    def draw_scores(self):
        """Draw the ui containing player statistics.
        """
        batch = Batch()
        labels = []
        header = ['pl', 'population', 'room', 'happiness', 'turn reward', 'total reward']
        sep = ' | '
        score_strings = [sep + sep.join(header) + sep]
        for player in self.game.players:
            scores = map(lambda x: str(round(x, 3)),
                         [player.player_id, player.population, player.room, player.happiness_penalty,
                          player.current_reward, player.total_reward])
            line = ''
            for i, score in enumerate(scores):
                score_len = len(str(score))
                head_len = len(header[i])
                line += ' ' * (head_len + len(sep) - score_len)
                line += score
            score_strings.append(line)

        # .75 for pt to px
        font_size = self.font_height / 0.75  # int(.75 * self.window_height * self.score_space / self.game.n_players)
        for i, score_str in enumerate(score_strings, start=1):
            if i > 1:
                c = self._get_player_color(self.game.players[i - 2]) + (255,)
                c = self.brighten(c, 50)
            else:
                c = (255,) * 4
            label = Label(score_str,
                          x=0,
                          y=self.height - 2 * self.font_height * i,
                          font_name='Consolas',
                          font_size=font_size,
                          color=c,
                          batch=batch)
            labels.append(label)
        batch.draw()

    def _get_canvas_pos(self, x, y):
        """Translate a position on the board to a position on the pyglet canvas.

        :param x: x coordinate
        :param y: y coordinate
        :return: canvas position in pixels.
        """
        return x * self.cell_size + self.padding, y * self.cell_size + self.padding

    def _get_piece_color(self, piece):
        """Get the color of a piece, depending on it's owner.

        :param piece: the piece to get the color for.
        :return: an rgb color tuple
        """
        return self._get_player_color(piece.player)

    def _get_player_color(self, player):
        """Get the color assigned to a player, returns gray if None.

        :param player: the player to get the associated color from.
        :return: a rgb tuple
        """
        if player is None:
            return 10, 10, 10

        return self.player_colors[player.player_id]

    @staticmethod
    def brighten(color, val):
        """Increase color intensity on all channels, clips anything above 255.0 or below 0.0.

        :param color: tuple representing the color to brighten.
        :param val: amount of added brightness.
        :return: a tuple
        """
        new_color = []
        for c in color:
            x = c + val
            if 0 <= x <= 255:
                new_color.append(x)
            elif x < 0:
                new_color.append(0)
            else:
                new_color.append(255)
        return new_color
