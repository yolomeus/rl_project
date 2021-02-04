import numpy as np
from gym import Space


class OneHot(Space):
    """n-dimensional space of one-hot vectors e.g. with shape (2, 3, 6), from which a sample is a tensor
    containing 2 x 3 six-dimensional one-hot vectors.
    """

    def __init__(self, shape):
        """

        :param shape: shape of the space, where the last dimension corresponds to the one-hot encoding's size.
        """
        if type(shape) not in [tuple, list, np.ndarray]:
            shape = (shape,)
        self.shape = shape
        super().__init__(shape, np.int8)

    def sample(self):
        n_one_hots = np.prod(self.shape[:-1])
        one_hot_dim = self.shape[-1]
        one_hots = self.np_random.multinomial(n=1,
                                              pvals=[1.0 / one_hot_dim] * one_hot_dim,
                                              size=n_one_hots)
        one_hots = np.reshape(one_hots, self.shape)
        return one_hots

    def contains(self, x):
        # TODO implement
        same_shape = x.shape == self.shape
        return same_shape
