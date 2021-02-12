import numpy as np
from gym.spaces import MultiBinary


class OneHot(MultiBinary):
    """n-dimensional space of one-hot vectors e.g. with shape (2, 3, 6), from which a sample is a tensor
    containing 2 x 3 six-dimensional one-hot vectors.
    """

    def __init__(self, shape):
        """

        :param shape: shape of the space, where the last dimension corresponds to the one-hot encoding's size.
        """
        super().__init__(shape)
        self.shape = shape
        self.one_hot_dim = self.shape[-1]

    def sample(self):
        n_one_hots = np.prod(self.shape[:-1])
        one_hots = self.np_random.multinomial(n=1,
                                              pvals=[1.0 / self.one_hot_dim] * self.one_hot_dim,
                                              size=n_one_hots)
        one_hots = np.reshape(one_hots, self.shape)
        return one_hots

    def contains(self, x):
        has_same_shape = x.shape == self.shape

        # check if one-hot vectors
        one_hots = x.reshape(-1, self.one_hot_dim)
        non_zero_count = np.count_nonzero(one_hots, axis=-1)

        has_single_element = np.all(non_zero_count == 1)
        sums_to_one = np.all(np.sum(one_hots, axis=-1) == 1)
        is_one_hot = has_single_element and sums_to_one

        return has_same_shape and is_one_hot
