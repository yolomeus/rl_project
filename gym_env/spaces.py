import numpy as np
from gym.spaces import MultiBinary, Box


class OneHot(MultiBinary):
    """Special case of MultiBinary space, where each entry along the last axis is a one-hot vector. e.g. with shape
    (2, 3, 6), from which a sample is a tensor containing 2 x 3 six-dimensional
    one-hot vectors.
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


class OneHotBox(Box):
    """Concatenation of a OneHot and Box space.
    """

    def __init__(self, one_hot, flat_box, flatten=True):
        self.one_hot = one_hot
        self.flat_box = flat_box
        self.flatten = flatten
        if not flatten:
            shape = list(one_hot.n)
            shape[-1] = shape[-1] + flat_box.shape[0]
            shape = tuple(shape)
        else:
            shape = (np.prod(one_hot.n) + flat_box.shape[0],)

        super().__init__(0, 1, shape)

    def sample(self):

        obs_0, obs_1 = self.one_hot.sample(), self.flat_box.sample()

        if not self.flatten:
            obs_1 = np.repeat(obs_1, np.prod(obs_0.shape[:-1])).reshape(obs_0.shape[:-1] + (-1,))
            x = np.concatenate([obs_0, obs_1], axis=-1)
            return x
        obs = np.concatenate([obs_0.ravel(), obs_1])
        return obs

    def contains(self, x):
        x = np.squeeze(x)
        n_one_hot = self.one_hot.one_hot_dim

        if not self.flatten:
            one_hots = x.reshape(-1, x.shape[-1])
            one_hot_obs, box_obs = one_hots[:, :n_one_hot], one_hots[0, n_one_hot:]
            one_hot_obs = one_hot_obs.reshape(self.one_hot.n)
        else:
            len_flat_box = self.flat_box.shape[0]
            one_hot_obs = x[:-len_flat_box].reshape(self.one_hot.n)
            box_obs = x[-len_flat_box:]

        contains_one_hot = self.one_hot.contains(one_hot_obs)
        contains_box = self.flat_box.contains(box_obs)

        return contains_one_hot and contains_box
