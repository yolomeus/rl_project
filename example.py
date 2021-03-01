from time import sleep

import hydra
from omegaconf import DictConfig

from gym_env.expando import Expando


@hydra.main(config_path='gym_env/config', config_name='config')
def main(cfg: DictConfig):
    env = Expando(**cfg)

    obs_0, obs_1 = env.reset()
    for _ in range(10000):
        action_0 = env.action_space.sample()
        obs_0, reward, done, info = env.step(action_0)
        obs_1 = info['obs_other'][0]
        env.render()
        sleep(.05)
        assert env.observation_space.contains(obs_0)


if __name__ == '__main__':
    main()
