from time import sleep

from omegaconf import OmegaConf
from stable_baselines3 import DQN

from gym_env.env import Expando


def main():
    policy = DQN.load('experiments/multirun/grid_sweep/2021-03-07/14-13-11/1/ckpts/rl_model_5000000_steps.zip')
    # or load environment using a yaml default_config:
    cfg = OmegaConf.load('experiments/multirun/grid_sweep/2021-03-07/14-13-11/1/.hydra/config.yaml')
    env_cfg = cfg.env
    env_cfg.render = True
    env = Expando(**env_cfg)
    obs_0 = env.reset()
    for i in range(10000):
        action_0 = policy.predict(obs_0)[0][0]
        obs_0, reward, done, info = env.step(action_0)
        env.render()
        if i == 2*99:
            while True:
                sleep(.01)


if __name__ == '__main__':
    main()
