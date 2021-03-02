from time import sleep

from gym_env.env import Expando


def main():
    env = Expando(grid_size=(8, 8), max_turns=200, observe_all=True, render=True)
    # or load environment using a yaml default_config:
    # env = Expando.from_config('gym_env/default_config/config.yaml')
    obs_0, obs_1 = env.reset()
    for _ in range(10000):
        action_0 = env.action_space.sample()
        obs_0, reward, done, info = env.step(action_0)
        obs_1 = info['obs_other'][0]
        env.render()
        sleep(.01)
        assert env.observation_space.contains(obs_0)


if __name__ == '__main__':
    main()
