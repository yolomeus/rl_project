from time import sleep

from stable_baselines3 import DQN

from gym_env.expando import Expando


def main():
    # load policy from a checkpoint and use it as opponent
    policy = DQN.load("ckpts/dqn_expando_15000000")
    env = Expando(grid_size=(12, 16), max_turns=200, render=True, ui_font_size=14, flat_observations=True,
                  observe_all=True, n_players=2, policies_other=[policy], seed=420)

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
