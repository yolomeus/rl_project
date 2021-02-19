from gym_env.expando import Expando


def main():
    env = Expando(grid_size=(12, 16), max_turns=500, render=True, ui_font_size=12)
    env.reset()
    for _ in range(1000):
        action_0, action_1 = env.action_space.sample(), env.action_space.sample()
        obs, reward, done, info = env.step(action_0, other_actions=[action_1])
        env.render()
        assert env.observation_space.contains(obs)


if __name__ == '__main__':
    main()
