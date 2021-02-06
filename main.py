from gym_env.expando import Expando


def main():
    env = Expando(grid_size=(3, 4, 8), n_building_types=2)
    env.reset()
    for _ in range(10):
        obs = env.observation_space.sample()
        action = env.action_space.sample()
        obs, reward, done, info = env.step([action, action])
        print(env.observation_space.contains(obs))


if __name__ == '__main__':
    main()
