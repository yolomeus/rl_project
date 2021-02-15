from gym_env.expando import Expando


def main():
    env = Expando(grid_size=(6, 8), max_turns=500)
    env.reset()
    for _ in range(1000):
        obs = env.observation_space.sample()
        action = env.action_space.sample()
        obs, reward, done, info = env.step([action, action])

        print('-' * 100)
        player_0 = env.game.players[0]
        print(player_0)
        player_1 = env.game.players[1]
        print(player_1)

        assert env.observation_space.contains(obs)


if __name__ == '__main__':
    main()
