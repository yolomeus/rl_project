from gym_env.expando import Expando


def main():
    env = Expando(grid_size=(6, 8), max_turns=500)
    env.reset()
    for _ in range(1000):
        test_obs = env.observation_space.sample()
        action_0, action_1 = env.action_space.sample(), env.action_space.sample()
        obs, reward, done, info = env.step(action_0, other_actions=[action_1])

        print('-' * 100)
        player_0 = env.game.players[0]
        print(player_0)
        player_1 = env.game.players[1]
        print(player_1)

        assert env.observation_space.contains(obs)


if __name__ == '__main__':
    main()
