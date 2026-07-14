from aesthetic_rl.envs.make_aesthetic_env import make_aesthetic_env


def main():
    env = make_aesthetic_env()

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    obs, info = env.reset()
    print("Reset obs:", obs)
    print("Reset info:", info)

    for step in range(5):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print("-" * 60)
        print("step:", step)
        print("action:", action)
        print("obs:", obs)
        print("reward:", reward)
        print("terminated:", terminated)
        print("truncated:", truncated)
        print("done:", terminated or truncated)
        print("info:", info)

        if terminated or truncated:
            break


if __name__ == "__main__":
    main()