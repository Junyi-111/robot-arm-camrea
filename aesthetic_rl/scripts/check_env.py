from gymnasium.utils.env_checker import check_env

from aesthetic_rl.envs.make_aesthetic_env import make_aesthetic_env


def main():
    env = make_aesthetic_env()
    check_env(env.unwrapped)
    print("Gymnasium env check passed.")


if __name__ == "__main__":
    main()