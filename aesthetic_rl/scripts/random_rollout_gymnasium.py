from aesthetic_rl.envs.make_aesthetic_env import make_aesthetic_env


def main():
    env = make_aesthetic_env()

    for ep in range(3):
        obs, info = env.reset()
        total_reward = 0.0

        for step in range(100):
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(
                f"ep={ep}, step={step}, "
                f"reward={reward:.3f}, total={total_reward:.3f}, "
                f"terminated={terminated}, truncated={truncated}, "
                f"image_idx={info.get('image_index')}"
            )

            if terminated or truncated:
                break

        print("=" * 60)
        print(f"Episode {ep} total reward: {total_reward:.3f}")


if __name__ == "__main__":
    main()