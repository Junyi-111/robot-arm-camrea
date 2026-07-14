from pathlib import Path

from aesthetic_rl.envs.aesthetic_continuous_env import AestheticContinuousEnv


def main():
    project_root = Path("/home/junyi/robot_aesthetic_rl")
    score_json_path = project_root / "aesthetic_rl" / "data" / "artimuse_scores.json"

    env = AestheticContinuousEnv(
        score_json_path=str(score_json_path),
        max_steps=10,
    )

    obs, info = env.reset()
    print("Initial obs:", obs)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    total_reward = 0.0

    for t in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print("-" * 60)
        print("step:", t)
        print("action:", action)
        print("obs:", obs)
        print("reward:", reward)
        print("image:", info["image_path"])
        print("position:", info["position"])
        print("done:", terminated or truncated)

        if terminated or truncated:
            break

    print("=" * 60)
    print("Total reward:", total_reward)


if __name__ == "__main__":
    main()