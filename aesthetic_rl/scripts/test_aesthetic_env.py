from pathlib import Path
import numpy as np

from aesthetic_rl.reward.artimuse_reward import ArtiMuseReward
from aesthetic_rl.envs.aesthetic_image_env import AestheticImageEnv


def main():
    project_root = Path("/home/junyi/robot_aesthetic_rl")

    model_path = project_root / "ArtiMuse" / "checkpoints" / "ArtiMuse"
    image_dir = project_root / "aesthetic_rl" / "data" / "test_images"

    print("Project root:", project_root)
    print("Model path:", model_path)
    print("Image dir:", image_dir)

    print("Loading ArtiMuse reward model...")
    reward_model = ArtiMuseReward(
        model_path=str(model_path),
        device="cuda:0",
    )
    print("Reward model loaded.")

    env = AestheticImageEnv(
        image_dir=str(image_dir),
        reward_model=reward_model,
        max_steps=5,
        cache_reward=True,
    )

    obs, info = env.reset()
    print("Initial obs:", obs)
    print("Number of images:", env.num_images)

    total_reward = 0.0

    for t in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        print("-" * 60)
        print(f"Step: {t}")
        print(f"Action: {action}")
        print(f"Image: {info['image_path']}")
        print(f"Reward: {reward}")
        print(f"Obs: {obs}")
        print(f"Done: {terminated or truncated}")

        if terminated or truncated:
            break

    print("=" * 60)
    print("Total reward:", total_reward)


if __name__ == "__main__":
    main()