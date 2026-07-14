import json
from pathlib import Path

from aesthetic_rl.reward.artimuse_reward import ArtiMuseReward


def main():
    project_root = Path("/home/junyi/robot_aesthetic_rl")

    model_path = project_root / "ArtiMuse" / "checkpoints" / "ArtiMuse"
    image_dir = project_root / "aesthetic_rl" / "data" / "test_images"
    output_path = project_root / "aesthetic_rl" / "data" / "artimuse_scores.json"

    image_paths = sorted(
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png"))
    )

    print("Number of images:", len(image_paths))
    print("Loading ArtiMuse reward model...")

    reward_model = ArtiMuseReward(
        model_path=str(model_path),
        device="cuda:0",
    )

    scores = {}

    for i, image_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] scoring {image_path.name}")
        score = float(reward_model.score_image(str(image_path)))
        scores[str(image_path)] = score
        print("score:", score)

    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    print("Saved scores to:", output_path)


if __name__ == "__main__":
    main()