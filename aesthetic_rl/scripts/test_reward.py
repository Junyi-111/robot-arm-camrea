import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
print("Project root:", ROOT)
print("Python:", sys.executable)

from aesthetic_rl.reward.artimuse_reward import ArtiMuseReward

model_path = ROOT / "ArtiMuse" / "checkpoints" / "ArtiMuse"
image_path = ROOT / "ArtiMuse" / "example" / "test.jpg"

print("Model path:", model_path)
print("Image path:", image_path)
print("Loading ArtiMuse reward model...")

reward_model = ArtiMuseReward(str(model_path), device="cuda:0")

print("Model loaded. Scoring image...")
score = reward_model.score_image(str(image_path))

print("Aesthetic reward:", score)
