import json
from pathlib import Path

import numpy as np
from PIL import Image


class DatasetImageSource:
    """Map a continuous camera position to an image from a fixed dataset."""

    def __init__(self, image_paths):
        self.image_paths = sorted(str(path) for path in image_paths)
        if not self.image_paths:
            raise ValueError("DatasetImageSource requires at least one image.")

    @property
    def num_images(self):
        return len(self.image_paths)

    def position_to_index(self, position):
        position = float(np.clip(position, -1.0, 1.0))
        normalized = (position + 1.0) / 2.0
        idx = int(round(normalized * (self.num_images - 1)))
        return int(np.clip(idx, 0, self.num_images - 1))

    def get_image_path(self, position):
        idx = self.position_to_index(position)
        return idx, self.image_paths[idx]

    def load_image(self, image_path, image_shape):
        height, width, channels = image_shape
        if channels != 3:
            raise ValueError("DatasetImageSource expects RGB image_shape.")

        image = Image.open(image_path).convert("RGB")
        image = image.resize((width, height), Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


class CachedArtiMuseRewardProvider:
    """Return precomputed ArtiMuse scores keyed by image path."""

    def __init__(self, score_json_path):
        self.score_json_path = Path(score_json_path)
        with self.score_json_path.open("r") as f:
            self.score_dict = json.load(f)

        if not self.score_dict:
            raise ValueError("CachedArtiMuseRewardProvider found no scores.")

        self.image_paths = sorted(self.score_dict.keys())
        self.scores = np.array(
            [float(self.score_dict[path]) for path in self.image_paths],
            dtype=np.float32,
        )

    def score_image(self, image_path):
        return float(self.score_dict[str(image_path)])
