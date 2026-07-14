import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AestheticSERLWrapper(gym.Wrapper):
    """Wrap vector observations into the dict format expected by HIL-SERL."""

    def __init__(self, env, image_key="image", image_shape=(128, 128, 3)):
        super().__init__(env)
        self.image_key = image_key
        self.image_shape = image_shape
        self.image_cache = {}

        self.observation_space = spaces.Dict(
            {
                "state": self._make_state_space(env.observation_space),
                self.image_key: spaces.Box(
                    low=0,
                    high=255,
                    shape=self.image_shape,
                    dtype=np.uint8,
                ),
            }
        )

    def _make_state_space(self, observation_space):
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("AestheticSERLWrapper expects a Box observation space.")

        low = observation_space.low.astype(np.float32).copy()
        high = observation_space.high.astype(np.float32).copy()

        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_current_image_path(self):
        if hasattr(self.env, "get_current_image_path"):
            return self.env.get_current_image_path()

        if hasattr(self.env, "_get_current_image_info"):
            _, image_path, _ = self.env._get_current_image_info()
            return image_path

        idx = self.env._position_to_index(self.env.current_position)
        return self.env.image_paths[idx]

    def _load_image(self, image_path):
        if image_path not in self.image_cache:
            self.image_cache[image_path] = self.env.image_source.load_image(
                image_path,
                self.image_shape,
            )

        return self.image_cache[image_path].copy()

    def _wrap_obs(self, obs):
        image_path = self._get_current_image_path()
        return {
            "state": np.asarray(obs, dtype=np.float32),
            self.image_key: self._load_image(image_path),
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._wrap_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._wrap_obs(obs), reward, terminated, truncated, info
