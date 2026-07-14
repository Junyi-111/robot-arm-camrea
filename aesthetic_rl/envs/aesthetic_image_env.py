import os
import glob
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AestheticImageEnv(gym.Env):
    """
    Phase 2 toy environment.

    这个环境不控制真实机械臂。
    它只做一件事：
        action -> 选择一张图片 -> ArtiMuse 打分 -> reward
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        image_dir,
        reward_model,
        max_steps=5,
        cache_reward=True,
    ):
        super().__init__()

        self.image_dir = image_dir
        self.reward_model = reward_model
        self.max_steps = max_steps
        self.cache_reward = cache_reward

        self.image_paths = sorted(
            glob.glob(os.path.join(image_dir, "*.jpg"))
            + glob.glob(os.path.join(image_dir, "*.png"))
            + glob.glob(os.path.join(image_dir, "*.jpeg"))
        )

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

        self.num_images = len(self.image_paths)

        # action: 选择一张图片
        self.action_space = spaces.Discrete(self.num_images)

        # observation 暂时用一个简单向量：
        # [当前 step, 上一次 action, 上一次 reward]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([float(max_steps), float(self.num_images), 10.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.current_step = 0
        self.last_action = 0
        self.last_reward = 0.0

        # reward 缓存，避免重复图片反复跑 ArtiMuse
        self.reward_cache = {}

    def _get_obs(self):
        return np.array(
            [
                float(self.current_step),
                float(self.last_action),
                float(self.last_reward),
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.last_action = 0
        self.last_reward = 0.0

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        action = int(action)

        if action < 0 or action >= self.num_images:
            raise ValueError(f"Invalid action {action}, should be in [0, {self.num_images - 1}]")

        image_path = self.image_paths[action]

        # 调用 ArtiMuse reward
        if self.cache_reward and image_path in self.reward_cache:
            reward = self.reward_cache[image_path]
        else:
            reward = float(self.reward_model.score_image(image_path))
            if self.cache_reward:
                self.reward_cache[image_path] = reward

        self.current_step += 1
        self.last_action = action
        self.last_reward = reward

        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_obs()

        info = {
            "image_path": image_path,
            "action": action,
            "aesthetic_score": reward,
            "step": self.current_step,
        }

        return obs, reward, terminated, truncated, info