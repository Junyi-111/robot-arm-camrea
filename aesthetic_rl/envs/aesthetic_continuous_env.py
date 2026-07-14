import numpy as np
import gymnasium as gym
from gymnasium import spaces

from aesthetic_rl.envs.providers import (
    CachedArtiMuseRewardProvider,
    DatasetImageSource,
)


class AestheticContinuousEnv(gym.Env):
    """
    Phase 3 toy env.

    目的：
    用连续 action 模拟相机视角选择。
    action ∈ [-1, 1]
    action 会被映射到一张图片。
    reward 直接从 ArtiMuse 预计算分数表读取。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        score_json_path=None,
        image_source=None,
        reward_provider=None,
        max_steps=10,
        normalize_reward=True,
        reward_clip=5.0,
        reward_eps=1e-6,
    ):
        super().__init__()

        self.score_json_path = score_json_path
        self.max_steps = max_steps
        self.normalize_reward = normalize_reward
        self.reward_clip = reward_clip
        self.reward_eps = reward_eps

        if reward_provider is None:
            if score_json_path is None:
                raise ValueError("score_json_path or reward_provider is required.")
            reward_provider = CachedArtiMuseRewardProvider(score_json_path)

        if image_source is None:
            image_source = DatasetImageSource(reward_provider.image_paths)

        self.image_source = image_source
        self.reward_provider = reward_provider

        self.image_paths = self.image_source.image_paths
        self.scores = np.array(
            [self.reward_provider.score_image(path) for path in self.image_paths],
            dtype=np.float32,
        )
        self.num_images = self.image_source.num_images

        # 连续动作：模拟一个一维相机视角
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # 简单 observation:
        # [当前 step, 当前归一化位置, 上一步 reward]
        if self.normalize_reward:
            reward_low = -float(self.reward_clip)
            reward_high = float(self.reward_clip)
        else:
            reward_low = float(np.min(self.scores))
            reward_high = float(np.max(self.scores))

        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, reward_low], dtype=np.float32),
            high=np.array([float(max_steps), 1.0, reward_high], dtype=np.float32),
            dtype=np.float32,
        )

        self.current_step = 0
        self.current_position = 0.0
        self.last_reward = 0.0
        self.seen_image_rewards = {}

    def _position_to_index(self, position):
        """
        position ∈ [-1, 1]
        映射到图片 index.
        """
        return self.image_source.position_to_index(position)

    def _get_obs(self):
        return np.array(
            [
                float(self.current_step),
                float(self.current_position),
                float(self.last_reward),
            ],
            dtype=np.float32,
        )

    def _get_current_image_info(self):
        idx, image_path = self.image_source.get_image_path(self.current_position)
        raw_reward = self.reward_provider.score_image(image_path)
        return idx, image_path, raw_reward

    def get_current_image_path(self):
        _, image_path, _ = self._get_current_image_info()
        return image_path

    def _normalize_reward(self, image_path, raw_reward):
        if not self.normalize_reward:
            return raw_reward, None, None, len(self.seen_image_rewards)

        seen_rewards = np.array(
            list(self.seen_image_rewards.values()),
            dtype=np.float32,
        )

        if len(seen_rewards) < 2:
            normalized_reward = raw_reward / 100.0
            reward_mean = None
            reward_std = None
        else:
            reward_mean = float(np.mean(seen_rewards))
            reward_std = float(np.std(seen_rewards))
            normalized_reward = (raw_reward - reward_mean) / (
                reward_std + self.reward_eps
            )
            normalized_reward = float(
                np.clip(normalized_reward, -self.reward_clip, self.reward_clip)
            )

        self.seen_image_rewards[image_path] = raw_reward
        return normalized_reward, reward_mean, reward_std, len(self.seen_image_rewards)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.current_position = 0.0
        self.last_reward = 0.0

        obs = self._get_obs()
        idx, image_path, raw_reward = self._get_current_image_info()
        info = {
            "image_path": image_path,
            "image_index": idx,
            "position": self.current_position,
            "aesthetic_score": raw_reward,
            "raw_reward": raw_reward,
        }

        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        # 这里把 action 当成“相机位置变化”
        delta = float(action[0])
        self.current_position = float(
            np.clip(self.current_position + 0.2 * delta, -1.0, 1.0)
        )

        idx, image_path, raw_reward = self._get_current_image_info()
        reward, reward_mean, reward_std, reward_count = self._normalize_reward(
            image_path,
            raw_reward,
        )

        self.current_step += 1
        self.last_reward = reward

        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_obs()

        info = {
            "image_path": image_path,
            "image_index": idx,
            "position": self.current_position,
            "aesthetic_score": raw_reward,
            "raw_reward": raw_reward,
            "normalized_reward": reward,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_count": reward_count,
        }

        return obs, reward, terminated, truncated, info
