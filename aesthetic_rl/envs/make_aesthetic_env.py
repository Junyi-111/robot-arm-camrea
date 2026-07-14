from pathlib import Path
import sys

from aesthetic_rl.envs.aesthetic_continuous_env import AestheticContinuousEnv
from aesthetic_rl.envs.providers import (
    CachedArtiMuseRewardProvider,
    DatasetImageSource,
)
from aesthetic_rl.envs.serl_wrapper import AestheticSERLWrapper


PROJECT_ROOT = Path("/home/junyi/robot_aesthetic_rl")


def _add_hil_serl_to_path():
    serl_launcher_path = PROJECT_ROOT / "hil-serl" / "serl_launcher"
    if str(serl_launcher_path) not in sys.path:
        sys.path.insert(0, str(serl_launcher_path))


def make_aesthetic_env():
    score_json_path = (
        PROJECT_ROOT
        / "aesthetic_rl"
        / "data"
        / "artimuse_scores.json"
    )

    reward_provider = CachedArtiMuseRewardProvider(score_json_path)
    image_source = DatasetImageSource(reward_provider.image_paths)

    env = AestheticContinuousEnv(
        image_source=image_source,
        reward_provider=reward_provider,
        max_steps=10,
    )

    return env


def make_serl_aesthetic_env(obs_horizon=1, image_key="image"):
    _add_hil_serl_to_path()

    from serl_launcher.wrappers.chunking import ChunkingWrapper

    env = make_aesthetic_env()
    env = AestheticSERLWrapper(env, image_key=image_key)
    env = ChunkingWrapper(env, obs_horizon=obs_horizon, act_exec_horizon=None)

    return env
