import sys
from pathlib import Path

from experiments.config import DefaultTrainingConfig


PROJECT_ROOT = Path("/home/junyi/robot_aesthetic_rl")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aesthetic_rl.envs.make_aesthetic_env import make_serl_aesthetic_env


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["image"]
    classifier_keys = []
    proprio_keys = ["state"]

    setup_mode = "single-arm-fixed-gripper"
    encoder_type = "resnet"

    max_steps = 1000
    max_traj_length = 10
    replay_buffer_capacity = 10000
    training_starts = 100
    random_steps = 100
    batch_size = 32
    cta_ratio = 2
    steps_per_update = 50
    log_period = 10
    checkpoint_period = 0
    buffer_period = 0

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        return make_serl_aesthetic_env(obs_horizon=1, image_key=self.image_keys[0])

    def process_demos(self, demo):
        return demo
