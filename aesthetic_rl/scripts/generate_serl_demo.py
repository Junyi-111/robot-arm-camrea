import argparse
import pickle
from pathlib import Path

import numpy as np

from aesthetic_rl.envs.make_aesthetic_env import make_serl_aesthetic_env


def collect_random_transitions(num_transitions, seed):
    env = make_serl_aesthetic_env()
    rng = np.random.default_rng(seed)

    transitions = []
    obs, _ = env.reset(seed=seed)

    while len(transitions) < num_transitions:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)

        done = bool(terminated)
        transitions.append(
            {
                "observations": obs,
                "actions": np.asarray(action, dtype=env.action_space.dtype),
                "next_observations": next_obs,
                "rewards": float(reward),
                "masks": float(1.0 - done),
                "dones": done,
            }
        )

        if terminated or truncated:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        else:
            obs = next_obs

    return transitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aesthetic_rl/data/aesthetic_continuous_demo.pkl"),
    )
    parser.add_argument("--num-transitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    transitions = collect_random_transitions(args.num_transitions, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(transitions, f)

    rewards = np.array([transition["rewards"] for transition in transitions])
    print(f"Saved {len(transitions)} transitions to {args.output}")
    print(
        "Reward stats: "
        f"min={rewards.min():.3f}, "
        f"mean={rewards.mean():.3f}, "
        f"max={rewards.max():.3f}"
    )


if __name__ == "__main__":
    main()
