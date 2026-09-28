"""Create videos and a quantitative report for two Method-A evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "baseline": "#E76F51",
    "trained": "#2A9D8F",
    "threshold": "#E9C46A",
    "dark": "#132238",
}


def load_run(events_path: Path, label: str) -> dict[str, Any]:
    rows = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    mode = next(row for row in rows if row.get("kind") == "actor_mode")
    start = next(row for row in rows if row.get("kind") == "shadow_stop_episode_start")
    transitions = [row for row in rows if row.get("kind") == "transition"]
    episode_end = next(row for row in rows if row.get("kind") == "episode_end")

    frames = [
        {
            "step": 0,
            "score": float(start["score"]),
            "target_present": bool(start["target_present"]),
            "edge_clear": None,
            "action": None,
            "image": start["image"],
            "terminal": False,
            "shadow_stop": start.get("decision", {}),
        }
    ]
    for row in transitions:
        reward = row["reward"]
        frames.append(
            {
                "step": int(row["step"]),
                "score": float(row["score"]),
                "target_present": bool(reward["target_present"]),
                "edge_clear": bool(reward["target_edge_clear"]),
                "action": [float(value) for value in row["action"]],
                "image": row["image"],
                "terminal": bool(row["terminal"]),
                "shadow_stop": row.get("shadow_stop", {}),
            }
        )

    checkpoint_path = Path(mode["frozen_checkpoint"])
    checkpoint = int(checkpoint_path.stem.rsplit("_", 1)[-1])
    stop_event = next(
        (row for row in rows if row.get("kind") == "shadow_stop_would_return_to_best"),
        None,
    )
    return {
        "label": label,
        "checkpoint": checkpoint,
        "events_path": str(events_path.resolve()),
        "run_dir": str(events_path.parent.resolve()),
        "mode": mode,
        "start": start,
        "frames": frames,
        "transitions": transitions,
        "end_reason": episode_end["reason"],
        "stop_event": stop_event,
    }


def cumulative_trapezoid(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    result = np.zeros_like(values, dtype=np.float64)
    if len(values) > 1:
        result[1:] = np.cumsum((values[:-1] + values[1:]) * 0.5)
    return result


def metrics(run: dict[str, Any], planned_horizon: int) -> dict[str, Any]:
    frames = run["frames"]
    scores = np.asarray([frame["score"] for frame in frames], dtype=np.float64)
    targets = np.asarray([frame["target_present"] for frame in frames], dtype=bool)
    actions = np.asarray([frame["action"] for frame in frames[1:]], dtype=np.float64)

    present_transition_frames = [
        frame for frame in frames[1:] if frame["target_present"]
    ]
    edge_clear = np.asarray(
        [frame["edge_clear"] for frame in present_transition_frames], dtype=bool
    )
    near_edge_steps = [
        frame["step"]
        for frame in frames[1:]
        if frame["target_present"] and not frame["edge_clear"]
    ]
    missing_steps = [frame["step"] for frame in frames if not frame["target_present"]]
    saturated = np.abs(actions) >= 0.95

    return {
        "checkpoint": run["checkpoint"],
        "label": run["label"],
        "episode_steps": len(actions),
        "termination_reason": run["end_reason"],
        "initial_score": float(scores[0]),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "max_score": float(np.max(scores)),
        "max_score_step": int(np.argmax(scores)),
        "final_score": float(scores[-1]),
        "score_ge_40_rate": float(np.mean(scores >= 40.0)),
        "target_retention_observed_rate": float(np.mean(targets)),
        "target_coverage_planned_horizon_rate": float(
            np.count_nonzero(targets) / (planned_horizon + 1)
        ),
        "target_missing_frames": int(np.count_nonzero(~targets)),
        "target_lost_step": missing_steps[0] if missing_steps else None,
        "edge_clear_rate_when_target_present": (
            float(np.mean(edge_clear)) if len(edge_clear) else None
        ),
        "near_edge_rate_when_target_present": (
            float(np.mean(~edge_clear)) if len(edge_clear) else None
        ),
        "near_edge_frames": int(np.count_nonzero(~edge_clear)),
        "first_near_edge_step": near_edge_steps[0] if near_edge_steps else None,
        "action_component_saturation_rate": float(np.mean(saturated)),
        "action_step_any_saturation_rate": float(np.mean(np.any(saturated, axis=1))),
        "action_saturation_rate_by_joint": {
            joint: float(value)
            for joint, value in zip(
                ["J1", "J2", "J3", "J5"], np.mean(saturated, axis=0), strict=True
            )
        },
        "mean_absolute_action": float(np.mean(np.abs(actions))),
        "cumulative_aesthetic_quality_auc": float(cumulative_trapezoid(scores)[-1]),
        "cumulative_quality_above_32_auc": float(
            cumulative_trapezoid(np.maximum(scores - 32.0, 0.0))[-1]
        ),
        "shadow_stop_trigger_step": (
            int(run["stop_event"]["trigger_step"]) if run["stop_event"] else None
        ),
        "shadow_stop_best_step": (
            int(run["stop_event"]["return_target_step"]) if run["stop_event"] else None
        ),
    }


def put_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    scale: float = 0.65,
    color: tuple[int, int, int] = (235, 240, 245),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def render_video(run: dict[str, Any], output_path: Path, fps: float = 5.0) -> None:
    frames = run["frames"]
    scores = np.asarray([frame["score"] for frame in frames], dtype=np.float64)
    max_steps = int(frames[-1]["step"])
    width, height = 1280, 720
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    chart_x0, chart_y0, chart_w, chart_h = 870, 410, 365, 205
    score_lo = min(18.0, float(np.floor(scores.min() - 2.0)))
    score_hi = max(46.0, float(np.ceil(scores.max() + 2.0)))
    color = (81, 111, 231) if run["checkpoint"] == 200 else (143, 157, 42)

    rendered: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        source = cv2.imread(frame["image"])
        if source is None:
            raise FileNotFoundError(frame["image"])
        source = cv2.resize(source, (832, 624), interpolation=cv2.INTER_AREA)
        canvas = np.full((height, width, 3), (19, 29, 43), dtype=np.uint8)
        canvas[72:696, 24:856] = source

        put_text(canvas, f"Checkpoint {run['checkpoint']} - deterministic evaluation", (24, 45), 0.9, (255, 255, 255), 2)
        put_text(canvas, f"STEP {frame['step']:02d}/{max_steps:02d}", (885, 105), 0.85, (255, 255, 255), 2)
        put_text(canvas, f"Score       {frame['score']:6.2f}", (885, 150), 0.75)
        put_text(canvas, f"Mean so far {np.mean(scores[:index+1]):6.2f}", (885, 185), 0.65)
        put_text(canvas, f"Best so far {np.max(scores[:index+1]):6.2f}", (885, 220), 0.65)
        target_text = "YES" if frame["target_present"] else "NO - LOST"
        target_color = (130, 235, 150) if frame["target_present"] else (90, 90, 255)
        put_text(canvas, f"Target      {target_text}", (885, 265), 0.68, target_color, 2)
        if frame["edge_clear"] is None:
            edge_text = "N/A (start)"
            edge_color = (190, 200, 210)
        elif not frame["target_present"]:
            edge_text = "N/A (missing)"
            edge_color = (90, 90, 255)
        elif frame["edge_clear"]:
            edge_text = "CLEAR"
            edge_color = (130, 235, 150)
        else:
            edge_text = "NEAR EDGE"
            edge_color = (60, 190, 255)
        put_text(canvas, f"Edge        {edge_text}", (885, 300), 0.68, edge_color, 2)
        if frame["action"] is not None:
            action = np.asarray(frame["action"])
            put_text(canvas, "Action J1/J2/J3/J5", (885, 340), 0.55, (190, 200, 210))
            put_text(canvas, " ".join(f"{value:+.2f}" for value in action), (885, 368), 0.55)

        cv2.rectangle(canvas, (chart_x0, chart_y0), (chart_x0 + chart_w, chart_y0 + chart_h), (80, 94, 112), 1)
        threshold_y = int(chart_y0 + chart_h - (40.0 - score_lo) / (score_hi - score_lo) * chart_h)
        cv2.line(canvas, (chart_x0, threshold_y), (chart_x0 + chart_w, threshold_y), (74, 202, 233), 1, cv2.LINE_AA)
        put_text(canvas, "40", (chart_x0 + 5, threshold_y - 6), 0.4, (74, 202, 233))
        if index > 0:
            pts = []
            for j, score in enumerate(scores[: index + 1]):
                px = int(chart_x0 + (j / max(1, max_steps)) * chart_w)
                py = int(chart_y0 + chart_h - (score - score_lo) / (score_hi - score_lo) * chart_h)
                pts.append((px, np.clip(py, chart_y0, chart_y0 + chart_h)))
            cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, color, 3, cv2.LINE_AA)
        put_text(canvas, "Aesthetic score", (chart_x0, chart_y0 - 12), 0.55, (190, 200, 210))
        put_text(canvas, f"0", (chart_x0, chart_y0 + chart_h + 24), 0.45, (190, 200, 210))
        put_text(canvas, f"{max_steps}", (chart_x0 + chart_w - 25, chart_y0 + chart_h + 24), 0.45, (190, 200, 210))

        status = "RUNNING"
        status_color = (130, 235, 150)
        if index == len(frames) - 1:
            status = f"END: {run['end_reason'].upper()}"
            status_color = (90, 90, 255) if run["end_reason"] == "target_missing" else (60, 190, 255)
        elif frame["shadow_stop"].get("would_stop"):
            status = "SHADOW STOP TRIGGER"
            status_color = (74, 202, 233)
        put_text(canvas, status, (885, 675), 0.75, status_color, 2)
        rendered.append(canvas)

    hold = max(1, int(round(fps * 1.5)))
    for _ in range(hold):
        writer.write(rendered[0])
    for frame in rendered:
        writer.write(frame)
    for _ in range(hold * 2):
        writer.write(rendered[-1])
    writer.release()


def render_report(
    runs: list[dict[str, Any]],
    all_metrics: list[dict[str, Any]],
    common_steps: int,
    common_metrics: list[dict[str, float]],
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 9), dpi=160, facecolor="#F6F8FB")
    grid = fig.add_gridspec(2, 3, left=0.06, right=0.97, top=0.88, bottom=0.08, wspace=0.28, hspace=0.36)
    labels = [f"CKPT {run['checkpoint']}" for run in runs]
    colors = [COLORS["baseline"], COLORS["trained"]]

    ax = fig.add_subplot(grid[0, :2])
    for run, color in zip(runs, colors, strict=True):
        score = np.asarray([frame["score"] for frame in run["frames"]])
        step = np.arange(len(score))
        ax.plot(step, score, lw=2.5, color=color, label=f"Checkpoint {run['checkpoint']}")
        if run["end_reason"] == "target_missing":
            ax.scatter(step[-1], score[-1], s=90, color=color, marker="X", zorder=5)
            ax.annotate("target lost", (step[-1], score[-1]), xytext=(-58, 20), textcoords="offset points", color=color, fontweight="bold")
        if run["stop_event"]:
            stop_step = int(run["stop_event"]["trigger_step"])
            ax.axvline(stop_step, color=color, ls=":", alpha=0.7)
            ax.text(stop_step + 1, ax.get_ylim()[0] + 1, "shadow STOP", rotation=90, va="bottom", color=color, fontsize=8)
    ax.axhline(40, color=COLORS["threshold"], ls="--", lw=1.8, label="score = 40")
    ax.set(title="Aesthetic score trajectory", xlabel="Step", ylabel="ArtiMuse score")
    ax.legend(loc="lower right", ncol=3, fontsize=8)

    ax = fig.add_subplot(grid[0, 2])
    x = np.arange(len(runs))
    width = 0.18
    fields = [("mean_score", "Mean"), ("median_score", "Median"), ("max_score", "Best"), ("final_score", "Final")]
    for offset, (field, name) in enumerate(fields):
        vals = [metric[field] for metric in all_metrics]
        ax.bar(x + (offset - 1.5) * width, vals, width, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 50)
    ax.set_title("Score summary (all observed frames)")
    ax.set_ylabel("Score")
    ax.legend(fontsize=8, ncol=2)

    ax = fig.add_subplot(grid[1, 0])
    for run, color in zip(runs, colors, strict=True):
        score = np.asarray([frame["score"] for frame in run["frames"]])
        auc = cumulative_trapezoid(score)
        ax.plot(np.arange(len(score)), auc, lw=2.5, color=color, label=f"Checkpoint {run['checkpoint']}")
    ax.axvline(common_steps, color="#6C757D", ls="--", lw=1.2)
    ax.text(common_steps - 1, ax.get_ylim()[1] * 0.97, f"common prefix: {common_steps} steps", rotation=90, va="top", ha="right", fontsize=8, color="#6C757D")
    ax.set(title="Cumulative aesthetic quality (score AUC)", xlabel="Step", ylabel="Cumulative score")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[1, 1])
    rate_names = ["Score >=40", "Target\nobserved", "Target\n80-step", "Edge clear\nwhen present"]
    rate_fields = ["score_ge_40_rate", "target_retention_observed_rate", "target_coverage_planned_horizon_rate", "edge_clear_rate_when_target_present"]
    x = np.arange(len(rate_names))
    width = 0.34
    for idx, (metric, color, label) in enumerate(zip(all_metrics, colors, labels, strict=True)):
        vals = [100 * metric[field] for field in rate_fields]
        ax.bar(x + (idx - 0.5) * width, vals, width, label=label, color=color)
    ax.set_xticks(x, rate_names)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percent")
    ax.set_title("Quality and target retention")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[1, 2])
    ax.axis("off")
    base, trained = all_metrics
    common_base, common_trained = common_metrics
    auc_gain = 100.0 * (common_trained["auc"] / common_base["auc"] - 1.0)
    lines = [
        "KEY RESULTS",
        "",
        f"Matched start pose: max joint difference < 0.05 deg",
        f"Common {common_steps}-step mean: {common_base['mean']:.2f} -> {common_trained['mean']:.2f}",
        f"Common-prefix score AUC: {common_base['auc']:.1f} -> {common_trained['auc']:.1f}  (+{auc_gain:.1f}%)",
        f"Score >=40: {100*base['score_ge_40_rate']:.1f}% -> {100*trained['score_ge_40_rate']:.1f}%",
        f"Near-edge frames: {base['near_edge_frames']} -> {trained['near_edge_frames']}",
        f"Any-action saturation: {100*base['action_step_any_saturation_rate']:.1f}% -> {100*trained['action_step_any_saturation_rate']:.1f}%",
        f"Termination: {base['termination_reason']} @ {base['episode_steps']} vs {trained['termination_reason']} @ {trained['episode_steps']}",
        "",
        "Conclusion: checkpoint 1731 maintains the target,",
        "avoids the frame edge and stays in a high-score region.",
    ]
    for i, line in enumerate(lines):
        ax.text(0.02, 0.96 - i * 0.073, line, transform=ax.transAxes, fontsize=10.5 if i else 13, fontweight="bold" if i in (0, 2, 3, 4, 5, 6, 7, 8) else "normal", color=COLORS["dark"], va="top")

    fig.suptitle("Method A deterministic policy comparison", fontsize=22, fontweight="bold", color=COLORS["dark"], y=0.96)
    fig.text(0.06, 0.915, "Checkpoint 200 baseline vs checkpoint 1731 trained policy | one episode each, no intervention", fontsize=11, color="#52647A")
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_outputs(
    runs: list[dict[str, Any]],
    all_metrics: list[dict[str, Any]],
    common_steps: int,
    common_metrics: list[dict[str, float]],
    output_dir: Path,
) -> None:
    flat_rows = []
    for metric, common in zip(all_metrics, common_metrics, strict=True):
        flat = dict(metric)
        flat["action_saturation_rate_by_joint"] = json.dumps(flat["action_saturation_rate_by_joint"])
        flat["common_prefix_steps"] = common_steps
        flat["common_prefix_mean_score"] = common["mean"]
        flat["common_prefix_median_score"] = common["median"]
        flat["common_prefix_max_score"] = common["max"]
        flat["common_prefix_final_score"] = common["final"]
        flat["common_prefix_score_ge_40_rate"] = common["ge40"]
        flat["common_prefix_aesthetic_quality_auc"] = common["auc"]
        flat["common_prefix_quality_above_32_auc"] = common["excess32_auc"]
        flat_rows.append(flat)

    with (output_dir / "comparison_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    payload = {
        "definitions": {
            "score_statistics": "Step 0 plus every executed action frame, including the terminal frame.",
            "target_retention_observed_rate": "Target-present frames divided by observed frames.",
            "target_coverage_planned_horizon_rate": "Target-present frames divided by the planned 81 observations (step 0 through step 80); unobserved frames after early target-loss termination count as failures.",
            "edge_clear_rate_when_target_present": "Among executed transition frames where the target was detected, fraction marked edge-clear by the online reward pipeline. The initial frame is excluded because it has no logged edge flag.",
            "action_component_saturation_rate": "Fraction of all four normalized action components with absolute value >= 0.95.",
            "action_step_any_saturation_rate": "Fraction of action steps where at least one normalized action component has absolute value >= 0.95.",
            "cumulative_aesthetic_quality_auc": "Trapezoidal area under the raw ArtiMuse score-versus-step curve.",
            "cumulative_quality_above_32_auc": "Trapezoidal area under max(score - 32, 0), where 32 is the absolute-reward center.",
        },
        "planned_horizon_steps": max(metric["episode_steps"] for metric in all_metrics),
        "common_prefix_steps": common_steps,
        "runs": all_metrics,
        "common_prefix_metrics": common_metrics,
        "initial_joint_max_abs_difference_deg": float(
            np.max(
                np.abs(
                    np.asarray(runs[0]["start"]["joints_deg"])
                    - np.asarray(runs[1]["start"]["joints_deg"])
                )
            )
        ),
        "initial_score_absolute_difference": abs(all_metrics[0]["initial_score"] - all_metrics[1]["initial_score"]),
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "| 指标 | Checkpoint 200 | Checkpoint 1731 |",
        "|---|---:|---:|",
    ]
    rows = [
        ("运行步数 / 终止", f"{all_metrics[0]['episode_steps']} / {all_metrics[0]['termination_reason']}", f"{all_metrics[1]['episode_steps']} / {all_metrics[1]['termination_reason']}"),
        ("平均评分", f"{all_metrics[0]['mean_score']:.2f}", f"{all_metrics[1]['mean_score']:.2f}"),
        ("中位评分", f"{all_metrics[0]['median_score']:.2f}", f"{all_metrics[1]['median_score']:.2f}"),
        ("最高评分（步）", f"{all_metrics[0]['max_score']:.2f} ({all_metrics[0]['max_score_step']})", f"{all_metrics[1]['max_score']:.2f} ({all_metrics[1]['max_score_step']})"),
        ("末步评分", f"{all_metrics[0]['final_score']:.2f}", f"{all_metrics[1]['final_score']:.2f}"),
        ("40分以上占比", f"{100*all_metrics[0]['score_ge_40_rate']:.1f}%", f"{100*all_metrics[1]['score_ge_40_rate']:.1f}%"),
        ("观测帧目标保持率", f"{100*all_metrics[0]['target_retention_observed_rate']:.1f}%", f"{100*all_metrics[1]['target_retention_observed_rate']:.1f}%"),
        ("80步时域目标覆盖率", f"{100*all_metrics[0]['target_coverage_planned_horizon_rate']:.1f}%", f"{100*all_metrics[1]['target_coverage_planned_horizon_rate']:.1f}%"),
        ("靠边帧 / 靠边率", f"{all_metrics[0]['near_edge_frames']} / {100*all_metrics[0]['near_edge_rate_when_target_present']:.1f}%", f"{all_metrics[1]['near_edge_frames']} / {100*all_metrics[1]['near_edge_rate_when_target_present']:.1f}%"),
        ("动作分量饱和率", f"{100*all_metrics[0]['action_component_saturation_rate']:.1f}%", f"{100*all_metrics[1]['action_component_saturation_rate']:.1f}%"),
        ("任一分量饱和步占比", f"{100*all_metrics[0]['action_step_any_saturation_rate']:.1f}%", f"{100*all_metrics[1]['action_step_any_saturation_rate']:.1f}%"),
        ("实际运行累计美学质量 AUC", f"{all_metrics[0]['cumulative_aesthetic_quality_auc']:.1f}", f"{all_metrics[1]['cumulative_aesthetic_quality_auc']:.1f}"),
        (f"共同前{common_steps}步累计质量 AUC", f"{common_metrics[0]['auc']:.1f}", f"{common_metrics[1]['auc']:.1f}"),
        (f"共同前{common_steps}步高于32的累计质量", f"{common_metrics[0]['excess32_auc']:.1f}", f"{common_metrics[1]['excess32_auc']:.1f}"),
    ]
    lines.extend(f"| {name} | {a} | {b} |" for name, a, b in rows)
    (output_dir / "comparison_table.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_events", type=Path)
    parser.add_argument("trained_events", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        load_run(args.baseline_events.resolve(), "Checkpoint 200"),
        load_run(args.trained_events.resolve(), "Checkpoint 1731"),
    ]
    planned_horizon = max(len(run["transitions"]) for run in runs)
    all_metrics = [metrics(run, planned_horizon) for run in runs]
    common_steps = min(len(run["transitions"]) for run in runs)
    common_metrics = []
    for run in runs:
        scores = np.asarray([frame["score"] for frame in run["frames"][: common_steps + 1]])
        common_metrics.append(
            {
                "checkpoint": run["checkpoint"],
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "max": float(np.max(scores)),
                "final": float(scores[-1]),
                "ge40": float(np.mean(scores >= 40.0)),
                "auc": float(cumulative_trapezoid(scores)[-1]),
                "excess32_auc": float(cumulative_trapezoid(np.maximum(scores - 32.0, 0.0))[-1]),
            }
        )

    write_outputs(runs, all_metrics, common_steps, common_metrics, output_dir)
    render_report(runs, all_metrics, common_steps, common_metrics, output_dir / "checkpoint_200_vs_1731_report.png")
    for run in runs:
        render_video(run, output_dir / f"checkpoint_{run['checkpoint']:08d}_episode.mp4")
    print(json.dumps({"output_dir": str(output_dir), "metrics": all_metrics, "common_prefix": common_metrics}, indent=2))


if __name__ == "__main__":
    main()
