#!/usr/bin/env python3
"""Submit one PBS job per Stable hyperparameter point."""

import argparse
import shlex
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-num-values", nargs="+", default=["2", "4", "6", "8", "10", "all"])
    parser.add_argument("--sampling-repeats-values", nargs="+", default=["1", "3", "5", "10"])
    parser.add_argument("--queues", nargs="+", default=["ai4090d", "ai4090"])
    parser.add_argument("--submit", action="store_true", help="Run qsub. Without this, only print commands.")
    parser.add_argument("--base-output-dir", default=str(PROJECT_ROOT / "results/stable_hparam_split_count_curves"))
    parser.add_argument("--model", default="gradient_based_joint_edge_score_stable")
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--task", default="original")
    parser.add_argument("--downstream", default="gcn")
    parser.add_argument("--scoring-model", default="gcn")
    parser.add_argument("--seeds", default="42 43 44")
    parser.add_argument("--num-repeats", default="3")
    parser.add_argument("--num-steps", default="10")
    parser.add_argument("--epochs", default="200")
    parser.add_argument("--scoring-train-epochs", default="200")
    parser.add_argument("--stability-penalty", default="0.5")
    parser.add_argument("--random-repeats", default="50")
    return parser.parse_args()


def script_for_queue(queue: str) -> Path:
    if queue == "ai4090":
        return PROJECT_ROOT / "scripts/stable_hparam_point_4090.pbs"
    if queue == "ai4090d":
        return PROJECT_ROOT / "scripts/stable_hparam_point_4090d.pbs"
    raise ValueError(f"Unsupported queue: {queue}")


def main() -> None:
    args = parse_args()
    job_index = 0

    for subset_num in args.subset_num_values:
        for sampling_repeats in args.sampling_repeats_values:
            queue = args.queues[job_index % len(args.queues)]
            script = script_for_queue(queue)
            variables = {
                "BASE_OUTPUT_DIR": args.base_output_dir,
                "MODEL": args.model,
                "DATASET": args.dataset,
                "TASK_NAME": args.task,
                "DOWNSTREAM": args.downstream,
                "SCORING_MODEL": args.scoring_model,
                "SEEDS": args.seeds,
                "NUM_REPEATS": args.num_repeats,
                "NUM_STEPS": args.num_steps,
                "EPOCHS": args.epochs,
                "SCORING_TRAIN_EPOCHS": args.scoring_train_epochs,
                "SAMPLING_REPEATS": sampling_repeats,
                "SAMPLING_SUBSET_NUM": subset_num,
                "STABILITY_PENALTY": args.stability_penalty,
                "RANDOM_REPEATS": args.random_repeats,
            }
            variable_arg = ",".join(f"{key}={value}" for key, value in variables.items())
            command = ["qsub", "-v", variable_arg, str(script)]
            print(" ".join(shlex.quote(part) for part in command))
            if args.submit:
                subprocess.run(command, check=True)
            job_index += 1


if __name__ == "__main__":
    main()
