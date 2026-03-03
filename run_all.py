import subprocess
import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent

STEP_MAP = {
    "inspect": "00_inspect_dataset.py",
    "prepare": "01_prepare_data.py",
    "check_mini": "02_check_features_on_mini.py",
    "build_features": "03_build_feature_dataset.py",
    "train": "04_train_gated_experts.py",
    "eval": "05_eval_by_snr.py",
    "confidence": "06_analyze_confidence.py",
    "snr_analysis": "07_analyze_by_snr.py",
    "ablation": "08_targeted_ablation.py",
}

DEFAULT_ORDER = [
    "inspect",
    "prepare",
    "build_features",
    "train",
    "eval",
    "confidence",
    "snr_analysis",
]

FULL_ORDER = [
    "inspect",
    "prepare",
    "check_mini",
    "build_features",
    "train",
    "eval",
    "confidence",
    "snr_analysis",
    "ablation",
]


def run_step(step_key: str):
    script_name = STEP_MAP[step_key]
    script_path = ROOT / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"\n{'=' * 60}")
    print(f"Running: {script_name}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT)
    )

    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {script_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the AMC pipeline scripts in order."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=list(STEP_MAP.keys()),
        help="Run only selected steps in the given order."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full pipeline including mini-check and targeted ablation."
    )
    args = parser.parse_args()

    if args.steps:
        steps_to_run = args.steps
    elif args.full:
        steps_to_run = FULL_ORDER
    else:
        steps_to_run = DEFAULT_ORDER

    print("Selected steps:")
    for s in steps_to_run:
        print(f"  - {s} -> {STEP_MAP[s]}")

    for step in steps_to_run:
        run_step(step)

    print("\nAll selected steps completed successfully.")


if __name__ == "__main__":
    main()