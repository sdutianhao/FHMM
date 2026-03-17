import argparse
import subprocess
import sys

from repo_paths import REPO_ROOT


MODEL_SPECS = {
    "M0": {
        "script": "M0.py",
        "description": "Baseline FHMM without correlation modelling.",
        "assets": ["Data.xls", "Data_history.xls", "M0_gammas.npy"],
    },
    "M1a": {
        "script": "M1a.py",
        "description": "Joint log-normal FHMM without MI weighting.",
        "assets": ["Data.xls", "Data_history.xls", "M1_gammas.npy"],
    },
    "M1b": {
        "script": "M1b.py",
        "description": "Joint log-normal FHMM with normalized MI weighting.",
        "assets": ["Data.xls", "Data_history.xls", "M1_gammas.npy"],
    },
    "M2a": {
        "script": "M2a.py",
        "description": "Gaussian-copula FHMM without MI weighting.",
        "assets": ["Data.xls", "Data_history.xls", "M2_gammas.npy"],
    },
    "M2b": {
        "script": "M2b.py",
        "description": "Gaussian-copula FHMM with raw MI weighting.",
        "assets": ["Data.xls", "Data_history.xls", "M2_gammas.npy"],
    },
    "M2c": {
        "script": "M2c.py",
        "description": "Gaussian-copula FHMM with normalized MI weighting.",
        "assets": ["Data.xls", "Data_history.xls", "M2_gammas.npy"],
    },
    "M2d": {
        "script": "M2d.py",
        "description": "Final optimized FHMM with normalized MI weighting and global tuning.",
        "assets": ["Data.xls", "Data_history.xls", "M2_gammas.npy", "data_of_3d_macro_micro_F1.txt"],
    },
}


def list_models() -> str:
    lines = ["Available models:"]
    for model_code, spec in MODEL_SPECS.items():
        lines.append(f"  {model_code:3}  {spec['description']}")
    return "\n".join(lines)


def missing_assets(model_code: str) -> list[str]:
    spec = MODEL_SPECS[model_code]
    missing = []
    for asset in spec["assets"]:
        if not (REPO_ROOT / asset).exists():
            missing.append(asset)
    return missing


def build_command(model_code: str, python_executable: str) -> list[str]:
    script_path = REPO_ROOT / MODEL_SPECS[model_code]["script"]
    return [python_executable, str(script_path)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one of the FHMM model scripts from a consistent repository root."
    )
    parser.add_argument("model", nargs="?", choices=sorted(MODEL_SPECS), help="Model code to run.")
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python interpreter used to launch the model script.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Show the available model codes and exit.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate that the required assets for the chosen model exist, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be executed without running the model.",
    )
    args = parser.parse_args()

    if args.list_models or not args.model:
        print(list_models())
        if not args.model:
            print("\nExample:")
            print("  python run_model.py M2c")
            print("  python run_model.py M2d --check-only")
        return 0

    model_code = args.model
    missing = missing_assets(model_code)
    if missing:
        print(f"[ERROR] Missing required assets for {model_code}:")
        for asset in missing:
            print(f"  - {asset}")
        return 1

    command = build_command(model_code, args.python_executable)

    print(f"[FHMM] Repository root: {REPO_ROOT}")
    print(f"[FHMM] Model: {model_code}")
    print(f"[FHMM] Script: {MODEL_SPECS[model_code]['script']}")
    print("[FHMM] Required assets:")
    for asset in MODEL_SPECS[model_code]["assets"]:
        print(f"  - {asset}")
    print(f"[FHMM] Command: {' '.join(command)}")

    if args.check_only or args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=str(REPO_ROOT))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
