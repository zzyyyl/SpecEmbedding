import argparse
import os
import subprocess
import sys
from pathlib import Path

from SpecEmbedding.config import config

REPOSITORY_ROOT = Path(__file__).resolve().parent


def get_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPOSITORY_ROOT,
        ).decode("ascii").strip()
    except Exception:
        print("Warning: Could not get git commit hash. Using 'unknown'.")
        return "unknown"

def run_command(command):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=REPOSITORY_ROOT)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def append_optional_arg(command, flag, value):
    if value:
        command.extend([flag, value])

def main():
    parser = argparse.ArgumentParser(description="Unified training and evaluation pipeline.")
    parser.add_argument("dataset_type", nargs="?", default="massspecgym", help="Type of dataset (default: massspecgym)")
    parser.add_argument("--mode", choices=["train", "eval", "test", "all"], default="all", help="Execution mode: train, eval/test, or all (default: all)")
    parser.add_argument("--no-pretrain", action="store_true", help="Run without pre-trained model (equivalent to nostage1)")
    parser.add_argument("--save_dir", help="Explicit save directory (optional)")
    parser.add_argument("--data_path", default=config.data.data_path, help="Processed dataset root")
    parser.add_argument("--cache_path", default=config.data.cache_path, help="Reusable TokenSet cache root")
    parser.add_argument("--tokenset_cache", help="Exact TokenSet cache file for alignment training")
    parser.add_argument("--pretrained_spec", default="checkpoints/model.ckpt", help="Pretrained spectrum encoder checkpoint")
    parser.add_argument(
        "--candidate_type",
        choices=["mass", "formula", "supplied"],
        default=None,
        help="Defaults to supplied for NPLIB1 and mass otherwise.",
    )
    parser.add_argument("--candidate_path", help="Custom evaluation candidates pickle")
    parser.add_argument("--device", help='Device to use, for example "cpu", "cuda", "cuda:0", or "cuda:1"')
    parser.add_argument("--mol_norm_type", choices=["layernorm", "rmsnorm"], help="Normalization used in the molecule GINE encoder.")
    parser.add_argument("--seed", type=int, default=config.general.seed, help="Random seed for alignment training")

    args = parser.parse_args()
    if args.candidate_type is None:
        args.candidate_type = "supplied" if args.dataset_type == "nplib1" else "mass"
    if args.seed < 0:
        parser.error("--seed must be a non-negative integer")

    commit_hash = get_commit_hash()
    suffix = "_nopretrain" if args.no_pretrain else ""
    seed_suffix = "" if args.seed == config.general.seed else f"_seed{args.seed}"
    save_dir = args.save_dir or f"checkpoints_align/{commit_hash}_{args.dataset_type}{suffix}{seed_suffix}"

    do_train = args.mode in ["train", "all"]
    do_eval = args.mode in ["eval", "test", "all"]

    if do_train:
        print("=" * 56)
        print("Starting Training Pipeline" + (" (without pre-trained model)" if args.no_pretrain else ""))
        print(f"Target Directory: {save_dir}")
        print(f"Dataset Type: {args.dataset_type}")
        print("=" * 56)

        # 1. Train the alignment model
        train_cmd = [
            sys.executable, str(REPOSITORY_ROOT / "train_align.py"),
            "--dataset_type", args.dataset_type,
            "--data_path", args.data_path,
            "--cache_path", args.cache_path,
            "--save_dir", save_dir,
            "--seed", str(args.seed),
        ]
        append_optional_arg(train_cmd, "--tokenset_cache", args.tokenset_cache)
        append_optional_arg(train_cmd, "--device", args.device)
        append_optional_arg(train_cmd, "--mol_norm_type", args.mol_norm_type)
        if not args.no_pretrain:
            append_optional_arg(train_cmd, "--pretrained_spec", args.pretrained_spec)

        run_command(train_cmd)

    if do_eval:
        # 2. Evaluate the aligned model
        print("=" * 56)
        print("Starting Evaluation")
        print("=" * 56)

        if args.no_pretrain:
            # Evaluate only best_model_stage2.pth for no-pretrain mode
            eval_cmd = [
                sys.executable, str(REPOSITORY_ROOT / "eval_align.py"),
                "--dataset_type", args.dataset_type,
                "--data_path", args.data_path,
                "--checkpoint", os.path.join(save_dir, "best_model_stage2.pth"),
                "--candidate_type", args.candidate_type,
                "--no-mces"
            ]
            append_optional_arg(eval_cmd, "--candidate_path", args.candidate_path)
            append_optional_arg(eval_cmd, "--device", args.device)
            append_optional_arg(eval_cmd, "--mol_norm_type", args.mol_norm_type)
            run_command(eval_cmd)
        else:
            # Standard evaluation sequence
            eval_stage1_cmd = [
                sys.executable, str(REPOSITORY_ROOT / "eval_align.py"),
                "--dataset_type", args.dataset_type,
                "--data_path", args.data_path,
                "--checkpoint", os.path.join(save_dir, "best_model_stage1.pth"),
                "--candidate_type", args.candidate_type,
                "--no-mces"
            ]
            append_optional_arg(eval_stage1_cmd, "--candidate_path", args.candidate_path)
            append_optional_arg(eval_stage1_cmd, "--device", args.device)
            append_optional_arg(eval_stage1_cmd, "--mol_norm_type", args.mol_norm_type)
            run_command(eval_stage1_cmd)

            eval_stage2_cmd = [
                sys.executable, str(REPOSITORY_ROOT / "eval_align.py"),
                "--dataset_type", args.dataset_type,
                "--data_path", args.data_path,
                "--checkpoint", os.path.join(save_dir, "best_model_stage2.pth"),
                "--candidate_type", args.candidate_type,
                "--no-mces"
            ]
            append_optional_arg(eval_stage2_cmd, "--candidate_path", args.candidate_path)
            append_optional_arg(eval_stage2_cmd, "--device", args.device)
            append_optional_arg(eval_stage2_cmd, "--mol_norm_type", args.mol_norm_type)
            run_command(eval_stage2_cmd)

            eval_final_cmd = [
                sys.executable, str(REPOSITORY_ROOT / "eval_align.py"),
                "--dataset_type", args.dataset_type,
                "--data_path", args.data_path,
                "--checkpoint", os.path.join(save_dir, "final_aligned_model.pth"),
                "--candidate_type", args.candidate_type,
                "--no-mces"
            ]
            append_optional_arg(eval_final_cmd, "--candidate_path", args.candidate_path)
            append_optional_arg(eval_final_cmd, "--device", args.device)
            append_optional_arg(eval_final_cmd, "--mol_norm_type", args.mol_norm_type)
            run_command(eval_final_cmd)

if __name__ == "__main__":
    main()
