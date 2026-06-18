import argparse
import os
import subprocess
import sys


def get_commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        print("Warning: Could not get git commit hash. Using 'unknown'.")
        return "unknown"

def run_command(command):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def append_device(command, device):
    if device:
        command.extend(["--device", device])

def main():
    parser = argparse.ArgumentParser(description="Unified training and evaluation pipeline.")
    parser.add_argument("dataset_type", nargs="?", default="massspecgym", help="Type of dataset (default: massspecgym)")
    parser.add_argument("--mode", choices=["train", "eval", "test", "all"], default="all", help="Execution mode: train, eval/test, or all (default: all)")
    parser.add_argument("--no-pretrain", action="store_true", help="Run without pre-trained model (equivalent to nostage1)")
    parser.add_argument("--save_dir", help="Explicit save directory (optional)")
    parser.add_argument("--device", help='Device to use, for example "cpu", "cuda", "cuda:0", or "cuda:1"')

    args = parser.parse_args()

    commit_hash = get_commit_hash()
    suffix = "_nopretrain" if args.no_pretrain else ""
    save_dir = args.save_dir or f"checkpoints_align/{commit_hash}_{args.dataset_type}{suffix}"

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
            "python", "train_align.py",
            "--dataset_type", args.dataset_type,
            "--save_dir", save_dir
        ]
        append_device(train_cmd, args.device)
        if not args.no_pretrain:
            train_cmd.extend(["--pretrained_spec", "checkpoints/model.ckpt"])

        run_command(train_cmd)

    if do_eval:
        # 2. Evaluate the aligned model
        print("=" * 56)
        print("Starting Evaluation")
        print("=" * 56)

        if args.no_pretrain:
            # Evaluate only best_model_stage2.pth for no-pretrain mode
            eval_cmd = [
                "python", "eval_align.py",
                "--dataset_type", args.dataset_type,
                "--checkpoint", os.path.join(save_dir, "best_model_stage2.pth")
            ]
            append_device(eval_cmd, args.device)
            run_command(eval_cmd)
        else:
            # Standard evaluation sequence
            eval_stage1_cmd = [
                "python", "eval_align.py",
                "--dataset_type", args.dataset_type,
                "--checkpoint", os.path.join(save_dir, "best_model_stage1.pth"),
                "--no-mces"
            ]
            append_device(eval_stage1_cmd, args.device)
            run_command(eval_stage1_cmd)

            eval_stage2_cmd = [
                "python", "eval_align.py",
                "--dataset_type", args.dataset_type,
                "--checkpoint", os.path.join(save_dir, "best_model_stage2.pth")
            ]
            append_device(eval_stage2_cmd, args.device)
            run_command(eval_stage2_cmd)

            eval_final_cmd = [
                "python", "eval_align.py",
                "--dataset_type", args.dataset_type,
                "--checkpoint", os.path.join(save_dir, "final_aligned_model.pth"),
                "--no-mces"
            ]
            append_device(eval_final_cmd, args.device)
            run_command(eval_final_cmd)

if __name__ == "__main__":
    main()
