import argparse
import subprocess
import sys
from pathlib import Path


def get_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception:
        print("Warning: Could not get git commit hash. Using 'unknown'.")
        return "unknown"


def run_command(command, dry_run: bool = False):
    print(f"Running: {' '.join(command)}")
    if dry_run:
        return
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def append_optional_arg(command, flag, value):
    if value is not None:
        command.extend([flag, str(value)])


def ensure_file(path: str | Path, description: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def candidate_label(args):
    if args.candidate_path:
        return Path(args.candidate_path).stem
    return args.candidate_type


def topk_suffix(pre_top_k):
    if pre_top_k is None:
        return ""
    return f"_topk{pre_top_k}"


def limit_suffix(limit):
    if limit is None or limit <= 0:
        return ""
    return f"_limit{limit}"


def append_name_suffix(name: str, suffix: str):
    if not suffix or name.endswith(suffix) or f"{suffix}_" in name:
        return name
    return f"{name}{suffix}"


def append_path_suffix(path: Path, suffix: str):
    if not suffix or path.name.endswith(suffix):
        return path
    return path.with_name(f"{path.name}{suffix}")


def default_align_save_dir(args):
    suffix = "_nopretrain" if args.no_pretrain else ""
    return Path("checkpoints_align") / f"{get_commit_hash()}_{args.dataset_type}{suffix}"


def append_output_suffixes(name_or_path, suffixes):
    result = name_or_path
    for suffix in suffixes:
        if isinstance(result, Path):
            result = append_path_suffix(result, suffix)
        else:
            result = append_name_suffix(result, suffix)
    return result


def resolve_paths(args):
    align_save_dir = Path(args.align_save_dir) if args.align_save_dir else default_align_save_dir(args)
    checkpoint = Path(args.checkpoint) if args.checkpoint else align_save_dir / args.align_checkpoint_name

    label = candidate_label(args)
    output_suffixes = [topk_suffix(args.pre_top_k), limit_suffix(args.limit)]
    run_name = append_output_suffixes(args.run_name or f"{align_save_dir.name}_{label}", output_suffixes)
    cache_dir = append_output_suffixes(Path(args.cache_dir), output_suffixes) if args.cache_dir else Path("rerank_cache") / run_name
    save_dir = append_output_suffixes(Path(args.save_dir), output_suffixes) if args.save_dir else Path("checkpoints_rerank") / run_name

    caches = {
        split: cache_dir / f"{args.dataset_type}_{label}_{split}.pt"
        for split in ["train", "val", "test"]
    }
    return align_save_dir, checkpoint, cache_dir, save_dir, caches


def build_prepare_command(args, checkpoint, split, save_path):
    command = [
        sys.executable,
        "prepare_rerank_cache.py",
        "--checkpoint",
        str(checkpoint),
        "--dataset_type",
        args.dataset_type,
        "--split",
        split,
        "--candidate_type",
        args.candidate_type,
        "--save_path",
        str(save_path),
    ]
    append_optional_arg(command, "--data_path", args.data_path)
    append_optional_arg(command, "--device", args.device)
    append_optional_arg(command, "--candidate_path", args.candidate_path)
    append_optional_arg(command, "--pre_top_k", args.pre_top_k)
    append_optional_arg(command, "--limit", args.limit)
    append_optional_arg(command, "--mol_norm_type", args.mol_norm_type)
    append_optional_arg(command, "--mol_norm_eps", args.mol_norm_eps)

    if split == "train":
        command.append("--force_include_positive")
    else:
        command.append("--no-force_include_positive")
    return command


def build_train_command(args, save_dir, caches):
    command = [
        sys.executable,
        "train_rerank.py",
        "--train_cache",
        str(caches["train"]),
        "--val_cache",
        str(caches["val"]),
        "--save_dir",
        str(save_dir),
    ]
    append_optional_arg(command, "--device", args.device)
    append_optional_arg(command, "--model_type", args.model_type)
    return command


def build_eval_command(args, save_dir, caches):
    command = [
        sys.executable,
        "eval_rerank.py",
        "--cache",
        str(caches["test"]),
        "--checkpoint",
        str(save_dir / "best_reranker.pth"),
        "--save_dir",
        str(save_dir),
    ]
    append_optional_arg(command, "--device", args.device)
    if args.mces:
        command.append("--mces")
    else:
        command.append("--no-mces")
    return command


def main():
    parser = argparse.ArgumentParser(description="Unified rerank cache/training/evaluation pipeline.")
    parser.add_argument("dataset_type", nargs="?", default="massspecgym", help="Dataset type")
    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "eval", "test", "all"],
        default="all",
        help="Execution mode. 'test' is an alias for 'eval'.",
    )
    parser.add_argument("--candidate_type", choices=["mass", "formula"], default="mass")
    parser.add_argument("--candidate_path", help="Custom candidates pickle. Overrides provider candidates.")
    parser.add_argument("--checkpoint", help="Aligned model checkpoint. Overrides --align_save_dir.")
    parser.add_argument("--align_save_dir", help="Directory produced by run_pipeline.py.")
    parser.add_argument("--align_checkpoint_name", default="best_model_stage2.pth")
    parser.add_argument("--no-pretrain", action="store_true", help="Infer run_pipeline no-pretrain save directory.")
    parser.add_argument("--cache_dir", help="Directory for generated rerank caches.")
    parser.add_argument("--save_dir", help="Directory for reranker checkpoints and logs.")
    parser.add_argument("--run_name", help="Name used for default cache/save directories.")
    parser.add_argument("--data_path", help="Processed data directory override.")
    parser.add_argument("--device", help='Device to use, for example "cpu", "cuda", "cuda:0", or "cuda:1".')
    parser.add_argument("--mol_norm_type", choices=["layernorm", "rmsnorm"], help="Must match the alignment checkpoint.")
    parser.add_argument("--mol_norm_eps", type=float, help="Must match the alignment checkpoint.")
    parser.add_argument("--model_type", choices=["transformer", "pointwise"], help="Reranker model variant.")
    parser.add_argument(
        "--pre_top_k",
        "--topk",
        dest="pre_top_k",
        type=int,
        help="Number of base-retrieved candidates kept for reranking; output dirs get a _topkN suffix.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Debug mode: only keep the first N matched spectra per split; output dirs get a _limitN suffix.",
    )
    parser.add_argument("--mces", action="store_true", help="Enable MCES@1 during final rerank evaluation.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    if args.pre_top_k is not None and args.pre_top_k <= 0:
        parser.error("--pre_top_k/--topk must be greater than 0")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be greater than or equal to 0")

    do_prepare = args.mode in ["prepare", "all"]
    do_train = args.mode in ["train", "all"]
    do_eval = args.mode in ["eval", "test", "all"]

    align_save_dir, checkpoint, cache_dir, save_dir, caches = resolve_paths(args)

    print("=" * 56)
    print("Rerank Pipeline")
    print(f"Dataset Type       : {args.dataset_type}")
    print(f"Candidate Source   : {candidate_label(args)}")
    print(f"Align Save Dir     : {align_save_dir}")
    print(f"Align Checkpoint   : {checkpoint}")
    print(f"Rerank Cache Dir   : {cache_dir}")
    print(f"Rerank Save Dir    : {save_dir}")
    if args.pre_top_k is not None:
        print(f"Pre Top K          : {args.pre_top_k}")
    if args.limit is not None and args.limit > 0:
        print(f"Debug Limit        : first {args.limit} matched spectra per split")
    print("=" * 56)

    if (do_prepare or args.mode == "all") and not args.dry_run:
        ensure_file(checkpoint, "Aligned model checkpoint")

    if do_prepare:
        if not args.dry_run:
            cache_dir.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            run_command(build_prepare_command(args, checkpoint, split, caches[split]), dry_run=args.dry_run)

    if do_train:
        if not args.dry_run:
            ensure_file(caches["train"], "Train rerank cache")
            ensure_file(caches["val"], "Validation rerank cache")
        if not args.dry_run:
            save_dir.mkdir(parents=True, exist_ok=True)
        run_command(build_train_command(args, save_dir, caches), dry_run=args.dry_run)

    if do_eval:
        if not args.dry_run:
            ensure_file(caches["test"], "Test rerank cache")
            ensure_file(save_dir / "best_reranker.pth", "Best reranker checkpoint")
        run_command(build_eval_command(args, save_dir, caches), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
