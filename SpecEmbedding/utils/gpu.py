import csv
import re
import subprocess
from datetime import datetime
from pathlib import Path


def parse_cuda_device(device: str) -> int:
    match = re.fullmatch(r"cuda:(\d+)", device)
    if match is None:
        raise ValueError(f"Device must use the explicit cuda:N form, got: {device}")
    return int(match.group(1))


def gpu_snapshot(device: str) -> tuple[dict, str]:
    gpu_index = parse_cuda_device(device)
    query_command = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    query_result = subprocess.run(query_command, capture_output=True, text=True, check=True)
    rows = list(csv.reader([query_result.stdout.strip()]))
    if len(rows) != 1 or len(rows[0]) != 8:
        raise RuntimeError(f"Unexpected nvidia-smi output for {device}: {query_result.stdout!r}")

    row = [item.strip() for item in rows[0]]
    state = {
        "index": int(row[0]),
        "uuid": row[1],
        "name": row[2],
        "memory_total_mib": int(row[3]),
        "memory_used_mib": int(row[4]),
        "memory_free_mib": int(row[5]),
        "utilization_gpu_pct": int(row[6]),
        "temperature_c": int(row[7]),
    }
    table_result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_index)],
        capture_output=True,
        text=True,
        check=True,
    )
    snapshot_text = (
        f"timestamp: {datetime.now().astimezone().isoformat(timespec='seconds')}\n"
        f"device: {device}\n"
        f"query: {query_result.stdout.strip()}\n\n"
        f"{table_result.stdout}"
    )
    return state, snapshot_text


def require_available_gpu(
    device: str,
    min_free_mib: int,
    max_utilization: int,
    snapshot_path: Path | None = None,
) -> dict:
    state, snapshot_text = gpu_snapshot(device)
    print(
        f"GPU check {device}: free={state['memory_free_mib']} MiB, "
        f"used={state['memory_used_mib']} MiB, util={state['utilization_gpu_pct']}%",
        flush=True,
    )
    if snapshot_path is not None:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(snapshot_text, encoding="utf-8")
    if state["memory_free_mib"] < min_free_mib:
        raise RuntimeError(
            f"{device} has only {state['memory_free_mib']} MiB free; "
            f"at least {min_free_mib} MiB is required."
        )
    if state["utilization_gpu_pct"] > max_utilization:
        raise RuntimeError(
            f"{device} utilization is {state['utilization_gpu_pct']}%; "
            f"the allowed maximum is {max_utilization}%."
        )
    return state
