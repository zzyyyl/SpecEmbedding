from collections import Counter

import numpy as np


def filter_classified_validation(classified_data: dict, val_raw: list, query_indices: list[int]):
    excluded = sorted(set(query_indices))
    invalid = [index for index in excluded if index < 0 or index >= len(val_raw)]
    if invalid:
        raise ValueError(f"Excluded validation query indices are out of range: {invalid}")
    if not excluded:
        return classified_data, {"query_indices": [], "smiles": [], "keys": []}

    excluded_smiles = [val_raw[index].get("smiles") for index in excluded]
    if any(not smiles for smiles in excluded_smiles):
        raise ValueError("Every excluded validation query must have a SMILES label")

    total_counts = Counter(spectrum.get("smiles") for spectrum in val_raw)
    cached_counts = Counter(
        sequence["smiles"]
        for sequences in classified_data["val_data"].values()
        for sequence in sequences
    )
    if cached_counts != total_counts:
        all_smiles = set(total_counts) | set(cached_counts)
        differing_labels = sum(
            total_counts[smiles] != cached_counts[smiles]
            for smiles in all_smiles
        )
        raise RuntimeError(
            "Raw validation data and the cached TokenSet have different SMILES counts: "
            f"raw={sum(total_counts.values())}, cached={sum(cached_counts.values())}, "
            f"differing_labels={differing_labels}"
        )
    excluded_counts = Counter(excluded_smiles)
    partial = {
        smiles: (excluded_counts[smiles], total_counts[smiles])
        for smiles in excluded_counts
        if excluded_counts[smiles] != total_counts[smiles]
    }
    if partial:
        raise ValueError(
            "The cached classified validation data can only exclude complete molecule groups; "
            f"partial exclusions were requested: {partial}"
        )

    excluded_smiles_set = set(excluded_smiles)
    val_data = classified_data["val_data"]
    excluded_keys = []
    matched_smiles = set()
    for key, sequences in val_data.items():
        sequence_smiles = {sequence["smiles"] for sequence in sequences}
        if len(sequence_smiles) != 1:
            raise RuntimeError(f"Validation key {key} contains multiple SMILES labels")
        smiles = next(iter(sequence_smiles))
        if smiles in excluded_smiles_set:
            excluded_keys.append(key)
            matched_smiles.add(smiles)

    missing_smiles = sorted(excluded_smiles_set - matched_smiles)
    if missing_smiles:
        raise RuntimeError(f"Excluded validation SMILES are absent from the TokenSet: {missing_smiles}")

    excluded_key_set = set(excluded_keys)
    original_keys = classified_data["val_keys"]
    filtered_keys = [key for key in original_keys if key not in excluded_key_set]
    if not filtered_keys:
        raise ValueError("Validation exclusion removed every classified validation key")
    if isinstance(original_keys, np.ndarray):
        filtered_keys = np.asarray(filtered_keys, dtype=original_keys.dtype)

    filtered = {
        **classified_data,
        "val_data": {
            key: sequences
            for key, sequences in val_data.items()
            if key not in excluded_key_set
        },
        "val_keys": filtered_keys,
    }
    report = {
        "query_indices": excluded,
        "smiles": sorted(excluded_smiles_set),
        "keys": sorted(int(key) for key in excluded_keys),
        "original_num_keys": len(original_keys),
        "filtered_num_keys": len(filtered_keys),
    }
    return filtered, report
