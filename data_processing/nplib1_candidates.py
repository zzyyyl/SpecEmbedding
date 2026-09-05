from numbers import Integral


def unpack_candidate_inchikeys(candidate_value) -> tuple[list[str], str]:
    """Normalize JESTR test lists and stateful train candidate records."""

    source_format = "plain_inchikey_list"
    candidate_items = candidate_value
    if (
        isinstance(candidate_value, (list, tuple))
        and len(candidate_value) == 2
        and isinstance(candidate_value[1], (list, tuple))
        and isinstance(candidate_value[0], Integral)
    ):
        candidate_items = candidate_value[1]
        source_format = "state_and_scored_candidate_list"

    inchikeys = []
    for item in candidate_items:
        if isinstance(item, str):
            inchikey = item
        elif isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
            inchikey = item[0]
            source_format = (
                "scored_candidate_list"
                if source_format == "plain_inchikey_list"
                else source_format
            )
        else:
            raise TypeError(f"Unsupported NPLIB1 candidate entry: {item!r}")
        inchikeys.append(inchikey)
    return inchikeys, source_format
