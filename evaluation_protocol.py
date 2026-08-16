"""Versioned metadata for the local MassSpecGym evaluation identity rule."""

EVALUATION_IDENTITY_PROTOCOL = {
    "dataset": "MassSpecGym",
    "candidate_source": "MassSpecGym-supplied candidate files",
    "local_positive_identity": "exact_target_smiles_equality",
    "local_positive_cardinality": "single_positive_per_query",
    "reference_default_identity": "two_dimensional_inchikey_equivalence",
    "reference_positive_cardinality": "may_be_multi_positive",
    "reference_code_commit": "4f501b3207c9f466bc745a0353f61dbfaf150d32",
    "identity_rule_effect_quantified": False,
    "official_evaluator_equivalent": False,
}

EVALUATION_IDENTITY_PROTOCOL_NOTE = (
    "Recall and MRR use MassSpecGym-supplied candidate files with a local "
    "exact-target-SMILES single-positive rule. The reference loader defaults to "
    "two-dimensional InChIKey equivalence and may yield multiple positives. The "
    "effect of this identity-rule difference is unquantified, so these are not "
    "official-evaluator-equivalent results."
)
