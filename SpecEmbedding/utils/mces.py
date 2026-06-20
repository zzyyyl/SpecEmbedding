import logging
import os
from concurrent.futures import ProcessPoolExecutor

import pulp
from myopic_mces.myopic_mces import MCES
from tqdm import tqdm

_mces_solver = None


def resolve_mces_solver():
    solvers = pulp.listSolvers(onlyAvailable=True)
    if not solvers:
        raise RuntimeError("No available pulp solver found for MCES calculation.")
    return "MOSEK" if "MOSEK" in solvers else solvers[0]


def get_mces_solver():
    global _mces_solver
    if _mces_solver is None:
        _mces_solver = resolve_mces_solver()
    return _mces_solver


def mces_worker(smiles_pair):
    pred_smiles, true_smiles = smiles_pair
    if pred_smiles == true_smiles:
        return 0.0, False
    try:
        solver = get_mces_solver()
        retval = MCES(
            smiles1=pred_smiles,
            smiles2=true_smiles,
            threshold=15,
            always_stronger_bound=True,
            solver=solver,
            solver_options=dict(msg=0),
        )
        return retval[1], False
    except Exception:
        return 0.0, True


def compute_mces(pairs, label: str = "MCES", max_workers: int | None = None):
    if not pairs:
        return None

    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, 16)

    solver = get_mces_solver()
    logging.info("Computing %s for %s pairs using %s solver...", label, len(pairs), solver)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(mces_worker, pairs), total=len(pairs), desc=label, ascii=True))

    mces = sum(item[0] for item in results) / len(pairs)
    errors = sum(item[1] for item in results)
    if errors:
        logging.warning("%s calculation errors: %s/%s", label, errors, len(pairs))
    return mces
