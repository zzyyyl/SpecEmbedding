import logging
from typing import Never, Optional, Union

from matchms import Spectrum
from rdkit import Chem


def is_valid_smiles(smiles: str) -> bool:
    """校验 SMILES 的合法性"""
    if not smiles or smiles.upper() in ['N/A', 'NA']:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def canonicalize_smiles(*invalid_args: Never, smiles: str = None, spectrum: Spectrum = None) -> Optional[Union[str, Spectrum]]:
    if invalid_args:
        raise ValueError("Invalid arguments provided. Please use either 'smiles' or 'spectrum' keyword argument.")
    if not smiles and not spectrum:
        raise ValueError("Either 'smiles' or 'spectrum' must be provided.")
    if smiles is not None and spectrum is not None:
        raise ValueError("Only one of 'smiles' or 'spectrum' should be provided.")

    if spectrum is not None:
        smiles = spectrum.get('smiles')

    if smiles is None:
        logging.warning("No SMILES found in the provided spectrum.")
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.warning(f"Invalid SMILES: {smiles}")
        return None

    canonicalized = Chem.MolToSmiles(mol, canonical=True)

    if spectrum is not None:
        spectrum.set('smiles', canonicalized)
        return spectrum

    return canonicalized
