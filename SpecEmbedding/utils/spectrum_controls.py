"""Label-independent spectrum interventions for full validation sensitivity audits."""

import hashlib
import json

import torch
from torch.utils.data import Dataset

from SpecEmbedding.data.datasets_eval import SpecSequenceDataset
from SpecEmbedding.utils.massspecgym_v15 import identity


class ControlledValidationSpectra(Dataset):
    def __init__(self, index, mode, settings):
        self.original = SpecSequenceDataset(index["sequences"])
        self.mode = mode
        count = len(self.original)
        if not count or len(index["raw_query_indices"]) != count:
            raise ValueError("Spectrum controls require every indexed validation query")
        self.metadata = {"mode": mode, "queries": count, "candidate_rows_unchanged": True,
                         "scope": "Full validation input sensitivity; not a new model or official test result"}
        if mode == "permuted":
            if count < 2:
                raise ValueError("Spectrum permutation requires at least two queries")
            seed = settings.permutation_seed
            generator = torch.Generator().manual_seed(seed)
            cycle = torch.randperm(count, generator=generator)
            self.donors = torch.empty(count, dtype=torch.long)
            self.donors[cycle] = cycle.roll(1)
            if (self.donors == torch.arange(count)).any() or len(self.donors.unique()) != count:
                raise ValueError("Spectrum permutation is not a full derangement")
            raw_donors = [index["raw_query_indices"][i] for i in self.donors.tolist()]
            mapping = json.dumps(raw_donors, separators=(",", ":")).encode()
            # Identity is used only to report residual same-structure donors, never to choose the permutation.
            identities = {seq["smiles"]: None for seq in index["sequences"]}
            identities = {smiles: identity(smiles) for smiles in identities}
            keys = [identities[seq["smiles"]] for seq in index["sequences"]]
            self.metadata.update(permutation_seed=seed, scheme="one seeded random cycle over all eligible queries",
                                 donor_raw_query_indices=raw_donors,
                                 donor_mapping_sha256=hashlib.sha256(mapping).hexdigest(), self_donors=0,
                                 same_2d_identity_donors=sum(keys[i] == keys[j] for i, j in enumerate(self.donors.tolist())),
                                 intervention="Permute all spectrum tokens including the precursor, intensity and padding mask")
        elif mode == "constant":
            width = len(index["sequences"][0]["mz"])
            mz = torch.tensor(settings.constant_mz, dtype=torch.float32)
            intensity = torch.tensor(settings.constant_intensity, dtype=torch.float32)
            if (mz.ndim != 1 or intensity.shape != mz.shape or not 1 <= len(mz) <= width
                    or not torch.isfinite(mz).all() or not torch.isfinite(intensity).all()
                    or (mz <= 0).any() or (intensity < 0).any() or not (intensity > 0).any()):
                raise ValueError("Invalid fixed spectrum control tokens")
            if any(len(seq[key]) != width for seq in index["sequences"] for key in ("mz", "intensity", "mask")):
                raise ValueError("Spectrum controls require consistent tokenizer widths")
            self.constant = {"spec_mz": torch.zeros(width), "spec_intensity": torch.zeros(width),
                             "spec_mask": torch.ones(width, dtype=torch.bool)}
            self.constant["spec_mz"][:len(mz)] = mz
            self.constant["spec_intensity"][:len(mz)] = intensity
            self.constant["spec_mask"][:len(mz)] = False  # True denotes padding in SiameseModel.
            self.metadata.update(constant_mz=mz.tolist(), constant_intensity=intensity.tolist(),
                                 token_width=width, valid_tokens=len(mz),
                                 intervention="Identical fixed tokens and padding for all queries; no measured precursor retained")
        elif mode == "precursor_only":
            width = len(index["sequences"][0]["mz"])
            if width < 1 or any(torch.as_tensor(seq[key]).shape != (width,)
                                for seq in index["sequences"] for key in ("mz", "intensity", "mask")):
                raise ValueError("Precursor control requires consistent nonempty tokenizer widths")
            # Tokenizer.get_metadata places the measured precursor first, with intensity 2.
            self.precursors = torch.tensor([seq["mz"][0] for seq in index["sequences"]], dtype=torch.float32)
            if (not torch.isfinite(self.precursors).all() or (self.precursors <= 0).any()
                    or any(seq["mask"][0] or seq["intensity"][0] != 2. for seq in index["sequences"])):
                raise ValueError("Precursor control requires an unmasked measured precursor token with intensity 2")
            self.constant = {"spec_mz": torch.zeros(width), "spec_intensity": torch.zeros(width),
                             "spec_mask": torch.ones(width, dtype=torch.bool)}
            self.constant["spec_intensity"][0] = 2.
            self.constant["spec_mask"][0] = False
            digest = hashlib.sha256(self.precursors.numpy().astype("<f4", copy=False).tobytes()).hexdigest()
            self.metadata.update(token_width=width, valid_tokens=1, precursor_intensity=2.,
                                 precursor_mz_float32_le_sha256=digest,
                                 intervention="Retain each query's measured precursor; remove all fragment tokens and peak-count information")
        else:
            raise ValueError(f"Unknown spectrum control: {mode}")

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        if self.mode == "permuted":
            sample = self.original[int(self.donors[index])]
            return {key: value for key, value in sample.items() if key != "smiles"}
        sample = {key: value.clone() for key, value in self.constant.items()}
        if self.mode == "precursor_only":
            sample["spec_mz"][0] = self.precursors[index]
        return sample
