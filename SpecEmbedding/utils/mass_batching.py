"""Full-spectrum batching with nearby-mass blocks and random block mixing."""

import hashlib
import math

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from torch.utils.data import Sampler


def molecular_exact_masses(smiles):
    """Compute masses from sanitized training structures; never trust source identity fields."""
    cache = {}
    values = []
    for text in smiles:
        if text not in cache:
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                raise ValueError(f"Cannot compute training molecular mass: {text}")
            cache[text] = rdMolDescriptors.CalcExactMolWt(mol)
        values.append(cache[text])
    return np.asarray(values, dtype=np.float64)


class MassBlockBatchSampler(Sampler[list[int]]):
    """Each query exactly once: sort -> random circular shift -> shuffle local blocks.

    Equal masses are randomly ordered before blocking. Shuffling whole small
    blocks gives each batch both nearby-mass and more widely mixed structures.
    The partial block is retained; circular shifts vary which queries occupy it.
    This changes training order only, not query eligibility or candidate pools.
    """

    def __init__(self, masses, *, batch_size, block_size, seed):
        self.masses = torch.as_tensor(masses, dtype=torch.float64, device="cpu").clone()
        if self.masses.ndim != 1 or not self.masses.numel():
            raise ValueError("Mass batching requires a nonempty one-dimensional mass array")
        if not torch.isfinite(self.masses).all() or not (self.masses > 0).all():
            raise ValueError("Mass batching requires finite, positive masses")
        if (not isinstance(batch_size, int) or not isinstance(block_size, int)
                or block_size < 1 or batch_size < 1 or batch_size % block_size):
            raise ValueError("Mass block size must be positive and divide batch size")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("Mass batching seed must be a non-negative integer")
        self.batch_size, self.block_size, self.seed = batch_size, block_size, seed
        self.epoch = 0
        self.last_audit = None
        self.mass_sha256 = hashlib.sha256(self.masses.numpy().astype("<f8").tobytes()).hexdigest()

    def __len__(self):
        return math.ceil(len(self.masses) / self.batch_size)

    def __iter__(self):
        n = len(self.masses)
        generator = torch.Generator(device="cpu").manual_seed(self.seed + self.epoch)
        self.epoch += 1
        tie_order = torch.randperm(n, generator=generator)
        order = tie_order[torch.argsort(self.masses[tie_order], stable=True)]
        order = order.roll(int(torch.randint(n, (1,), generator=generator)))
        full = n // self.block_size * self.block_size
        blocks = order[:full].reshape(-1, self.block_size)
        mixed = blocks[torch.randperm(len(blocks), generator=generator)].flatten()
        order = torch.cat((mixed, order[full:]))
        if not torch.equal(torch.bincount(order, minlength=n), torch.ones(n, dtype=torch.long)):
            raise RuntimeError("Mass batching lost or duplicated a training query")
        self.last_audit = {
            "scheme": "mass_blocks", "epoch": self.epoch, "seed": self.seed,
            "batch_size": self.batch_size, "block_size": self.block_size,
            "queries": n, "unique_queries": n, "mass_sha256": self.mass_sha256,
            "order_sha256": hashlib.sha256(order.numpy().astype("<i8").tobytes()).hexdigest(),
        }
        indices = order.tolist()
        for start in range(0, n, self.batch_size):
            yield indices[start:start + self.batch_size]
