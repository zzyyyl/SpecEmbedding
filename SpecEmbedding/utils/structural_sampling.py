"""Explicit mixed sampling of distinct natural negative identities, with private per-query RNG."""

import copy
from collections import OrderedDict
from pathlib import Path

import numpy as np

from SpecEmbedding.utils.fingerprint_cache import fingerprint_options
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.training_candidates import NegativeCandidates, TrainingCandidateIndex, _integer
from SpecEmbedding.utils.training_similarity import canonical_groups, load_training_similarity_cache, metadata_digest

SAMPLING_POLICY = ('Mixed distinct negative 2D identities: canonical identity order; uniform without replacement from '
                   'exact-Tanimoto near pool, then all remaining groups; shuffle groups, uniform original source entry; '
                   'min(budget, available); private NumPy SeedSequence(seed, epoch, raw_query_index)')


def validate_structural_sampling(settings, negative_count):
    keys = {'type', 'near_count', 'near_pool_size', 'fingerprint_radius', 'fingerprint_bits', 'cache_directory'}
    if not isinstance(settings, dict) or set(settings) != keys or settings['type'] != 'tanimoto_mixed':
        raise ValueError('Incomplete or unknown structural sampling configuration')
    near = _integer(settings['near_count'], 'near_count', 1)
    pool = _integer(settings['near_pool_size'], 'near_pool_size', 1)
    if not near <= min(pool, negative_count) or pool > 255:
        raise ValueError('Structural near count/pool must fit the declared negative budget and natural pool')
    fingerprint_options(settings['fingerprint_radius'], settings['fingerprint_bits'])
    if settings['fingerprint_bits'] >= 65535:
        raise ValueError('Structural fingerprint width exceeds exact cache counters')
    if not isinstance(settings['cache_directory'], str) or not Path(settings['cache_directory']).is_absolute():
        raise ValueError('Structural sampling requires an explicit absolute cache directory')


class StructuralTrainingCandidateIndex(TrainingCandidateIndex):
    def __init__(self, base, cache, receipt, settings):
        validate_structural_sampling(settings, 255)
        if (receipt['source']['metadata_sha256'] != base.provenance['sha256']
                or receipt['source']['metadata_content_sha256'] != metadata_digest(base.metadata)
                or cache.root != Path(receipt['directory'])
                or Path(settings['cache_directory']).resolve() != cache.root
                or receipt['source']['fingerprint_cache']['provenance']['options']
                != fingerprint_options(settings['fingerprint_radius'], settings['fingerprint_bits'])):
            raise ValueError('Structural sampling cache does not match this training index/configuration')
        super().__init__(base.metadata, base.provenance, pool_cache_size=base.pool_cache_size)
        self.cache = cache
        self._structural_pool_cache = OrderedDict()
        self.sampling_settings = copy.deepcopy(settings)
        self.provenance.update(sampling_policy=SAMPLING_POLICY, structural_sampling={
            'settings': copy.deepcopy(settings), 'cache': copy.deepcopy(receipt),
            'sampler_source_sha256': sha256_file(Path(__file__)),
        })

    def _groups(self, row):
        if row in self._structural_pool_cache:
            groups = self._structural_pool_cache.pop(row)
        else:
            groups = canonical_groups(self.metadata, row)
        self._structural_pool_cache[row] = groups
        if len(self._structural_pool_cache) > self.pool_cache_size:
            self._structural_pool_cache.popitem(last=False)
        return groups

    def sample(self, raw_query_index, *, negative_count, seed, epoch):
        query = _integer(raw_query_index, 'raw_query_index')
        if query >= len(self):
            raise ValueError('Training query index is outside the complete split')
        budget = _integer(negative_count, 'negative_count', 1)
        seed, epoch = _integer(seed, 'seed'), _integer(epoch, 'epoch', 1)
        row = int(self.metadata['query_target_rows'][query])
        groups = self._groups(row)
        b = min(budget, len(groups))
        near = self.cache.row(row)[:min(self.sampling_settings['near_pool_size'], len(groups))]
        generator = np.random.default_rng(np.random.SeedSequence([seed, epoch, query]))
        chosen_near = generator.choice(near, size=min(self.sampling_settings['near_count'], b), replace=False)
        remaining = np.ones(len(groups), dtype=np.bool_)
        remaining[chosen_near] = False
        chosen_rest = generator.choice(np.flatnonzero(remaining), size=b-len(chosen_near), replace=False)
        chosen = generator.permutation(np.concatenate([chosen_near, chosen_rest]))
        molecules, positions, identities = [], [], []
        for ordinal in chosen:
            key, columns = groups[int(ordinal)]
            col = columns[int(generator.integers(len(columns)))]
            molecules.append(int(self.metadata['candidate_indices'][row, col]))
            positions.append(int(self.metadata['source_positions'][row, col]))
            identities.append(key)
        molecule_array = np.asarray(molecules, dtype=np.int32)
        position_array = np.asarray(positions, dtype=np.int16)
        molecule_array.setflags(write=False)
        position_array.setflags(write=False)
        return NegativeCandidates(molecule_array, position_array, tuple(identities))


def load_structural_training_candidates(base, settings):
    validate_structural_sampling(settings, 255)
    cache, receipt = load_training_similarity_cache(base, settings['cache_directory'],
                                                    radius=settings['fingerprint_radius'], bits=settings['fingerprint_bits'])
    return StructuralTrainingCandidateIndex(base, cache, receipt, settings)


def reference_structural_sample(index, raw_query_index, *, negative_count, seed, epoch):
    """Independent full-source reconstruction for audits; does not call sampler/group helper or its LRU."""
    metadata = index.metadata
    row = int(metadata['query_target_rows'][raw_query_index])
    by_key = {}
    for mol, pos in zip(metadata['candidate_indices'][row], metadata['source_positions'][row], strict=True):
        if mol >= 0:
            key = metadata['mol_identity_2d'][int(mol)]
            if key != metadata['target_identity_2d'][row]:
                by_key.setdefault(key, []).append((int(mol), int(pos)))
    keys = sorted(by_key)
    b = min(negative_count, len(keys))
    pool = index.cache.row(row)[:min(index.sampling_settings['near_pool_size'], len(keys))].tolist()
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, raw_query_index]))
    first = [pool[int(i)] for i in rng.choice(len(pool), size=min(index.sampling_settings['near_count'], b), replace=False)]
    left = [i for i in range(len(keys)) if i not in first]
    second = [left[int(i)] for i in rng.choice(len(left), size=b-len(first), replace=False)]
    chosen = np.asarray(first + second, dtype=np.int64)
    rng.shuffle(chosen)
    molecules, positions, identities = [], [], []
    for group in chosen:
        key = keys[int(group)]
        entries = sorted(by_key[key], key=lambda pair: pair[1])
        molecule, position = entries[int(rng.integers(len(entries)))]
        molecules.append(molecule)
        positions.append(position)
        identities.append(key)
    return NegativeCandidates(np.asarray(molecules, dtype=np.int32), np.asarray(positions, dtype=np.int16), tuple(identities))
