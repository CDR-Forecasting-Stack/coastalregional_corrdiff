"""eval_io.py — transparent reader over a (possibly multi-shard) evaluation
dataset produced by eval/generate_eval_dataset.py.

Every figure/table script in this repo reads through this module instead of
opening shard_*.nc files directly, so:
  - callers never need to know how many shards exist or how samples were
    partitioned across GPU workers at generation time
  - global sample index -> (shard, local index) mapping lives in one place

Groups: low_res_input (34ch,32x32), low_res_input_upsampled (8ch,256x256),
regression (8ch,256x256), prediction (ensemble,8ch,256x256), target
(8ch,256x256). Root: sample_idx, lat, lon, time.
"""
from __future__ import annotations

import bisect
from pathlib import Path

import netCDF4 as nc
import numpy as np


class EvalDataset:
    def __init__(self, eval_dir: str | Path):
        self.eval_dir = Path(eval_dir)
        shard_paths = sorted(self.eval_dir.glob("shard_*.nc"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard_*.nc files found in {eval_dir}")
        self._ds = [nc.Dataset(p, "r") for p in shard_paths]
        for d in self._ds:
            d.set_auto_mask(False)  # skip masked-array overhead; every value here is finite
        sizes = [d.dimensions["sample"].size for d in self._ds]
        self._offsets = np.concatenate([[0], np.cumsum(sizes)])
        self.n_total = int(self._offsets[-1])
        self.ensemble_size = self._ds[0].dimensions["ensemble"].size
        incomplete = [p for p, d in zip(shard_paths, self._ds) if not getattr(d, "complete", 0)]
        if incomplete:
            print(f"WARNING: {len(incomplete)} shard(s) not marked complete "
                  f"(job may still be running or was interrupted): {incomplete}")

    def close(self):
        for d in self._ds:
            d.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _shard_for(self, gidx: int) -> tuple[int, int]:
        s = bisect.bisect_right(self._offsets, gidx) - 1
        return s, gidx - self._offsets[s]

    def random_indices(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(self.n_total, size=min(n, self.n_total), replace=False))

    def _read_group_var(self, group: str, var: str, indices: np.ndarray) -> np.ndarray:
        """indices: sorted global sample indices. Returns stacked array with
        leading axis matching `indices`, remaining axes as stored.

        Reads each needed shard's variable with a single contiguous `v[:]`
        (chunk-cache-friendly) and subsets in memory with numpy, rather than
        passing a scattered index list straight to netCDF4/HDF5 -- fancy
        ("point selection") indexing into compressed chunks was measured to
        take *minutes* per variable even for a few hundred samples (HDF5
        decompresses/caches per selected point rather than per chunk in that
        path), while a full contiguous read of the same shard takes seconds."""
        by_shard: dict[int, list[int]] = {}
        for gidx in indices:
            s, local = self._shard_for(int(gidx))
            by_shard.setdefault(s, []).append(local)
        out_by_gidx = {}
        for s, locals_ in by_shard.items():
            full = np.asarray(self._ds[s].groups[group].variables[var][:])
            for local in locals_:
                gidx = int(self._offsets[s] + local)
                out_by_gidx[gidx] = full[local]
        return np.stack([out_by_gidx[int(g)] for g in indices], axis=0)

    def get(self, group: str, var: str, indices=None) -> np.ndarray:
        """Returns (N, ...) for a list/array of global indices, or the full
        dataset if indices is None (careful: reads everything into memory)."""
        if indices is None:
            indices = np.arange(self.n_total)
        indices = np.atleast_1d(np.asarray(indices))
        return self._read_group_var(group, var, indices)

    def get_one(self, group: str, var: str, idx: int) -> np.ndarray:
        return self.get(group, var, [idx])[0]

    def wind_speed(self, group: str, indices=None) -> np.ndarray:
        u = self.get(group, "10u", indices)
        v = self.get(group, "10v", indices)
        return np.sqrt(u ** 2 + v ** 2)

    def root(self, name: str, indices=None) -> np.ndarray:
        if indices is None:
            return np.concatenate([np.array(d.variables[name][:]) for d in self._ds])
        indices = np.atleast_1d(np.asarray(indices))
        by_shard: dict[int, list[int]] = {}
        for gidx in indices:
            s, local = self._shard_for(int(gidx))
            by_shard.setdefault(s, []).append(local)
        out = {}
        for s, locals_ in by_shard.items():
            full = np.asarray(self._ds[s].variables[name][:])
            for local in locals_:
                gidx = int(self._offsets[s] + local)
                out[gidx] = full[local]
        return np.array([out[int(g)] for g in indices])
