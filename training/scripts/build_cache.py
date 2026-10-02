#!/usr/bin/env python3
"""Build the memory-mapped .npy caches next to regional_train.nc once, on a CPU node, before training.

The dataset class (physicsnemo/regional_hrrr.py) memory-maps `<data>.input.npy`, `<data>.output.npy`
and `<data>.invariant.npy`; building them lazily inside a multi-GPU job would make every rank read
and stack the whole NetCDF file. The output group is large (120,000 x 8 x 256 x 256 float32 = 252 GB)
and is stacked in memory before it is written: request >= 600 GB of RAM.

    PHYSICSNEMO_CORRDIFF_DIR=/path/to/physicsnemo/examples/weather/corrdiff \\
        python build_cache.py --data /path/to/regional_train.nc
"""
import argparse, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="path to regional_train.nc")
a = ap.parse_args()
sys.path.insert(0, os.environ["PHYSICSNEMO_CORRDIFF_DIR"])
from datasets.regional_hrrr import _load_dataset  # noqa: E402

t0 = time.time()
for group, kwargs in [("input", {}), ("output", {}),
                      ("invariant", {"variables": ["elev_mean", "lsm_mean"], "stack_axis": 0})]:
    print(f"[{time.time()-t0:.0f}s] building cache for group={group} ...", flush=True)
    data, variables = _load_dataset(a.data, group, **kwargs)
    print(f"[{time.time()-t0:.0f}s] group={group} cached, shape={data.shape}", flush=True)
print("all caches built", flush=True)
