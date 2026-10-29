#!/usr/bin/env python3
"""Compute a small production LCT volume from an ideal impulse, not a capture."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from util.lct import reconstruct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    parser.add_argument('--target', type=int, nargs=3, default=[4, 1, 2], metavar=('RANGE', 'Y', 'X'))
    args = parser.parse_args()
    shape = (8, 3, 5)
    if any(i < 0 or i >= n for i, n in zip(args.target, shape)):
        parser.error('target must lie inside the fixed [8, 3, 5] grid')
    if args.output.exists():
        parser.error('output exists; choose a new directory')
    meas = np.zeros(shape)
    meas[tuple(args.target)] = 1.0
    # Same geometry/calibration constants as the supported resolution_corner
    # path, applied to an ideal numerical impulse rather than measured echoes.
    max_dist = int(1.0 * 340/2 / 16)
    start, stop = 1000/18000*max_dist, 3800/18000*max_dist
    correction = 1 + (1000/3800)**2
    params = dict(S=2, slope_x=.9/stop*correction, slope_y=.95/stop*correction,
                  start=start, stop=stop, snr=8000.)
    out = reconstruct(meas, **params)
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output/'measurement.npy', meas, allow_pickle=False)
    np.save(args.output/'volume.npy', out, allow_pickle=False)
    np.savetxt(args.output/'max_projection_yx.csv', out.max(axis=0), delimiter=',')
    digest = lambda name: hashlib.sha256((args.output/name).read_bytes()).hexdigest()
    report = dict(evidence='production LCT on ideal synthetic impulse; no recorded scene or hardware',
                  axes=['range', 'y', 'x'], coordinate_units='grid indices; not measured position',
                  input_shape=list(shape), input_target=args.target, output_shape=list(out.shape),
                  peak_index=list(map(int, np.unravel_index(out.argmax(), out.shape))),
                  peak_value=float(out.max()), sum=float(out.sum()), parameters=params,
                  numpy_version=np.__version__,
                  measurement_sha256=digest('measurement.npy'), volume_sha256=digest('volume.npy'))
    (args.output/'measurements.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
