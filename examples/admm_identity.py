#!/usr/bin/env python3
"""Recompute bounded production ADMM on an ideal identity-PSF volume."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ADMMReconstruction import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='new output directory')
    parser.add_argument('--level', type=float, default=20., help='constant ideal measurement, 0 to 1000')
    args = parser.parse_args()
    if not np.isfinite(args.level) or not 0 <= args.level <= 1000:
        parser.error('level must be finite and between 0 and 1000')
    if args.output.exists():
        parser.error('output exists; choose a new directory')
    data = args.output / 'data'
    data.mkdir(parents=True)
    measurement = np.full((8, 4, 4), args.level)
    np.save(data/'letter_H_out.npy', dict(meas=measurement, T=1., v=340.,
            fend=3800., B=18000., fstart=1000.))
    np.save(data/'psf_gaussian.npy', np.ones((1, 1)))
    with (args.output/'solver.log').open('x') as log, contextlib.redirect_stdout(log):
        result = run('letter_H', data_dir=data, output_dir=args.output/'refined',
                     plotting=False, max_iters=1000,
                     absolute_tolerance=1e-6, relative_tolerance=1e-6)
    np.save(args.output/'volume.npy', result['out'], allow_pickle=False)
    # For a constant volume the TV term is zero. With identity PSF, the
    # independent scalar optimum is max(level-lambda_l1,0), lambda_l1=.9.
    reference = max(args.level-.9, 0)
    error = float(np.max(abs(result['out']-reference)))
    # Check the solution separately from the solver's norm-based stop gates.
    if error > 1e-4 + 1e-6*abs(reference):
        raise RuntimeError('ADMM solution exceeds the declared scalar-reference tolerance')
    report = dict(evidence='production ADMM on ideal identity PSF; no recorded scene or physical accuracy',
                  measurement_level=args.level, shape=list(measurement.shape),
                  analytic_scalar_optimum=reference, maximum_solution_error=error,
                  solver_tolerance=dict(absolute=1e-6, relative=1e-6),
                  reference_tolerance=dict(absolute=1e-4, relative=1e-6),
                  converged=result['converged'], iterations=result['iter']+1,
                  final_primal_residual=result['primal_residual'][-1],
                  final_dual_residual=result['dual_residual'][-1],
                  final_primal_tolerance=result['primal_tolerance'][-1],
                  final_dual_tolerance=result['dual_tolerance'][-1],
                  working_shape=list(result['working_shape']), variable_count=result['variable_count'],
                  constraint_count=result['constraint_count'], numpy_version=np.__version__,
                  files_sha256={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(args.output.rglob('*')) if p.is_file()})
    (args.output/'measurements.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
