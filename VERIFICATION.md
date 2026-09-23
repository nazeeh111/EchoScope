# Local verification

Numerical checks executed 2026-09-22; command-line dispatch checked 2026-09-23. These are bounded local checks, not research replication.

- The earlier publication compared 25 numerical source, calibration, and related files byte-for-byte. After the command-line fix below, 24 remain byte-identical; `AcousticNLOSReconstruction.py` differs only in the corrected dispatch assignment. Its entire reconstruction class and numerical methods remain unchanged.
- All 16 microphone calibration channels loaded; calibration array shape is 3000 × 16 and finite.
- Default chirp generated with finite samples.
- An 8 × 4 × 4 single-impulse volume through `run_lct` produced finite, nonzero output with exactly equal arrays against the retained baseline.

Not run: initial or iterative reconstruction from real acoustic captures, since the external dataset is absent. Legacy dependency pins were retained. The unused standalone `util.lct.lct` helper still has an inherited argument mismatch and ambiguous axis/shape handling; the supported reconstruction uses `run_lct` instead. Its intended math cannot be inferred safely from the missing argument alone, so it was not changed.

## Command-line regression

The `all` branch assigned `scenes` but the loop consumed `scene`, so it showed usage and processed no scenes. Correcting the assignment dispatches all nine supported scenes. Four regression tests execute the actual CLI block with its reconstruction worker mocked: all scenes, explicit scene order, invalid input, and no input. The all-scenes test failed before the fix; all four pass after it. No measurements are loaded by these tests. Run `python -m unittest discover -s tests -v` from the repository root.

## Test environment

An isolated local CPU environment used NumPy 2.5.3, PyTorch 2.14.0, h5py 3.16.0, SciPy 1.18.1, and Matplotlib 3.11.2. It did not replace the archived dependency manifests or modify shared environments.

[Machine-readable check results](verification.json) record dimensions and available checks.
