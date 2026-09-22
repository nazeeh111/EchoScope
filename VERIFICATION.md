# Local verification

Checked 2026-09-22. These are bounded local checks, not research replication.

- 25 numerical source, calibration, and related files are byte-identical to the retained pre-publication checkout.
- All 16 microphone calibration channels loaded; calibration array shape is 3000 × 16 and finite.
- Default chirp generated with finite samples.
- An 8 × 4 × 4 single-impulse volume through `run_lct` produced finite, nonzero output with exactly equal arrays against the retained baseline.

Not run: initial or iterative reconstruction from real acoustic captures, since the external dataset is absent. Legacy dependency pins were retained. The documented pre-existing `all` dispatch and unused `lct` helper issues remain; no scientific algorithm was silently changed.

## Test environment

An isolated local CPU environment used NumPy 2.5.3, PyTorch 2.14.0, h5py 3.16.0, SciPy 1.18.1, and Matplotlib 3.11.2. It did not replace the archived dependency manifests or modify shared environments.

[Machine-readable check results](verification.json) record dimensions and available checks.
