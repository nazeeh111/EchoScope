# Local verification

Numerical checks executed 2026-09-22; command-line dispatch checked 2026-09-23. These are bounded local checks, not research replication.

- At that earlier checkpoint, the publication compared 25 numerical source, calibration, and related files byte-for-byte. After the command-line fix below, 24 remained byte-identical; `AcousticNLOSReconstruction.py` differed only in the corrected dispatch assignment. Its entire reconstruction class and numerical methods were unchanged at that checkpoint.
- All 16 microphone calibration channels loaded; calibration array shape is 3000 × 16 and finite.
- Default chirp generated with finite samples.
- An 8 × 4 × 4 single-impulse volume through `run_lct` produced finite, nonzero output with exactly equal arrays against the retained baseline.

Not run: initial or iterative reconstruction from real acoustic captures, since the external dataset is absent. Legacy dependency pins were retained. The unused standalone `util.lct.lct` helper still has an inherited argument mismatch and ambiguous axis/shape handling; the supported reconstruction uses `run_lct` instead. Its intended math cannot be inferred safely from the missing argument alone, so it was not changed.

## Command-line regression

The `all` branch assigned `scenes` but the loop consumed `scene`, so it showed usage and processed no scenes. Correcting the assignment dispatches all nine supported scenes. Four regression tests execute the actual CLI block with its reconstruction worker mocked: all scenes, explicit scene order, invalid input, and no input. The all-scenes test failed before the fix; all four pass after it. No measurements are loaded by these tests. Run `python -m unittest discover -s tests -v` from the repository root.

## Test environment

An isolated local CPU environment used NumPy 2.5.3, PyTorch 2.14.0, h5py 3.16.0, SciPy 1.18.1, and Matplotlib 3.11.2. It did not replace the archived dependency manifests or modify shared environments.

[Machine-readable check results](verification.json) record dimensions and available checks.

## Continuous command checks

The scene-dispatch workflow runs the four existing standard-library regression tests on Python 3.12. It executes the actual command-line block with the capture-processing worker replaced, covering the all-scenes order, explicitly selected scenes, invalid input and empty input. It protects the repaired dispatcher without installing the legacy numerical stack or loading external measurements; it does not establish acoustic reconstruction accuracy.

## Subsequent optional confocal-save correction

The optional `confocalReconstruction(..., save_npy=True)` save referenced an undefined variable. It now saves the method's existing loaded, optionally background-subtracted `processed_data` under the same dictionary key. Filenames, metadata keys, the default `save_npy=False`, and the plotting-dependent save gate are preserved; no reconstruction arithmetic or ADMM behavior is changed. Three focused helper tests use tiny synthetic NumPy arrays and replace plotting and LCT. They check save payloads, metadata and existing no-save paths; they explicitly skip when NumPy is unavailable. No capture, calibration, PSF, model or full reconstruction pipeline is exercised by these tests.

## Confocal-save CI gate

The existing dispatcher job, workflow triggers and read-only permissions are retained. A separate job on Ubuntu 24.04 x64 uses Python 3.12 and installs only the NumPy 2.5.4 binary wheel, without dependencies. It discovers only `test_confocal_save.py` and rejects a method count other than three, any skip or expected failure, and any unsuccessful result. This NumPy pin is test-only; archived project dependency manifests are unchanged. It does not install the legacy stack or run acquisition, calibration, PSF, plotting, native LCT or ADMM. The optional no-NumPy skip remains available for local capture-free checks; the dedicated CI job cannot pass by skipping the save regressions.
