# Local verification

The sections below record successive checkpoints. Earlier statements about
plotting-dependent saves and unexecuted iterative code describe earlier source;
the final iterative/headless section supersedes them for current source.

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


## Production numerical core (2026-10-29)

The supported `run_lct` failed for odd spatial grids: an 8×3×5 input padded to 32×5×9, while its PSF was 32×6×10. The numerical block now lives in `util.lct.reconstruct`, shared by the existing class method and a NumPy-only example. Asymmetric right padding gives exactly doubled dimensions; explicit crop lengths also fix silently empty outputs on spatial axes of length two. The established one-bin offset and working even-grid behavior are retained. Four random-input grids with spatial axes at least four produced bit-identical arrays against the original method on the same runtime (maximum difference zero). The unrelated legacy standalone `lct` helper remains unsupported and unchanged.

Four numerical test methods exercise actual production reconstruction. A reference builds the PSF and interpolation weights independently and evaluates dense direct DFT matrices instead of FFTs, covering rectangular, odd/even and single-cell spatial grids. Further checks cover the existing class adapter with the real core, invalid inputs, zero input, changed numerical impulses and preservation of existing output bytes. This adapter test loads the method without running the capture/plotting constructor; it does not validate those dependencies or acquire data.

The example computes an 8×3×5 volume twice with different impulse indices and saves arrays, readable maximum projections and digest-bearing metadata. It does not model a complete acoustic forward scene or establish reconstruction quality on measured captures. Calibration tables, raw-data licensing/access, demodulation, NMO/DMO, PSF fitting and iterative ADMM reproduction remain separate unresolved gates. No external capture was downloaded or represented as a new measurement.

The numerical CI job installs only the dedicated NumPy binary pin, requires all four methods to pass without skips/expected failures, and preserves the existing dispatcher and confocal-save jobs.

The first direct-DFT comparison requested 1e-10 agreement and failed by up to 1.18e-6 on NumPy 2.3.5. Instrumentation found the retained float32 PSF yields a complex64 FFT kernel and float32 regularization denominator there, while the dense reference uses complex128. Production precision and kernels were preserved. The mathematical comparison now budgets float32 machine epsilon times eight operations per FFT stage and the inverse filter's maximum gain, `sqrt(snr)/2`; it also requires a deliberately shifted crop to fail that comparison. This is a conservative rounding budget for these bounded grids, not a general FFT error theorem. The independent high-precision reference arrays were not adjusted to match observations. Historical NumPy [FFT type promotion](https://numpy.org/doc/2.2/reference/routines.fft.html#type-promotion) differs, so bit identity is asserted only within the same runtime, not across NumPy versions.


## Iterative/headless follow-up, 2026-10-29

The previous-iterate bug was reproduced from exact `fce1a03` production code:
a constant 20 input yielded reported L1 dual residual 0 even though the independent
previous-zero calculation was 12.44507935. The iterate is now captured before
shrinkage. A zero-volume headless run previously entered another iteration after
writing a converged result; convergence now exits independently of plotting.
Optional data/output directories, headless execution and an iteration cap let
callers execute the actual engine without graphics. Exhaustion is an explicit
failure and is not written as a converged reconstruction. Residual/timing histories
now include the stopping iteration. Default scene settings and plotting remain.

A real PSF-to-OTF call failed on removed `np.alltrue` under NumPy 2; `np.all` now
preserves that predicate and unused SciPy imports no longer gate NumPy-only
operators. The zero-PSF fast path returns the requested output shape. Independent
spatial circular convolution checks the PSF centering/FFT operator. Zero-gradient
vector shrinkage avoids invalid divisions; the iterative LCT branch uses the same
explicit odd-grid padding/crop lengths as the direct core.

The six new numerical tests execute actual ADMM, including zero-input immediate
convergence, changed constant inputs with an independent scalar identity-PSF
optimum, the independently calculated first L1 dual residual, iteration exhaustion
without a false saved success, odd-grid LCT refinement, and actual headless
confocal saving with the real LCT engine. The old test encoding plotting-dependent
save gating was replaced with the explicit-save contract; save_npy=False still
preserves prior output. These are software/numerical checks, not measured-data
quality. Baseline failures and exact source version remain in task-local evidence.


Norm-based stopping thresholds do not themselves guarantee pointwise solution
error. A retained threshold-adjacent probe demonstrated that distinction. The
identity example explicitly requests tighter 1e-6 convergence tolerances and checks
its scalar optimum separately at 1e-4 absolute plus 1e-6 relative error. The library's
original 1e-3 defaults are unchanged. Tests include inputs below, at and above the
L1 threshold, not just the original 20/30 examples.
