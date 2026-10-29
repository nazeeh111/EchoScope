"""Real production LCT versus direct DFT; generated arrays, no captured scenes."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]


def reference(meas, S, sx, sy, start, stop, snr):
    """Scalar PSF/interpolation construction and dense DFT, without np.fft."""
    Z, Y, X = meas.shape
    psf = np.zeros((2*S*Z, 2*Y, 2*X), dtype=np.float32)
    zs = np.linspace(0, 2, 2*S*Z)
    for y, yy in enumerate(np.linspace(-1, 1, 2*Y)):
        for x, xx in enumerate(np.linspace(-1, 1, 2*X)):
            distances = [abs(sx*sx*xx*xx + sy*sy*yy*yy-z) for z in zs]
            for z, distance in enumerate(distances):
                if distance == min(distances):
                    psf[z, (y+Y) % (2*Y), (x+X) % (2*X)] = 1
    psf /= psf.sum()
    mtx = np.zeros((S*Z, Z))
    old = np.linspace(start, stop, Z)**2
    for row, value in enumerate(np.linspace(start*start, stop*stop, S*Z)):
        if value <= old[0]:
            mtx[row, 0] = 1
        elif value >= old[-1]:
            mtx[row, -1] = 1
        else:
            col = next(i for i in range(1, Z) if old[i] >= value)
            weight = (value-old[col-1])/(old[col]-old[col-1])
            mtx[row, col-1], mtx[row, col] = 1-weight, weight
    mtx /= mtx.sum(axis=0)
    inverse = mtx.T.copy()
    inverse /= inverse.sum(axis=0)

    def dft(data, inverse=False):
        for axis, n in enumerate(data.shape):
            matrix = np.exp((2j if inverse else -2j)*np.pi*np.outer(np.arange(n), np.arange(n))/n)
            if inverse:
                matrix /= n
            data = np.moveaxis(np.tensordot(matrix, data, axes=(1, axis)), 0, axis)
        return data

    stretched = (mtx @ meas.reshape(Z, -1)).reshape(S*Z, Y, X)
    padded = np.zeros(psf.shape)
    a, b, c = (n//2 for n in stretched.shape)
    padded[a:a+S*Z, b:b+Y, c:c+X] = stretched
    kernel = dft(psf)
    deconvolved = dft(dft(padded)*kernel.conj()/(abs(kernel)**2+1/snr), inverse=True)
    a, b, c = S*(Z//2)+1, Y//2+1, X//2+1
    cropped = abs(deconvolved[a:a+S*Z, b:b+Y, c:c+X])
    return (inverse @ cropped.reshape(S*Z, -1)).reshape(Z, Y, X)


@unittest.skipIf(np is None, 'Numerical core tests require NumPy')
class LCTCoreTests(unittest.TestCase):
    def test_grids_match_independent_direct_dft(self):
        from util.lct import reconstruct
        for shape, S in [((4, 2, 4), 2), ((5, 3, 5), 2), ((3, 1, 1), 1), ((4, 3, 2), 3)]:
            with self.subTest(shape=shape, S=S):
                meas = np.arange(np.prod(shape), dtype=float).reshape(shape)/np.prod(shape)
                actual = reconstruct(meas, S, .3, .5, .2, 1.1, 8000)
                expected = reference(meas, S, .3, .5, .2, 1.1, 8000)
                # Modern NumPy retains the float32 PSF's FFT precision. The
                # direct DFT is float64. Budget float32 transform rounding for
                # eight operations per FFT stage, amplified by the maximum
                # Wiener-filter gain sqrt(snr)/2. This bound is parameter-based,
                # not fitted to the observed discrepancy or expected arrays.
                stages = np.ceil(np.log2(np.prod((2*S*shape[0], 2*shape[1], 2*shape[2]))))
                tolerance = np.finfo(np.float32).eps * 8 * stages * np.sqrt(8000)/2
                scale = max(1., float(np.max(abs(expected))))
                np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance*scale)
                # The numerical budget must still detect a one-cell crop error.
                if shape[2] > 1:
                    self.assertFalse(np.allclose(np.roll(actual, 1, axis=2), expected,
                                                 rtol=tolerance, atol=tolerance*scale))
                self.assertEqual(actual.shape, shape)

    def test_existing_method_calls_actual_core(self):
        from util import lct
        # Load the real method without importing unrelated plotting/capture deps.
        source = ROOT/'AcousticNLOSReconstruction.py'
        cls = next(x for x in ast.parse(source.read_text()).body if isinstance(x, ast.ClassDef))
        method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'run_lct')
        namespace = {'np': np, 'lct': lct}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        worker = SimpleNamespace(T=1., v=340., channels=[16, 16], dx=.9, dy=.95,
                                 B=18000., fstart=1000., fend=3800., snr=8000.)
        meas = np.zeros((8, 3, 5)); meas[4, 1, 2] = 1
        actual = namespace['run_lct'](worker, meas, 5, 3, 8, 2, 1., 1.)
        stop = 3800/18000*10
        correction = 1+(1000/3800)**2
        expected = lct.reconstruct(meas, 2, .9/stop*correction, .95/stop*correction,
                                   1000/18000*10, stop, 8000.)
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaisesRegex(ValueError, 'shape'):
            namespace['run_lct'](worker, meas, 3, 5, 8, 2, 1., 1.)

    def test_invalid_inputs_fail_before_fft(self):
        from util.lct import reconstruct
        kwargs = dict(S=2, slope_x=.3, slope_y=.5, start=.2, stop=1.1, snr=8000)
        for meas in [np.ones((2, 3)), np.ones((0, 3, 2)), np.full((2, 3, 2), np.nan)]:
            with self.subTest(shape=meas.shape), self.assertRaises(ValueError):
                reconstruct(meas, **kwargs)
        for name, value in [('S', 0), ('S', 1.5), ('S', True), ('snr', 0),
                            ('slope_x', np.inf), ('start', -1), ('stop', .1)]:
            with self.subTest(parameter=name, value=value), self.assertRaises(ValueError):
                reconstruct(np.ones((2, 3, 2)), **dict(kwargs, **{name: value}))
        np.testing.assert_array_equal(reconstruct(np.zeros((3, 3, 5)), **kwargs), 0)

    def test_changed_input_example_and_existing_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/'a', Path(directory)/'b'
            command = [sys.executable, str(ROOT/'examples/lct_impulse.py')]
            for output, target in [(a, [4, 1, 2]), (b, [5, 0, 3])]:
                result = subprocess.run(command+['--output', str(output), '--target', *map(str, target)],
                                        check=True, capture_output=True, text=True)
                report = json.loads(result.stdout)
                self.assertEqual(report['input_target'], target)
                self.assertEqual(report['output_shape'], [8, 3, 5])
                self.assertIn('synthetic', report['evidence'])
            self.assertFalse(np.array_equal(np.load(a/'volume.npy'), np.load(b/'volume.npy')))
            original = (a/'volume.npy').read_bytes()
            failed = subprocess.run(command+['--output', str(a)], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual((a/'volume.npy').read_bytes(), original)
