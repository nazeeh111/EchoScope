"""Actual iterative engine and PSF operators on original generated arrays."""
import ast
import contextlib
import io
from pathlib import Path
import tempfile
import sys
from types import MethodType, SimpleNamespace
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(np is None, 'Iterative numerical tests require NumPy')
class ADMMCoreTests(unittest.TestCase):
    def fixture(self, root, level=20., shape=(8, 4, 4), scene='letter_H'):
        data = root / 'data'
        data.mkdir(parents=True)
        np.save(data / (scene + '_out.npy'), dict(meas=np.full(shape, level), T=1.,
                 v=340., fend=3800., B=18000., fstart=1000.))
        np.save(data / 'psf_gaussian.npy', np.ones((1, 1)))
        return data

    def execute(self, *args, **kwargs):
        from ADMMReconstruction import run
        with contextlib.redirect_stdout(io.StringIO()):
            return run(*args, plotting=False, **kwargs)

    def test_psf_operator_matches_direct_circular_convolution(self):
        from util.pypher import psf2otf
        image = np.arange(4*5*6, dtype=float).reshape(4, 5, 6) / 17
        kernel = np.arange(27, dtype=float).reshape(3, 3, 3) / 351
        expected = np.zeros_like(image)
        # Direct spatial convolution, independent of FFT/padding helpers.
        for index in np.ndindex(image.shape):
            for k in np.ndindex(kernel.shape):
                source = tuple((i - (j - 1)) % n for i, j, n in zip(index, k, image.shape))
                expected[index] += kernel[k] * image[source]
        actual = np.fft.ifftn(np.fft.fftn(image) * psf2otf(kernel, image.shape)).real
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
        zero = psf2otf(np.zeros((1, 1, 1)), image.shape)
        self.assertEqual(zero.shape, image.shape)
        np.testing.assert_array_equal(zero, 0)

    def test_shrinkage_zero_and_nonzero_vectors(self):
        from ADMMReconstruction import shrinkage
        with np.errstate(all='raise'):
            actual = shrinkage(np.array([0., 3., 6.]), np.array([0., 4., 8.]), np.zeros(3), 2.)
        np.testing.assert_allclose(actual[0], [0, 1.8, 4.8], rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(actual[1], [0, 2.4, 6.4], rtol=1e-15, atol=1e-15)
        np.testing.assert_array_equal(actual[2], 0)

    def test_real_headless_iterations_and_changed_input(self):
        hashes = []
        with tempfile.TemporaryDirectory() as temp:
            for level in (0., 20., 30.):
                root = Path(temp) / str(level)
                data = self.fixture(root, level)
                output = root / 'result'
                result = self.execute('letter_H', data_dir=data, output_dir=output, max_iters=1000)
                self.assertTrue(result['converged'])
                self.assertEqual(len(result['dual_residual']), result['iter'] + 1)
                self.assertEqual(len(result['timing']), result['iter'] + 1)
                self.assertEqual(len(result['primal_tolerance']), result['iter'] + 1)
                self.assertEqual(len(result['dual_tolerance']), result['iter'] + 1)
                self.assertEqual(result['variable_count'], 8*4*4)
                self.assertEqual(result['constraint_count'], 4*8*4*4)
                self.assertLess(result['primal_residual'][-1], result['primal_tolerance'][-1])
                self.assertLess(result['dual_residual'][-1], result['dual_tolerance'][-1])
                # Identity PSF, constant nonnegative data and zero spatial TV:
                # independent scalar soft threshold for .5||x-b||² + .9||x||1.
                # Tolerance follows the solver's 1e-3 absolute/relative stop gates.
                np.testing.assert_allclose(result['out'], max(level-.9, 0), rtol=1e-3, atol=1e-3)
                if level == 0:
                    self.assertEqual(result['iter'], 0)
                    self.assertEqual(result['dual_residual'], [0.])
                saved = np.load(output/'letter_H_admm_out.npy', allow_pickle=True).item()
                np.testing.assert_array_equal(saved['out'], result['out'])
                hashes.append(saved['out'].tobytes())
        self.assertEqual(len(set(hashes)), 3)
        # Norm-based stop gates are not a pointwise solution error guarantee.
        # Request tighter convergence for threshold-adjacent scalar oracles.
        with tempfile.TemporaryDirectory() as temp:
            for level in (.5, .9, .95, 1.):
                root = Path(temp)/str(level)
                data = self.fixture(root, level)
                result = self.execute('letter_H', data_dir=data, output_dir=root/'result',
                    max_iters=1000, absolute_tolerance=1e-6, relative_tolerance=1e-6)
                np.testing.assert_allclose(result['out'], max(level-.9, 0), rtol=1e-6, atol=1e-4)

    def test_independent_l1_dual_and_iteration_exhaustion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = self.fixture(root)
            output = root/'result'
            with self.assertRaisesRegex(RuntimeError, 'did not converge after 1 iterations') as caught:
                self.execute('letter_H', data_dir=data, output_dir=output, max_iters=1)
            actual = float(str(caught.exception).split('dual=')[1])
            # Initial z1=0, x=20; lambda/rho=.9/.1=9, hence new z1=11.
            expected = .1 * np.linalg.norm(np.full((8, 4, 4), 11.))
            self.assertAlmostEqual(actual, expected, places=12)
            self.assertFalse(output.exists(), 'An unconverged trial must not be saved as a converged result')

    def test_odd_grid_iterative_lct_zero_input_and_invalid_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = self.fixture(root, 0., (8, 3, 5), 'corner_reflectors')
            result = self.execute('corner_reflectors', data_dir=data, output_dir=root/'result', max_iters=1)
            self.assertEqual(result['iter'], 0)
            self.assertEqual(result['out'].shape, (8, 3, 5))
            self.assertEqual(result['working_shape'], (32, 6, 10))
            self.assertEqual(result['variable_count'], 32*6*10)
            self.assertEqual(result['constraint_count'], 4*32*6*10)
            np.testing.assert_array_equal(result['out'], 0)
            for limit in (0, True, 1.5):
                with self.assertRaises(ValueError):
                    self.execute('corner_reflectors', data_dir=data, max_iters=limit)
            for name, value in [('absolute_tolerance', 0), ('relative_tolerance', float('nan'))]:
                with self.assertRaises(ValueError):
                    self.execute('corner_reflectors', data_dir=data, **{name: value})

    def test_scaled_dual_and_working_grid_against_explicit_operator(self):
        from ADMMReconstruction import run
        source = Path(run.__code__.co_filename)
        observe_line = next(i for i, line in enumerate(source.read_text().splitlines(), 1)
                            if "print('Iter:" in line)
        absolute, relative, rho = 2e-4, 3e-4, .1
        class Observed(Exception):
            pass
        with tempfile.TemporaryDirectory() as temp:
            for scene, level in [('letter_H', 20.), ('corner_reflectors', 0.),
                                  ('corner_reflectors', .7)]:
                with self.subTest(scene=scene, level=level):
                    root = Path(temp)/(scene+str(level))
                    data = self.fixture(root, level, (3, 3, 3), scene)
                    observed = {}
                    def trace(frame, event, argument):
                        if (frame.f_code is run.__code__ and event == 'line'
                                and frame.f_lineno == observe_line):
                            state = frame.f_locals
                            shape = state['b'].shape
                            n = state['b'].size
                            identity = np.eye(n)
                            blocks = [identity]
                            # Independent dense K=[I;D0;D1;D2], using spatial
                            # differences on basis vectors, no FFT/PSF helper.
                            for axis in range(3):
                                blocks.append(np.column_stack([
                                    (basis.reshape(shape)-np.roll(basis.reshape(shape), 1, axis=axis)).ravel()
                                    for basis in identity.T]))
                            K = np.vstack(blocks)
                            kx = K @ state['x'].ravel()
                            z = np.concatenate([state['z'+str(i)].ravel() for i in range(1, 5)])
                            old = np.concatenate([state['z'+str(i)+'_prev'].ravel() for i in range(1, 5)])
                            u = np.concatenate([state['u'+str(i)].ravel() for i in range(1, 5)])
                            expected = [np.linalg.norm(kx-z), np.linalg.norm(rho*K.T@(z-old)),
                                np.sqrt(K.shape[0])*absolute+relative*max(np.linalg.norm(kx), np.linalg.norm(z)),
                                np.sqrt(K.shape[1])*absolute+relative*np.linalg.norm(rho*K.T@u)]
                            actual = [state['resid_pri'], state['resid_dual'], state['eps_pri'], state['eps_dual']]
                            np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-13)
                            observed['working_shape'] = shape
                            raise Observed()
                        return trace
                    previous_trace = sys.gettrace()
                    try:
                        sys.settrace(trace)
                        with self.assertRaises(Observed):
                            self.execute(scene, data_dir=data, output_dir=root/'result', max_iters=1,
                                         absolute_tolerance=absolute, relative_tolerance=relative)
                    finally:
                        sys.settrace(previous_trace)
                    expected_shape = (3, 3, 3) if scene == 'letter_H' else (12, 6, 6)
                    self.assertEqual(observed['working_shape'], expected_shape)

    def test_existing_result_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = self.fixture(root, 0.)
            output = root/'result'
            original = self.execute('letter_H', data_dir=data, output_dir=output, max_iters=1)
            target = output/'letter_H_admm_out.npy'
            original_bytes = target.read_bytes()
            np.save(data/'letter_H_out.npy', dict(meas=np.full((8,4,4), 30.), T=1.,
                    v=340., fend=3800., B=18000., fstart=1000.))
            with self.assertRaisesRegex(FileExistsError, 'Refusing to replace'):
                self.execute('letter_H', data_dir=data, output_dir=output, max_iters=1000)
            self.assertEqual(target.read_bytes(), original_bytes)
            np.testing.assert_array_equal(np.load(target, allow_pickle=True).item()['out'], original['out'])
            self.assertEqual(list(output.iterdir()), [target], 'No owned staging directory should remain')

            # Create an actual competing output after admission but before
            # publication. The exclusive final link must still preserve it.
            from ADMMReconstruction import run
            source = Path(run.__code__.co_filename)
            publish_line = next(i for i, line in enumerate(source.read_text().splitlines(), 1)
                                if 'with tempfile.TemporaryDirectory' in line)
            np.save(data/'letter_H_out.npy', dict(meas=np.zeros((8,4,4)), T=1.,
                    v=340., fend=3800., B=18000., fstart=1000.))
            raced_output = root/'raced'
            raced_output.mkdir()
            raced = raced_output/'letter_H_admm_out.npy'
            def create_competing_output(frame, event, argument):
                if frame.f_code is run.__code__ and event == 'line' and frame.f_lineno == publish_line:
                    raced.write_bytes(b'concurrent result')
                return create_competing_output
            previous_trace = sys.gettrace()
            try:
                sys.settrace(create_competing_output)
                with self.assertRaisesRegex(FileExistsError, 'Refusing to replace'):
                    self.execute('letter_H', data_dir=data, output_dir=raced_output, max_iters=1)
            finally:
                sys.settrace(previous_trace)
            self.assertEqual(raced.read_bytes(), b'concurrent result')
            self.assertEqual(list(raced_output.iterdir()), [raced])

    def test_explicit_headless_save_uses_actual_lct(self):
        from util import lct
        source = ROOT / 'AcousticNLOSReconstruction.py'
        cls = next(x for x in ast.parse(source.read_text()).body if isinstance(x, ast.ClassDef))
        methods = [x for x in cls.body if isinstance(x, ast.FunctionDef)
                   and x.name in ('confocalReconstruction', 'run_lct')]
        namespace = {'np': np, 'lct': lct}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'scene.npy'
            raw = np.zeros((2, 2, 3, 8))
            raw[0, 0] = np.arange(24).reshape(3, 8)
            raw[1, 1] = 100 + np.arange(24).reshape(3, 8)
            np.save(path, raw)
            worker = SimpleNamespace(channels=[2,2], samples_per_chirp=8, fstart_idx=0, fend_idx=8,
                use_lct=True, enable_plotting=False, f=np.arange(8), f0=2000., f1=20000.,
                B=18000., fstart=1000., fend=3800., T=1., v=340., dx=.9, dy=.95, snr=8000.)
            worker.run_lct = MethodType(namespace['run_lct'], worker)
            namespace['confocalReconstruction'](worker, str(path), save_npy=True)
            saved = np.load(path.with_name('scene_out.npy'), allow_pickle=True).item()
            expected = np.array([[[100+r, 108+r, 116+r], [r, 8+r, 16+r]] for r in range(8)])
            np.testing.assert_array_equal(saved['meas'], expected)
            np.testing.assert_array_equal(saved['processed_data'], raw)
            self.assertEqual(saved['out'].shape, expected.shape)
            self.assertTrue(np.isfinite(saved['out']).all())
            self.assertFalse(np.array_equal(saved['out'], saved['meas']))


if __name__ == '__main__':
    unittest.main()
