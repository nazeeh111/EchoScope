"""Optional confocal saves on tiny synthetic arrays; no acquisition or PSF."""
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

try:
    import numpy as np
except ImportError:
    np = None


def confocal_helper():
    source = Path(__file__).resolve().parents[1] / 'AcousticNLOSReconstruction.py'
    module = ast.parse(source.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef)
               and node.name == 'AcousticNLOSReconstruction')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'confocalReconstruction')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['confocalReconstruction']


@unittest.skipIf(np is None, 'Confocal-save tests require NumPy; dispatcher tests remain standard-library only')
class ConfocalSaveTests(unittest.TestCase):
    def fixture(self, directory, plotting=True, use_lct=False, background=False):
        path = Path(directory) / 'synthetic.npy'
        raw = np.arange(60, dtype=float).reshape((2, 2, 3, 5))
        np.save(path, raw)
        background_path = None
        if background:
            background_path = Path(directory) / 'background.npy'
            np.save(background_path, np.full(raw.shape, .5))
        worker = SimpleNamespace(channels=[2, 2], samples_per_chirp=5,
                                 fstart_idx=1, fend_idx=4, f=np.arange(5, dtype=float),
                                 f0=0., f1=4., B=4., fstart=1., fend=4., T=1., v=1.,
                                 enable_plotting=plotting, use_lct=use_lct)
        worker.plot_vol = Mock()
        worker.run_lct = Mock(side_effect=lambda meas, *args: meas + 100)
        # Independently specified diagonal/window/orientation contract for this fixture.
        meas = np.array([[[46., 51., 56.], [1., 6., 11.]],
                         [[47., 52., 57.], [2., 7., 12.]],
                         [[48., 53., 58.], [3., 8., 13.]]])
        if background:
            meas -= .5
        return path, background_path, worker, raw, meas

    def test_optional_save_preserves_arrays_metadata_and_filename(self):
        helper = confocal_helper()
        keys = {'out', 'meas', 'fstart_idx', 'fend_idx', 'f', 'f0', 'f1', 'B',
                'fstart', 'fend', 'T', 'v', 'processed_data'}
        for background in [False, True]:
            for use_lct in [False, True]:
                with self.subTest(background=background, mocked_lct=use_lct):
                    with tempfile.TemporaryDirectory() as directory:
                        path, bg, worker, raw, meas = self.fixture(directory, use_lct=use_lct, background=background)
                        original_bytes = path.read_bytes()
                        self.assertIsNone(helper(worker, str(path), str(bg) if bg else None, save_npy=True))
                        saved = np.load(path.with_name('synthetic_out.npy'), allow_pickle=True).item()
                        self.assertEqual(set(saved), keys)
                        np.testing.assert_array_equal(saved['processed_data'], raw - (.5 if background else 0))
                        np.testing.assert_array_equal(saved['meas'], meas)
                        expected_out = meas + (100 if use_lct else 0)
                        np.testing.assert_array_equal(saved['out'], expected_out)
                        np.testing.assert_array_equal(saved['f'], worker.f)
                        for name in keys - {'out', 'meas', 'processed_data', 'f'}:
                            self.assertEqual(saved[name], getattr(worker, name))
                        self.assertEqual(path.read_bytes(), original_bytes)
                        worker.plot_vol.assert_called_once()
                        np.testing.assert_array_equal(worker.plot_vol.call_args.args[0], expected_out)
                        self.assertEqual(worker.plot_vol.call_args.args[1], 'Confocal reconstruction')
                        self.assertEqual(worker.plot_vol.call_args.kwargs, {'savefile': str(path.with_suffix('.png'))})
                        if use_lct:
                            worker.run_lct.assert_called_once()
                            np.testing.assert_array_equal(worker.run_lct.call_args.args[0], meas)
                            self.assertEqual(worker.run_lct.call_args.args[1:], (3, 2, 3, 2, 1, 1))
                        else:
                            worker.run_lct.assert_not_called()

    def test_plotting_off_keeps_numpy_save_disabled(self):
        helper = confocal_helper()
        for save in [False, True]:
            with self.subTest(save_npy=save):
                with tempfile.TemporaryDirectory() as directory:
                    path, _, worker, _, _ = self.fixture(directory, plotting=False)
                    output = path.with_name('synthetic_out.npy')
                    output.write_bytes(b'prior synthetic result')
                    self.assertIsNone(helper(worker, str(path), save_npy=save))
                    self.assertEqual(output.read_bytes(), b'prior synthetic result')
                    worker.plot_vol.assert_not_called()
                    worker.run_lct.assert_not_called()

    def test_default_save_false_keeps_existing_plot_only_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, worker, _, meas = self.fixture(directory)
            self.assertIsNone(confocal_helper()(worker, str(path)))
            worker.plot_vol.assert_called_once()
            np.testing.assert_array_equal(worker.plot_vol.call_args.args[0], meas)
            self.assertFalse(path.with_name('synthetic_out.npy').exists())
            worker.run_lct.assert_not_called()


if __name__ == '__main__':
    unittest.main()
