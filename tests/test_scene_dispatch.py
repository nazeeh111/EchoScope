"""Exercise the actual CLI dispatch block without loading captures or devices."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class SceneDispatchTests(unittest.TestCase):
    def dispatch(self, *arguments):
        script = Path(__file__).resolve().parents[1] / "AcousticNLOSReconstruction.py"
        module = ast.parse(script.read_text())
        # Execute the real __main__ block with only the expensive worker replaced.
        main = module.body[-1]
        self.assertIsInstance(main, ast.If)
        worker = Mock()
        namespace = {
            "__name__": "__main__",
            "sys": SimpleNamespace(argv=[str(script), *arguments]),
            "AcousticNLOSReconstruction": Mock(return_value=worker),
        }
        exec(compile(ast.Module(body=[main], type_ignores=[]), str(script), "exec"), namespace)
        return worker

    def test_all_runs_each_supported_scene_in_order(self):
        worker = self.dispatch("all")
        self.assertEqual(
            [call.args[0] for call in worker.run.call_args_list],
            ["double", "letter_H", "corner_reflectors", "psf",
             "resolution_corner1m", "resolution_corner2m",
             "resolution_plane1m", "resolution_plane2m", "letters_LT"],
        )
        worker.usage.assert_not_called()

    def test_explicit_scenes_keep_requested_order(self):
        worker = self.dispatch("letter_H", "psf")
        self.assertEqual([call.args[0] for call in worker.run.call_args_list], ["letter_H", "psf"])
        worker.usage.assert_not_called()

    def test_invalid_scene_shows_usage_without_processing_it(self):
        worker = self.dispatch("unknown")
        worker.run.assert_not_called()
        worker.usage.assert_called_once_with()

    def test_no_scenes_shows_usage_without_processing(self):
        worker = self.dispatch()
        worker.run.assert_not_called()
        worker.usage.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
