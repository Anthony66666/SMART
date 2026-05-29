import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class PretrainCheckpointLoadingTest(unittest.TestCase):
    def _load_params_calls(self, script_name):
        tree = ast.parse((REPO_ROOT / script_name).read_text())
        return [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'load_params_from_file'
        ]

    def test_pretrain_loads_checkpoint_on_cpu_before_ddp_placement(self):
        for script_name in ('train.py', 'val.py'):
            with self.subTest(script_name=script_name):
                calls = self._load_params_calls(script_name)
                self.assertEqual(len(calls), 1)
                to_cpu_keywords = [
                    keyword for keyword in calls[0].keywords
                    if keyword.arg == 'to_cpu'
                ]
                self.assertEqual(len(to_cpu_keywords), 1)
                self.assertIsInstance(to_cpu_keywords[0].value, ast.Constant)
                self.assertIs(to_cpu_keywords[0].value.value, True)
