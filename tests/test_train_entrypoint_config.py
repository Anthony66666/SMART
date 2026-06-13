import unittest
from types import SimpleNamespace

import train


class TrainEntrypointConfigTest(unittest.TestCase):
    def test_resolves_configured_max_steps(self):
        cfg = SimpleNamespace(max_steps=1000)

        self.assertEqual(train.resolve_max_steps(cfg), 1000)

    def test_missing_max_steps_uses_lightning_default(self):
        cfg = SimpleNamespace()

        self.assertEqual(train.resolve_max_steps(cfg), -1)


if __name__ == "__main__":
    unittest.main()
