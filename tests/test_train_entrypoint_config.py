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

    def test_missing_val_check_interval_uses_lightning_default(self):
        cfg = SimpleNamespace()

        self.assertEqual(train.resolve_val_check_interval(cfg), 1.0)

    def test_resolves_configured_val_check_interval(self):
        cfg = SimpleNamespace(val_check_interval=1000)

        self.assertEqual(train.resolve_val_check_interval(cfg), 1000)

    def test_step_checkpoint_disables_epoch_checkpoint_trigger(self):
        cfg = SimpleNamespace(checkpoint_every_n_train_steps=1000)

        self.assertEqual(train.resolve_checkpoint_every_n_train_steps(cfg), 1000)
        self.assertIsNone(train.resolve_checkpoint_every_n_epochs(cfg))

    def test_epoch_checkpoint_trigger_is_default(self):
        cfg = SimpleNamespace()

        self.assertIsNone(train.resolve_checkpoint_every_n_train_steps(cfg))
        self.assertEqual(train.resolve_checkpoint_every_n_epochs(cfg), 1)

    def test_last_checkpoint_is_saved_by_default(self):
        cfg = SimpleNamespace()

        self.assertTrue(train.resolve_save_last_checkpoint(cfg))


if __name__ == "__main__":
    unittest.main()
