import unittest

import numpy as np
import pandas as pd
import torch

from battery_fusion.training.target_transform import TargetTransform, fit_target_transform


class TargetTransformTests(unittest.TestCase):
    def test_standardize_round_trips_arrays_and_tensors(self) -> None:
        transform = fit_target_transform(np.array([10.0, 20.0, 30.0]), "standardize")

        scaled = transform.transform_array(np.array([10.0, 20.0, 30.0]))
        restored = transform.inverse_array(scaled)
        self.assertTrue(np.allclose(restored, np.array([10.0, 20.0, 30.0])))
        self.assertAlmostEqual(float(np.mean(scaled)), 0.0)
        self.assertAlmostEqual(float(np.std(scaled)), 1.0)

        tensor = torch.tensor([10.0, 20.0, 30.0])
        self.assertTrue(torch.allclose(transform.inverse_tensor(transform.transform_tensor(tensor)), tensor))

    def test_none_transform_leaves_values_unchanged(self) -> None:
        transform = fit_target_transform(np.array([5.0, 7.0]), "none")

        values = np.array([1.0, 2.0])
        self.assertTrue(np.allclose(transform.transform_array(values), values))
        self.assertTrue(np.allclose(transform.inverse_array(values), values))

    def test_from_config_handles_legacy_missing_values(self) -> None:
        transform = TargetTransform.from_config({})

        self.assertEqual(transform.kind, "none")
        self.assertEqual(transform.mean, 0.0)
        self.assertEqual(transform.std, 1.0)

    def test_prediction_frame_inverse_transforms_predictions_and_targets(self) -> None:
        transform = TargetTransform(kind="standardize", mean=10.0, std=2.0)
        frame = pd.DataFrame({"sample_id": ["a"], "y_true": [0.5], "y_pred": [1.5]})

        restored = transform.inverse_prediction_frame(frame, inverse_y_true=True)

        self.assertAlmostEqual(restored.loc[0, "y_true"], 11.0)
        self.assertAlmostEqual(restored.loc[0, "y_pred"], 13.0)


if __name__ == "__main__":
    unittest.main()
