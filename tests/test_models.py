import unittest

import torch

from battery_fusion.models.lstm import RdfLSTMRegressor


class ModelTests(unittest.TestCase):
    def test_rdf_lstm_forward_shape(self) -> None:
        model = RdfLSTMRegressor(input_size=400, hidden_size=8, output_size=1)
        output = model(torch.ones(3, 400))
        self.assertEqual(tuple(output.shape), (3,))

    def test_rdf_lstm_uses_publication_400_feature_sequence_encoder(self) -> None:
        model = RdfLSTMRegressor(input_size=400, hidden_size=8, output_size=1)

        self.assertEqual(model.lstm_cell.input_size, 400)
        self.assertEqual(model.lstm_cell.hidden_size, 8)


if __name__ == "__main__":
    unittest.main()
