import unittest

from battery_fusion.experiments.modality_dropout import modality_dropout_conditions


class PublicationModalityDropoutTests(unittest.TestCase):
    def test_publication_dropout_conditions_match_documented_protocol(self) -> None:
        conditions = modality_dropout_conditions()
        by_name = {condition.name: condition.available_modalities for condition in conditions}

        self.assertEqual(
            list(by_name),
            [
                "full",
                "drop_composition",
                "drop_graph",
                "drop_rdf",
                "composition_only_fallback",
                "graph_only_fallback",
                "rdf_only_fallback",
            ],
        )
        self.assertEqual(by_name["full"], ("tabular", "structure", "rdf"))
        self.assertEqual(by_name["drop_composition"], ("structure", "rdf"))
        self.assertEqual(by_name["drop_graph"], ("tabular", "rdf"))
        self.assertEqual(by_name["drop_rdf"], ("tabular", "structure"))
        self.assertEqual(by_name["composition_only_fallback"], ("tabular",))
        self.assertEqual(by_name["graph_only_fallback"], ("structure",))
        self.assertEqual(by_name["rdf_only_fallback"], ("rdf",))


if __name__ == "__main__":
    unittest.main()
