import unittest

import pandas as pd
import torch

from survival_simulation.data import (
    PatientSequenceDataset,
    apply_categorical_vocabs,
    collate_patient_sequences,
    create_landmark_snapshots,
    fit_categorical_vocabs,
)


class LandmarkTests(unittest.TestCase):
    def test_landmark_labels_are_residual_and_columns_are_unique(self):
        frame = pd.DataFrame(
            {
                "PATIENT_ID": ["P1", "P1", "P1"],
                "START_DATE": [-10, 100, 300],
                "stop": [500, 500, 500],
                "entry": [50, 50, 50],
                "dead": [1, 1, 1],
                "time": [450, 450, 450],
                "feature": [1.0, 2.0, 3.0],
            }
        )

        result = create_landmark_snapshots(frame, [200])

        self.assertTrue(result.columns.is_unique)
        self.assertEqual(result["SAMPLE_ID"].nunique(), 1)
        self.assertEqual(result["time"].unique().tolist(), [250.0])
        self.assertEqual(result["dead"].unique().tolist(), [1.0])
        self.assertEqual(result["START_DATE"].tolist(), [-10, 100])

    def test_unknown_category_uses_unk_token(self):
        train = pd.DataFrame({"kind": ["A", "B", None]})
        validation = pd.DataFrame({"kind": ["C"]})
        vocabs = fit_categorical_vocabs(train, ["kind"])

        result = apply_categorical_vocabs(validation, vocabs)

        self.assertEqual(result["kind_encoded"].iloc[0], vocabs["kind"]["<UNK>"])


class SequenceBatchTests(unittest.TestCase):
    def test_collate_sorts_events_and_builds_padding_mask(self):
        frame = pd.DataFrame(
            {
                "SAMPLE_ID": ["A", "A", "B"],
                "START_DATE": [20.0, 10.0, 15.0],
                "value": [2.0, 1.0, 3.0],
                "kind_encoded": [1, 2, 1],
                "time": [100.0, 100.0, 80.0],
                "dead": [1.0, 1.0, 0.0],
            }
        )
        dataset = PatientSequenceDataset(frame, ["value"], ["kind_encoded"])

        batch = collate_patient_sequences([dataset[0], dataset[1]])

        self.assertTrue(torch.equal(batch["numerical"][0, :, 0], torch.tensor([1.0, 2.0])))
        self.assertTrue(
            torch.equal(batch["padding_mask"], torch.tensor([[False, False], [False, True]]))
        )


if __name__ == "__main__":
    unittest.main()
