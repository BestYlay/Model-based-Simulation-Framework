"""Landmark construction and sequence batching utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MISSING_TOKEN = "Missing"


def _validate_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def create_landmark_snapshots(
    frame: pd.DataFrame,
    landmark_times: Sequence[float],
) -> pd.DataFrame:
    """Create event histories and residual survival labels at each landmark.

    Events at the landmark itself are excluded to make the prediction cutoff
    explicit. Patients must still be under observation at the landmark.
    """

    _validate_columns(frame, ["PATIENT_ID", "START_DATE", "stop", "dead"])
    if not landmark_times:
        raise ValueError("landmark_times must contain at least one value")
    if any(time < 0 for time in landmark_times):
        raise ValueError("landmark_times must be non-negative")

    working = frame.copy()
    if "entry" not in working.columns:
        working["entry"] = 0.0

    snapshots: list[pd.DataFrame] = []
    for patient_id, patient_rows in working.groupby("PATIENT_ID", sort=False):
        outcomes = patient_rows[["stop", "dead", "entry"]].drop_duplicates()
        if len(outcomes) != 1:
            raise ValueError(f"Inconsistent outcomes for patient {patient_id!r}")

        outcome = outcomes.iloc[0]
        total_time = float(outcome["stop"] - outcome["entry"])
        if total_time < 0:
            continue

        patient_rows = patient_rows.sort_values("START_DATE", kind="stable")
        for landmark in sorted(set(landmark_times)):
            if total_time < landmark:
                continue
            history = patient_rows.loc[patient_rows["START_DATE"] < landmark].copy()
            if history.empty:
                continue

            history["SAMPLE_ID"] = f"{patient_id}_{landmark:g}"
            history["time_landmark"] = float(landmark)
            history["time"] = total_time - float(landmark)
            history["dead"] = float(outcome["dead"])
            history.drop(columns=["time_total"], errors="ignore", inplace=True)
            snapshots.append(history)

    if not snapshots:
        raise ValueError("No eligible landmark snapshots were created")

    result = pd.concat(snapshots, ignore_index=True)
    if not result.columns.is_unique:
        raise RuntimeError("Landmark construction produced duplicate column names")
    return result


def fit_categorical_vocabs(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Fit deterministic vocabularies using training data only."""

    _validate_columns(frame, columns)
    vocabs: dict[str, dict[str, int]] = {}
    for column in columns:
        values = frame[column].fillna(MISSING_TOKEN).astype(str)
        categories = sorted(values.unique().tolist())
        vocab = {PAD_TOKEN: 0}
        vocab.update({value: index + 1 for index, value in enumerate(categories)})
        vocab[UNK_TOKEN] = len(vocab)
        vocabs[column] = vocab
    return vocabs


def apply_categorical_vocabs(
    frame: pd.DataFrame,
    vocabs: Mapping[str, Mapping[str, int]],
) -> pd.DataFrame:
    """Apply pre-fitted vocabularies without leaking validation categories."""

    _validate_columns(frame, vocabs.keys())
    transformed = frame.copy()
    for column, vocab in vocabs.items():
        if PAD_TOKEN not in vocab or UNK_TOKEN not in vocab:
            raise ValueError(f"Vocabulary for {column!r} lacks PAD or UNK token")
        values = transformed[column].fillna(MISSING_TOKEN).astype(str)
        transformed[f"{column}_encoded"] = (
            values.map(vocab).fillna(vocab[UNK_TOKEN]).astype(np.int64)
        )
    return transformed


def fit_standardizer(frame: pd.DataFrame, columns: Sequence[str]) -> StandardScaler | None:
    """Fit a numerical standardizer on training rows."""

    if not columns:
        return None
    _validate_columns(frame, columns)
    scaler = StandardScaler()
    scaler.fit(frame[list(columns)])
    return scaler


def apply_standardizer(
    frame: pd.DataFrame,
    scaler: StandardScaler | None,
) -> pd.DataFrame:
    """Apply a pre-fitted standardizer to its original feature columns."""

    transformed = frame.copy()
    if scaler is None:
        return transformed
    columns = list(scaler.feature_names_in_)
    _validate_columns(transformed, columns)
    transformed[columns] = scaler.transform(transformed[columns])
    return transformed


class PatientSequenceDataset(Dataset):
    """Convert event rows into one chronological sequence per SAMPLE_ID."""

    def __init__(
        self,
        frame: pd.DataFrame,
        numerical_columns: Sequence[str],
        categorical_encoded_columns: Sequence[str],
    ) -> None:
        required = [
            "SAMPLE_ID",
            "time",
            "dead",
            *numerical_columns,
            *categorical_encoded_columns,
        ]
        _validate_columns(frame, required)
        self._samples: list[dict[str, object]] = []

        for sample_id, rows in frame.groupby("SAMPLE_ID", sort=False):
            if "START_DATE" in rows.columns:
                rows = rows.sort_values("START_DATE", kind="stable")
            if rows["time"].nunique(dropna=False) != 1 or rows["dead"].nunique(dropna=False) != 1:
                raise ValueError(f"Inconsistent labels within sample {sample_id!r}")
            numerical = torch.as_tensor(
                rows[list(numerical_columns)].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
            categorical = {
                column.removesuffix("_encoded"): torch.as_tensor(
                    rows[column].to_numpy(dtype=np.int64), dtype=torch.long
                )
                for column in categorical_encoded_columns
            }
            self._samples.append(
                {
                    "sample_id": str(sample_id),
                    "numerical": numerical,
                    "categorical": categorical,
                    "time": torch.tensor(float(rows["time"].iloc[0]), dtype=torch.float32),
                    "event": torch.tensor(float(rows["dead"].iloc[0]), dtype=torch.float32),
                }
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._samples[index]


def collate_patient_sequences(samples: Sequence[dict[str, object]]) -> dict[str, object]:
    """Pad variable-length sequences and return an explicit padding mask."""

    if not samples:
        raise ValueError("Cannot collate an empty batch")
    numerical_sequences = [sample["numerical"] for sample in samples]
    lengths = torch.tensor([sequence.shape[0] for sequence in numerical_sequences])
    numerical = pad_sequence(numerical_sequences, batch_first=True, padding_value=0.0)
    positions = torch.arange(numerical.shape[1]).unsqueeze(0)
    padding_mask = positions >= lengths.unsqueeze(1)

    categorical_keys = list(samples[0]["categorical"].keys())
    categorical = {
        key: pad_sequence(
            [sample["categorical"][key] for sample in samples],
            batch_first=True,
            padding_value=0,
        )
        for key in categorical_keys
    }
    return {
        "sample_ids": [sample["sample_id"] for sample in samples],
        "numerical": numerical,
        "categorical": categorical,
        "padding_mask": padding_mask,
        "time": torch.stack([sample["time"] for sample in samples]),
        "event": torch.stack([sample["event"] for sample in samples]),
    }
