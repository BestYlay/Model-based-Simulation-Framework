"""Train the core model on synthetic longitudinal data in under a minute."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from survival_simulation import (
    PatientSequenceDataset,
    SurvivalTransformer,
    apply_categorical_vocabs,
    apply_standardizer,
    collate_patient_sequences,
    create_landmark_snapshots,
    evaluate_loss,
    fit_categorical_vocabs,
    fit_standardizer,
    predict_risk,
    resolve_device,
    set_seed,
    train_one_epoch,
)

NUMERICAL_COLUMNS = ["START_DATE", "VALUE_NUMERIC", "EVENT_DURATION", "AGE"]
CATEGORICAL_COLUMNS = ["EVENT_TYPE", "EVENT_SUBTYPE", "VALUE_CATEGORICAL"]


def make_synthetic_events(patient_count: int = 48, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    event_types = np.array(["Diagnosis", "Imaging", "Lab", "Treatment"])
    for patient_number in range(patient_count):
        patient_id = f"P-{patient_number:04d}"
        age = int(rng.integers(35, 85))
        event_count = int(rng.integers(4, 10))
        dates = np.sort(rng.integers(-180, 720, size=event_count))
        latent_risk = (age - 60) / 20 + rng.normal(0, 0.4)
        total_time = max(760.0, 1500.0 - 220.0 * latent_risk + rng.normal(0, 100))
        dead = float(rng.random() < 0.65)
        for date in dates:
            event_type = str(rng.choice(event_types))
            rows.append(
                {
                    "PATIENT_ID": patient_id,
                    "START_DATE": float(date),
                    "EVENT_DURATION": float(max(0, rng.normal(14, 20))),
                    "VALUE_NUMERIC": float(latent_risk + rng.normal(0, 0.5)),
                    "EVENT_TYPE": event_type,
                    "EVENT_SUBTYPE": f"{event_type}_event",
                    "VALUE_CATEGORICAL": "abnormal" if latent_risk > 0 else "normal",
                    "AGE": float(age),
                    "entry": 0.0,
                    "stop": total_time,
                    "dead": dead,
                }
            )
    return pd.DataFrame(rows)


def prepare_split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict, object]:
    patient_ids = frame["PATIENT_ID"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)
    boundary = int(len(patient_ids) * 0.75)
    train_ids = set(patient_ids[:boundary])
    train = frame[frame["PATIENT_ID"].isin(train_ids)].copy()
    validation = frame[~frame["PATIENT_ID"].isin(train_ids)].copy()

    vocabs = fit_categorical_vocabs(train, CATEGORICAL_COLUMNS)
    scaler = fit_standardizer(train, NUMERICAL_COLUMNS)
    train = apply_standardizer(apply_categorical_vocabs(train, vocabs), scaler)
    validation = apply_standardizer(apply_categorical_vocabs(validation, vocabs), scaler)
    return train, validation, vocabs, scaler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    set_seed(42)
    device = resolve_device(args.device)
    events = make_synthetic_events()
    landmarks = create_landmark_snapshots(events, landmark_times=[180, 365])
    train, validation, vocabs, _ = prepare_split(landmarks, seed=42)
    encoded_columns = [f"{column}_encoded" for column in CATEGORICAL_COLUMNS]

    train_loader = DataLoader(
        PatientSequenceDataset(train, NUMERICAL_COLUMNS, encoded_columns),
        batch_size=16,
        shuffle=True,
        collate_fn=collate_patient_sequences,
    )
    validation_loader = DataLoader(
        PatientSequenceDataset(validation, NUMERICAL_COLUMNS, encoded_columns),
        batch_size=16,
        shuffle=False,
        collate_fn=collate_patient_sequences,
    )

    vocab_sizes = {column: len(vocab) for column, vocab in vocabs.items()}
    embedding_dims = {column: min(16, len(vocab)) for column, vocab in vocabs.items()}
    model = SurvivalTransformer(
        vocab_sizes,
        embedding_dims,
        num_numerical_features=len(NUMERICAL_COLUMNS),
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=64,
        dropout=0.1,
        pooling="attention",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    print(f"device={device} train_samples={len(train_loader.dataset)} val_samples={len(validation_loader.dataset)}")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        validation_loss = evaluate_loss(model, validation_loader, device)
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={validation_loss:.4f}")

    sample_ids, risks = predict_risk(model, validation_loader, device)
    print(f"generated {len(risks)} validation risks; first_sample={sample_ids[0]} risk={risks[0]:.4f}")


if __name__ == "__main__":
    main()
