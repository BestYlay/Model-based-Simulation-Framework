# Preparing the MSK-CHORD data

The project expects the public **MSK-CHORD (Nature 2024)** study from cBioPortal:

- Study page: <https://www.cbioportal.org/study/summary?id=msk_chord_2024>
- Raw download: timeline files such as `data_timeline_*.txt`
- Processed download: cohort files such as `*_dx_1st_seq_OS.csv`

The data are not committed to this repository. Review and follow the provider's license and data
use terms.

## 1. Import into SQLite

Create `msk_chord.db` with any SQLite client. Import the timeline tables using these names:

```text
timeline_diagnosis
timeline_cancer_presence
timeline_tumor_sites
timeline_progression
timeline_cea_labs
timeline_psa_labs
timeline_ca_15-3_labs
timeline_ca_19-9_labs
timeline_treatment
timeline_radiation
timeline_prior_meds
timeline_surgery
timeline_gleason
timeline_mmr
timeline_pdl1
timeline_specimen_surgery
timeline_specimen
timeline_performance_status
```

Import the five processed cohort CSV files as:

```text
source_brca_dx_1st_seq_OS
source_crc_dx_1st_seq_OS
source_nsclc_dx_1st_seq_OS
source_panc_dx_1st_seq_OS
source_prostate_dx_1st_seq_OS
```

## 2. Build engineered views

Run `process_raw_data_in_sqlite.sql`. It creates one unified event timeline and five cohort views:

```text
brca_survival_lstm_dataset
crc_survival_lstm_dataset
nsclc_survival_lstm_dataset
panc_survival_lstm_dataset
prostate_survival_lstm_dataset
```

## 3. Export and landmark

`load_data.ipynb` exports each cohort view and computes follow-up time from `stop - entry`.
`process_data.ipynb` is the original exploratory landmarking notebook. For reusable code, prefer:

```python
import pandas as pd
from survival_simulation import create_landmark_snapshots

events = pd.read_csv("df_brca.csv")
landmarks = create_landmark_snapshots(events, [180, 365, 545, 730, 1095, 1825])
landmarks.to_csv("df_brca_landmarks.csv", index=False)
```

All generated CSV, SQLite, model, and plot files are excluded by `.gitignore`.
