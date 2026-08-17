# IMU Reliability Passport for Medical Imaging AI

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamamh66/IMU-Reliability-Passport-Medical-Imaging-AI/blob/main/IMU_Reliability_Passport_Medical_Imaging_AI.ipynb)

A reproducible, Google Colab–ready implementation of the machine-readable reliability-passport framework described in:

> **Machine-Readable Reliability Passports for Medical Imaging AI: A Provenance-Aware Informatics Framework and Chest-Radiograph Case Study**

The notebook audits data identity, leakage, label conflict, provenance, shortcut risk, calibration, uncertainty, selective referral, reproducibility, and cross-dataset portability. It produces structured JSON/CSV evidence, publication-ready figures, and a compact results archive.

## Repository contents

| File | Purpose |
|---|---|
| `IMU_Reliability_Passport_Medical_Imaging_AI.ipynb` | Complete executable Colab workflow |
| `README.md` | Repository documentation and reproducibility guide |
| `requirements.txt` | Python dependencies for a local environment |

The notebook creates the following result directories at runtime:

```text
IMU_Reliability_Passport/
├── Figures/              # Six publication-oriented figures
├── Tables/               # Metrics, audits, claim gates, and bootstrap contrasts
├── Raw/                  # Configuration, environment, manifests, and passport JSON
├── Models/               # Fitted reference-model artifacts
├── HumanReview/          # Export for blinded adjudication
├── ComparisonRuns/       # Cross-environment comparison inputs
└── IMU_Reliability_Passport_Compact.zip
```

## What the notebook evaluates

1. **Dataset identity and integrity** — byte, decoded-pixel, and perceptual hashes; contradictory labels; cross-partition identity leakage.
2. **Controlled fault injection** — exact copies, re-encoding, resizing, brightness changes, metadata-only changes, and unrelated negative pairs.
3. **Group-aware CXR evaluation** — direct three-class and hierarchical reference pipelines with paired bootstrap contrasts.
4. **Shortcut controls** — metadata-only endpoint prediction, with an explicit block on unsupported clinical generalization.
5. **Reliability evidence** — calibration, conformal prediction, and selective referral.
6. **Portability** — the same audit modules applied to PathMNIST without changing the passport schema.
7. **Claim governance** — versioned claim–rule–evidence records that distinguish supported, conditional, blocked, and not-evaluated conclusions.

## Quick start in Google Colab

1. Upload the notebook to Colab or use the badge above after the repository is published.
2. Select a GPU runtime: **Runtime → Change runtime type → T4 GPU**.
3. Run all cells in order.
4. If Kaggle credentials are not already configured, upload `kaggle.json` when prompted.
5. Download `IMU_Reliability_Passport_Compact.zip` when the run finishes.

Large datasets, caches, and intermediate images remain under `/content`. Google Drive mounting is optional, so a full Drive quota does not stop the experiment. Compact Drive export is disabled by default and can be enabled in `CONFIG`.

## Local execution

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab IMU_Reliability_Passport_Medical_Imaging_AI.ipynb
```

CUDA is recommended for the reference-model sections. The audit-only sections can run on CPU.

## Data sources

- **Chest radiographs:** Kaggle dataset slug `muhammadrehan00/chest-xray-dataset`. The notebook downloads it through the Kaggle API when no local copy is found.
- **PathMNIST:** obtained programmatically through the `medmnist` package.

Dataset files are not redistributed by this repository. Users remain responsible for the original datasets' terms, licenses, and access requirements.

## Configuration and reproducibility

The principal defaults are:

| Parameter | Default |
|---|---:|
| Random seed | `20260816` |
| Passport schema | `1.0.0` |
| Workflow version | `0.2.0` |
| Perceptual-hash radius | `4` |
| Bootstrap replicates | `500` |
| Conformal error level | `0.10` |
| Target automatic coverage | `0.75` |
| Encoder | `mobilenetv3_small_100` |
| Image size | `160 × 160` |

Each run records the full configuration, software environment, dataset-manifest hashes, and an environment fingerprint. The generated passport is validated against the embedded JSON schema before export.

## Reference results from the article run

These values document the completed article run; they are not hard-coded as new experimental output.

| Endpoint | Result |
|---|---:|
| Controlled near-identity detection at radius 4 | Precision `1.000`, recall `0.900`, F1 `0.947`, FPR `0.000` |
| Direct CXR macro-F1 | `0.9790` |
| Hierarchical CXR macro-F1 | `0.9822` |
| Paired macro-F1 difference | `+0.0031`, 95% bootstrap CI `[-0.0021, 0.0090]` |
| Metadata-only macro-F1 | `0.9881` |
| Conformal empirical coverage | `0.9104` |
| Selective automatic coverage | `0.7395` |
| Accuracy among automatically accepted cases | `1.000` |
| PathMNIST portability | `7/8` modules completed; one pHash split-overlap group detected |
| Within-environment rerun | Exact agreement `1.000`; ARI `1.000` |

The high metadata-only score is treated as shortcut-risk evidence, not as evidence of clinical generalization. Patient-level independence, external clinical validation, a second software environment, and human adjudication were not established and remain explicitly gated.

## Important interpretation boundary

The workflow demonstrates **results-audit reproducibility** for the evaluated settings. It does not establish clinical safety, transportability across institutions, patient-level independence where identifiers are absent, or regulatory fitness. The generated claim gates should be read together with the underlying evidence tables.

## Citing this repository

```bibtex
@misc{Mallek2026IMUReliabilityPassport,
  author       = {Fatma Mallek and Rahma Zayoud and Habib Hamam},
  title        = {IMU Reliability Passport for Medical Imaging AI},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/hamamh66/IMU-Reliability-Passport-Medical-Imaging-AI}
}
```

## Authors

- Fatma Mallek
- Rahma Zayoud
- Habib Hamam

## Responsible use

This research software is intended for auditing and methodological evaluation. It is not a medical device and must not be used to make clinical decisions.
