# IMU Reliability Passport for Medical Imaging AI

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamamh66/IMU-Reliability-Passport-Medical-Imaging-AI/blob/main/notebooks/IMU_Reliability_Passport_Experiments_v0.3.ipynb)

A reproducible, Google Colab-ready implementation of the machine-readable reliability-passport framework described in:

> **Machine-Readable Reliability Passports for Medical Imaging AI: A Provenance-Aware Informatics Framework and Chest-Radiograph Case Study**

The notebook audits data identity, leakage, label conflict, provenance, shortcut risk, calibration, uncertainty, selective referral, reproducibility, and cross-dataset portability. It produces structured JSON/CSV evidence, publication-ready figures, and a compact results archive.

## Repository contents

| Path | Purpose |
|---|---|
| `notebooks/IMU_Reliability_Passport_Experiments_v0.2.ipynb` | Software version 0.2 — the run that produced the reported results (passport schema 1.1.0) |
| `notebooks/IMU_Reliability_Passport_Experiments_v0.3.ipynb` | Software version 0.3 — emits `model_scope`, per-model prediction digests and gate C5b natively (passport schema 1.1.1) |
| `IMU_Reliability_Passport_Experiments.ipynb` | Version 0.1, kept for provenance; superseded by the notebooks above |
| `passport/` | Passport records and JSON Schemas (1.1.0 and 1.1.1), claim-gate tables, and `regenerate_claim_gates.py` |
| `results/` | Evidence files from the reported run: metrics, audits, bootstrap contrasts, conformal and selective-referral summaries, configuration and environment fingerprints |
| `requirements.txt` | Python dependencies for a local environment |
| `LICENSE` | License |

The notebook creates the following result directories at runtime:

```text
IMU_Reliability_Passport/
├── Figures/              # Publication-oriented figures
├── Tables/               # Metrics, audits, claim gates, and bootstrap contrasts
├── Raw/                  # Configuration, environment, manifests, and passport JSON
├── Models/               # Fitted reference-model artifacts
├── HumanReview/          # Export for blinded adjudication
├── ComparisonRuns/       # Cross-environment comparison inputs
└── IMU_Reliability_Passport_Compact.zip
```

## What the notebook evaluates

1. **Dataset identity and integrity** — byte, decoded-pixel, and perceptual hashes; contradictory labels; cross-partition identity leakage.
2. **Controlled fault injection** — exact copies, re-encoding, resizing, brightness changes, source-border changes, metadata-only changes, and unrelated negative pairs.
3. **Group-aware CXR evaluation** — direct three-class, hierarchical, and fully fine-tuned pipelines with paired bootstrap contrasts.
4. **Shortcut controls** — metadata-only endpoint prediction, with an explicit block on unsupported clinical generalization.
5. **Reliability evidence** — calibration, class-conditional (Mondrian) split conformal prediction, and selective referral.
6. **Portability** — the same audit modules applied to PathMNIST without changing the passport schema.
7. **Claim governance** — versioned claim–rule–evidence records with declared prerequisites, effective-state propagation, threshold provenance, and model scope.

## The passport schema

| Version | Adds |
|---|---|
| `1.0.0` | Claim–rule–evidence records |
| `1.1.0` | `depends_on`, `effective_state` (prerequisite propagation), `threshold_provenance` |
| `1.1.1` | `model_scope` — binds a model-dependent gate to the model it was computed from, plus a canonical digest of that model's test-set probability matrix |

Gate dependencies: C1 → C2, C2 → C4, {C2, C4} → C5 and C5b. A gate that is locally supported but has a non-supported prerequisite becomes `conditional`.

`passport/regenerate_claim_gates.py` derives the 1.1.1 records from a 1.1.0 run without recomputing anything:

```bash
python passport/regenerate_claim_gates.py results   # writes passport_1.1.1.json, schema, and CSV
```

Version 0.3 of the notebook emits these records directly.

## Quick start in Google Colab

1. Open the badge above, or upload a notebook from `notebooks/` to Colab.
2. Select a GPU runtime: **Runtime → Change runtime type → T4 GPU**.
3. Run all cells in order.
4. If Kaggle credentials are not already configured, upload `kaggle.json` when prompted.
5. Download `IMU_Reliability_Passport_Compact.zip` when the run finishes.

Keep `output_root` under `/content` (the default) rather than on Drive: a full Drive quota can silently truncate the exported outputs. Set `auto_download_archive: True` in `CONFIG` to have the compact archive downloaded at the end of the run.

## Local execution

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab notebooks/IMU_Reliability_Passport_Experiments_v0.3.ipynb
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
| Passport schema | `1.1.0` (v0.2) / `1.1.1` (v0.3) |
| Software version | `0.2.0` / `0.3.0` |
| Perceptual-hash radius | `4` (sweep: `0, 2, 4, 6, 8`) |
| Bootstrap replicates | `2000` |
| Conformal error level | `0.10` |
| Target automatic coverage | `0.75` |
| Encoder | `mobilenetv3_small_100` |
| Image size | `160 × 160` |

Each run records the full configuration, software environment, dataset-manifest hashes, and an environment fingerprint. The generated passport is validated against the embedded JSON schema before export.

## Reference results from the article run

These values document the completed run whose evidence files are in `results/`; they are not hard-coded as new experimental output.

| Endpoint | Result |
|---|---:|
| Near-identity detection at radius 4 (all derived pairs) | Precision `1.000`, pooled recall `0.899` (728/810), F1 `0.947`, FPR `0.000` |
| Residual misses | Source-border derivatives only (88/90 detected at the stratum level) |
| Two-stage border-aware detector (radius 8 + embedding cosine, τ = 0.80) | Pooled recall `1.000`, FPR `0.000` |
| Direct CXR macro-F1 | `0.9790` |
| Hierarchical CXR macro-F1 | `0.9822` |
| Paired macro-F1 difference (hierarchical − direct) | `+0.0033`, 95% bootstrap CI `[-0.0020, 0.0089]` |
| Fine-tuned CXR macro-F1 | `0.9930` (ECE `0.0053`) |
| Metadata-only macro-F1 | `0.9881` |
| Conformal marginal coverage (hierarchical, C5) | `0.9104` |
| Conformal marginal coverage (fine-tuned, C5b) | `0.8922` |
| Selective automatic coverage | `0.7395` |
| Accuracy among automatically accepted cases | `1.000` (rule-of-three upper bound on selective risk `0.0028`) |
| PathMNIST portability | `7/8` modules completed; reliability module not evaluated (no image model) |
| Within-environment rerun | Exact agreement `1.000` |

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
