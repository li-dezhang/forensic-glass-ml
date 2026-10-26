# Forensic Glass Classification and Uncertainty-Aware Machine Learning

A small, real, reproducible Python research project using **UCI Glass Identification**. It compares conventional classifiers and explores confidence-based abstention. This is an AI-assisted, unpublished training study—not individual-source identification or a validated forensic decision system.

[English report](reports/REPORT_EN.md) · [中文报告](reports/REPORT_ZH.md) · [CV evidence](reports/CV_EVIDENCE.md)

## Questions

1. How accurately can conventional models classify broad glass categories using refractive index and elemental composition?
2. How does withholding low-confidence predictions affect accuracy among accepted predictions and sample coverage?

## Data and design

Official data were downloaded with `ucimlrepo.fetch_ucirepo(id=42)`: **214 records, 9 numeric predictors, 6 present classes**. [Source, dictionary and licence](data/README.md).

One exact feature/label duplicate is removed for modeling (ID 40 duplicates ID 39), leaving **213 records: 159 train / 54 test**. Original data remain intact. ID never enters a model. Random seed is **42**. Five-fold stratified CV uses only training data; StandardScaler is inside Logistic Regression and kNN pipelines. No hyperparameter search or test-set tuning.

The winner is selected by mean CV macro-F1 before all four fixed models are evaluated on the held-out set. See the [frozen protocol](EXPERIMENT_PROTOCOL.md).

## Actual results

| Model | CV macro-F1 | Test accuracy | Test macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| Dummy majority | 0.0879 | 0.3519 | 0.0868 | 0.1667 |
| Logistic Regression | 0.4717 | 0.6852 | 0.6702 | 0.7164 |
| kNN | 0.4839 | 0.7407 | 0.6088 | 0.6107 |
| Random Forest | **0.7255** | **0.7778** | **0.7585** | **0.7691** |

Random Forest correctly classifies **42/54** held-out records. These results do not establish universal superiority.

| Probability threshold | Coverage | Selective accuracy | Review required |
|---|---:|---:|---:|
| 0.50 | 43/54 (79.63%) | 38/43 (88.37%) | 11 |
| 0.60 | 33/54 (61.11%) | 32/33 (96.97%) | 21 |
| 0.70 | 26/54 (48.15%) | 25/26 (96.15%) | 28 |
| 0.80 | 18/54 (33.33%) | 17/18 (94.44%) | 36 |

Low-confidence outputs are marked `REVIEW_REQUIRED` in [confidence_predictions.csv](results/confidence_predictions.csv). Selective accuracy excludes reviewed cases; it is not all-sample accuracy. Higher thresholds did not monotonically improve it. One wrong prediction has probability 0.90; class 3 receives zero automatic coverage at thresholds 0.70/0.80. Raw probabilities are **uncalibrated**, and the thresholds are exploratory, not evidence-reliability guarantees.

![Comparison](results/model_comparison.png)
![Confidence tradeoff](results/confidence_tradeoff.png)

Additional figures: [confusion matrix](results/confusion_matrix.png), [class distribution](results/class_distribution.png).

## Install and run

Verified on **macOS 27.0.1 ARM64 / Python 3.13.2**. Versions are pinned in `requirements.txt` and recorded in `results/run.json`. Other platforms have not been tested.

From this project directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt
.venv/bin/python src/run.py --out results/my-run
.venv/bin/python src/verify.py --run results/my-run --compare results
.venv/bin/python -m unittest discover -s tests -v
```

Use a fresh output directory each time. Existing measured results are never silently overwritten by model fitting. With no `--out` argument, the default is `results/`; the supplied results already occupy that directory, so use the fresh-directory command above.

Acquisition uses the bundled, hash-verified official table when available. In a fresh data directory, the code first downloads via ucimlrepo; if that fails, it prints the actual failure and tries the official UCI CSV. No fallback was needed in the completed run; that error branch is not claimed to have been live-tested. No credentials or synthetic research data are used.

For separate stages:

```sh
.venv/bin/python src/run.py --stage models --out results/staged-run
.venv/bin/python src/run.py --stage analysis --out results/staged-run
```

To check and regenerate supplied figures/analysis without retraining:

```sh
.venv/bin/python src/verify.py
```

Unit/integration tests inspect the supplied `results/` and real data. The verifier independently recomputes metrics from predictions, regenerates analysis and figures, and compares hashes. The completed second full fit reproduced **13 CSV files and 4 PNGs** exactly. [Verification](results/verification.json) · [12 passing tests](results/tests.txt). This is same-platform reproducibility, not a guarantee of identical images on other versions/platforms.

## Files

- `src/run.py`: acquisition, quality checks, fixed split, train-only CV, comparison, error analysis and abstention.
- `src/verify.py`: independent count-based metric checks and deterministic output comparison.
- `tests/test_study.py`: data/schema, leakage precautions, outputs, scoring and abstention edge cases.
- `data/`: official table cache, dictionary, metadata, attribution and SHA-256.
- `results/metrics.csv`: CV/test metrics for four models.
- `results/predictions.csv`: 216 model–sample predictions, labels and six class probabilities.
- `results/`: split/fold IDs, class counts, errors, confusion matrix, class recalls, abstention summaries and figures.
- `reports/`: English/Chinese interpretation and factual CV materials.

## Limitations and attribution

One small historical dataset and one random split; no external validation, calibrated probabilities, unseen-class detection, deployment or laboratory experiments. Rare-class test denominators are as small as two. Exact deduplication does not establish physical-source independence. No individual-source, causal or courtroom-readiness claim is supported.

Data: German, B. (1987), [UCI Glass Identification](https://doi.org/10.24432/C5WW2P), **CC BY 4.0**. Citations for preprocessing and selective classification appear in the reports. Dataset licensing does not automatically license the original project code.

Codex/AI implemented and ran the code and assisted with analysis and writing. Student contributions and understanding must be confirmed before using first-person CV claims. No GitHub repository was created or pushed.

