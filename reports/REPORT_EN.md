# Forensic Glass Classification and Uncertainty-Aware Machine Learning

Computational research-training report · 22 October 2026 · AI-assisted, unpublished  
[Code](../src/run.py) · [Exact metrics](../results/metrics.csv) · [Verification](../results/verification.json)

## 1. Background

Refractive index and elemental composition provide measurable characteristics for broad glass-category classification. A classifier's category prediction, however, is not an identification of a particular broken window, object, crime scene or individual source. This project evaluates conventional machine-learning methods on a small public benchmark and examines what happens when predictions with low model confidence are withheld.

The goal is a transparent computational exercise, not a forensic decision system. Classification with a reject option trades automatic coverage against error among retained predictions; this general distinction is discussed by El-Yaniv and Wiener [4]. Their theoretical guarantees are not established or claimed for the simple probability-threshold procedure implemented here.

## 2. Research Questions

**RQ1:** How accurately can conventional machine-learning models classify broad glass categories using refractive index and elemental composition?

**RQ2:** How does withholding low-confidence predictions affect classification accuracy and sample coverage?

The questions concern this recorded dataset and fixed experiment. The study does not estimate likelihood ratios, individual-source probabilities or admissibility of forensic evidence.

## 3. Dataset

The official UCI Glass Identification dataset [1] was obtained successfully using `ucimlrepo.fetch_ucirepo(id=42)` on 22 October 2026. No fallback or synthetic observations were used. The downloaded table has 214 records and nine numeric predictors: RI, Na, Mg, Al, Si, K, Ca, Ba and Fe. RI is refractive index; the remaining measurements are expressed as weight percent in the corresponding oxide. The dictionary and acquisition metadata are preserved locally.

Six labels occur: 1, 2, 3, 5, 6 and 7; category 4 is absent. The recorded categories are not invented ground truth, but they have not been independently re-established by laboratory examination in this project.

| Code | Recorded category | Original | After deduplication | Train | Test |
|---|---|---:|---:|---:|---:|
| 1 | Building windows, float processed | 70 | 69 | 51 | 18 |
| 2 | Building windows, non-float processed | 76 | 76 | 57 | 19 |
| 3 | Vehicle windows, float processed | 17 | 17 | 13 | 4 |
| 5 | Containers | 13 | 13 | 10 | 3 |
| 6 | Tableware | 9 | 9 | 7 | 2 |
| 7 | Headlamps | 29 | 29 | 21 | 8 |
| Total | | 214 | 213 | 159 | 54 |

Actual inspection found no missing cells, nonfinite feature values or conflicting labels among identical feature vectors. There are no duplicate complete rows when ID is included, but IDs 39 and 40 have identical nine-feature measurements and the same label. ID 40 was excluded from modeling using a fixed first-occurrence rule. The original table remains intact. This conservatively prevents identical measurements from entering different partitions; it does not prove that the records originated from one physical object. IDs are used only for tracing examples, never as predictors.

The cached `glass.csv` is a serialization of the official returned table, not a claim of byte identity with the HTTP response. Its SHA-256 is `c4677c97a86cc9f63c5756f13671a7a9ffb3f7dc3031524023deebcae75d14ad`. The data are credited under CC BY 4.0.

## 4. Methods

The [protocol](../EXPERIMENT_PROTOCOL.md) was fixed before model fitting. A stratified 75/25 split with random_state=42 produced 159 training and 54 test records after deduplication. Five-fold stratified cross-validation, shuffled with the same seed, used the training partition only. Every fold had all six classes. No imputation, outlier filtering, oversampling, synthetic augmentation or hyperparameter search was performed.

Four fixed candidates were evaluated: a most-frequent-class DummyClassifier; Logistic Regression with C=1, lbfgs and max_iter=3000; kNN with five uniformly weighted neighbors and Euclidean distance; and Random Forest with 300 trees, unlimited depth, minimum leaf size one, square-root feature sampling and one processing thread. Randomized components use seed 42. StandardScaler is inside the Logistic Regression and kNN pipelines, so each CV training fold fits its own transformation. Test rows do not influence these fitted transforms [2].

Mean training CV macro-F1 selects the winner, with model-order tie breaking fixed in advance. Selection is recorded before any test predictions are evaluated. All four models are subsequently refitted on the 159 training rows and evaluated on the same 54 held-out rows for a descriptive comparison. The test table does not reselect the winner.

Accuracy counts correct predictions divided by 54. Macro-F1 averages six class-specific F1 scores with zero_division=0. Balanced accuracy averages class recall; the confusion matrix uses true classes as rows and predicted classes as columns. Rare classes retain equal weight in macro metrics.

For the CV-selected model, maximum raw `predict_proba` is compared with fixed thresholds 0.50, 0.60, 0.70 and 0.80. Equality is accepted; lower scores produce `REVIEW_REQUIRED`. Coverage is accepted count divided by all test records, selective accuracy is correct accepted predictions divided by accepted count, and review count is the remainder. A zero accepted count gives undefined selective accuracy, not zero or perfect performance. Correct automatic decisions divided by all test records and class-specific coverage are also saved. Threshold exploration does not recalibrate probabilities or simulate successful human review [3].

## 5. Results

### Model comparison

| Model | Training CV macro-F1 mean | CV SD | Test accuracy | Test macro-F1 | Test balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Dummy majority | 0.0879 | 0.0028 | 0.3519 | 0.0868 | 0.1667 |
| Logistic Regression | 0.4717 | 0.0422 | 0.6852 | 0.6702 | 0.7164 |
| kNN | 0.4839 | 0.1169 | 0.7407 | 0.6088 | 0.6107 |
| Random Forest | 0.7255 | 0.0540 | 0.7778 | 0.7585 | 0.7691 |

CV SD is the sample standard deviation across five folds, not a confidence interval. Random Forest was selected by training CV and correctly classified 42/54 test records. The majority baseline correctly classified 19/54 while recognizing only category 2. kNN had higher overall accuracy than Logistic Regression but lower macro-F1 and balanced accuracy, illustrating why overall accuracy alone can obscure uneven class performance.

![Model comparison](../results/model_comparison.png)

### Error analysis

Random Forest's class recalls were 16/18 (88.89%) for class 1, 13/19 (68.42%) for class 2, 2/4 (50.00%) for class 3, 2/3 (66.67%) for class 5, 2/2 (100%) for class 6 and 7/8 (87.50%) for class 7.

Of twelve errors, five were confusions between building-window categories 1 and 2: three class-2 samples predicted as 1, and two class-1 samples predicted as 2. Two vehicle-window samples were predicted as building-window float glass. Class 3 had the lowest recall and was never correctly predicted by Logistic Regression or kNN on this test split. This is consistent with a difficult and sparsely represented category, but the experiment does not isolate imbalance as the cause.

![Confusion matrix](../results/confusion_matrix.png)

Sample ID 3 is a particularly informative error: true category 1, predicted category 2, maximum probability 0.90. It remains an error under every threshold. Sample IDs 152 and 153, both true class 3, were predicted as class 1 at probabilities approximately 0.58 and 0.587. Their removal at threshold 0.60 improves selective accuracy but also reduces class-3 coverage from four accepted samples to one.

### Confidence-based abstention

| Threshold | Accepted / total | Coverage | Correct / accepted | Selective accuracy | Review required |
|---|---:|---:|---:|---:|---:|
| No abstention | 54/54 | 100.00% | 42/54 | 77.78% | 0 |
| 0.50 | 43/54 | 79.63% | 38/43 | 88.37% | 11 |
| 0.60 | 33/54 | 61.11% | 32/33 | 96.97% | 21 |
| 0.70 | 26/54 | 48.15% | 25/26 | 96.15% | 28 |
| 0.80 | 18/54 | 33.33% | 17/18 | 94.44% | 36 |

![Coverage and selective accuracy](../results/confidence_tradeoff.png)

Selective accuracy does not increase monotonically. The same high-confidence error survives, while stricter thresholds remove additional correct predictions. At 0.80, only 17/54 records receive a correct automatic decision; 36 remain unresolved. Class 3 has zero coverage at 0.70 and 0.80, and class 6 has zero coverage at 0.80. Their conditional accuracies are undefined. Therefore, high aggregate selective accuracy should not be read as good automatic coverage of every category.

## 6. Interpretation

The results support useful discrimination within this benchmark, with Random Forest strongest among the prespecified candidates on the training criterion. They do not establish universal model superiority. The minority classes have very small test denominators: one decision changes class-6 recall by fifty percentage points. Its observed 100% recall is only two successful predictions.

Abstention can concentrate correct predictions into a smaller subset, but the benefit is conditional on accepting a substantial review workload. The highest observed selective accuracy at 0.60 is an exploratory result on the same held-out sample, not a validated recommendation for operational threshold selection. A fresh validation design would be needed to choose and externally test an operating policy. Raw probabilities are not calibrated evidence reliability, and the retained 0.90-probability error provides a concrete warning.

The implementation and outputs are reviewable through sample IDs, stored class probabilities, CV fold assignments, explicit denominators and source hashes. Twelve unit/integration tests passed. Independent confusion-count calculations reproduce model metrics; a second full fit and deterministic plot regeneration reproduce thirteen CSVs and four PNGs exactly. Platform and package versions are recorded. A later change standardized plotting style across separate-stage and combined execution; it did not change model settings, predictions or scores.

## 7. Limitations

This is one small, historical benchmark and one random split. There is no repeated-split uncertainty assessment, external laboratory validation, modern instrument comparison or temporal/geographical holdout. Exact measurement deduplication cannot exclude shared-source dependence or near duplicates. Class 4 is absent. Unknown or out-of-distribution glass is not modeled. Probabilities were not calibrated, reviewer performance was not studied, and no costs were assigned to different mistakes.

The task classifies broad recorded categories, not individual objects or common sources. It provides neither evidential likelihood ratios nor a scientifically established courtroom decision rule. No model is claimed to meet operational forensic standards. No original algorithm, peer-reviewed publication or independent student implementation is claimed.

Codex/AI implemented and executed the pipeline and assisted with interpretation and drafting under user direction. Any CV statement must reflect the student's actual review, understanding and contribution; these artifacts do not verify those personal contributions.

## References

1. German, B. (1987). *Glass Identification* [Dataset]. UCI Machine Learning Repository. [DOI:10.24432/C5WW2P](https://doi.org/10.24432/C5WW2P). [Official page](https://archive.ics.uci.edu/dataset/42/glass+identification), accessed 22 October 2026.
2. scikit-learn developers. *Common pitfalls and recommended practices*, data leakage and pipelines. [Official documentation](https://scikit-learn.org/stable/common_pitfalls.html).
3. scikit-learn developers. *Probability calibration*. [Official documentation](https://scikit-learn.org/stable/modules/calibration.html).
4. El-Yaniv, R., & Wiener, Y. (2010). On the foundations of noise-free selective classification. *Journal of Machine Learning Research*, 11(53), 1605–1641. [Publisher abstract and paper](https://jmlr.org/papers/v11/el-yaniv10a.html). Cited for the reject-option/risk–coverage framing, not as a guarantee for this empirical procedure.

