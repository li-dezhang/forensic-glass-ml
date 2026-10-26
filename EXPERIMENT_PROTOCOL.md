# Fixed protocol — 2026-10-22 (Asia/Shanghai)

Recorded after source/data inspection and before model fitting/test evaluation.

- Original source: UCI Glass Identification, id 42, 214 rows, nine numeric features. Preserve all source rows. Remove exact feature+label duplicates before splitting, keeping the first source row. Detect conflicting labels for identical features and stop if present. Do not claim duplicates prove a common physical source; this is a conservative measurement-level leakage precaution.
- ID is retained for traceability only. Feature allowlist: RI, Na, Mg, Al, Si, K, Ca, Ba, Fe. Target codes: 1,2,3,5,6,7. No missing-value synthesis, resampling, augmentation or outlier removal.
- Stratified 75/25 train/test split with random_state=42; training only: StratifiedKFold(5, shuffle=True, random_state=42). Save sample IDs and CV fold assignment. No test-set tuning.
- Fixed candidates: DummyClassifier(strategy=most_frequent); StandardScaler→LogisticRegression(C=1, max_iter=3000, solver=lbfgs, random_state=42); StandardScaler→KNeighborsClassifier(n_neighbors=5, weights=uniform, p=2); RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=1, max_features=sqrt, class_weight=None, random_state=42, n_jobs=1). No parameter search.
- Select largest mean training CV macro-F1. Ties use the listed model order. Freeze selection before evaluating any test model. Fit all four on the training partition; report all test metrics without reselecting the winner on test performance.
- Accuracy over all test rows; macro-F1 over all six classes with zero_division=0; balanced accuracy and per-class recall; confusion matrix rows=true, columns=predicted.
- Best-model raw predict_proba maximum controls exploratory abstention. Thresholds fixed at .50,.60,.70,.80; accept scores >= threshold. Coverage=accepted/N; selective accuracy=correct accepted/accepted (undefined if zero); review count=N−accepted. Also report correct automated decisions/N and per-class coverage to expose discarded samples.
- Threshold results are descriptive test-set exploration, not threshold selection, calibration or a guarantee of safety. No likelihood ratios, source attribution, courtroom validation or independent-human-authorship claims.
- First run model comparison, then analyze stored test predictions. A full repeat and independent rescoring check deterministic outputs. No model persistence or deployment.
