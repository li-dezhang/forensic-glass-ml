"""Run real UCI glass experiments. No synthetic research data or test-set tuning."""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import urllib.request
import warnings

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
LABELS = [1, 2, 3, 5, 6, 7]
TARGET = "Type_of_glass"
SEED = 42
THRESHOLDS = [0.50, 0.60, 0.70, 0.80]
CLASS_NAMES = {
    1: "Building windows: float", 2: "Building windows: non-float",
    3: "Vehicle windows: float", 5: "Containers", 6: "Tableware", 7: "Headlamps",
}
DATA_URL = "https://archive.ics.uci.edu/static/public/42/data.csv"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/42/glass+identification"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

def acquire():
    """Prefer official ucimlrepo; fall back only to the official published CSV."""
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    manifest = folder / "manifest.json"
    if manifest.exists():
        record = json.loads(manifest.read_text())
        for name, entry in record["files"].items():
            if sha(folder / name) != entry["sha256"]:
                raise ValueError(f"Source file changed: {name}")
        print("Verified cached official source hashes", flush=True)
        return pd.read_csv(folder / "glass.csv")
    failure = None
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=42)
        frame = dataset.data.original
        dataset.variables.to_csv(folder / "dictionary.csv", index=False)
        write_json(folder / "uci_metadata.json", dataset.metadata)
        url = dataset.metadata.data_url
        method = "ucimlrepo.fetch_ucirepo(id=42)"
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        print("ucimlrepo acquisition failed:", failure, flush=True)
        print("Trying official UCI CSV:", DATA_URL, flush=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            payload = response.read(1_000_001)
        if len(payload) > 1_000_000:
            raise ValueError("Official CSV exceeds bounded download limit")
        frame = pd.read_csv(io.BytesIO(payload))
        (folder / "official_data.csv").write_bytes(payload)
        method, url = "official UCI CSV fallback", DATA_URL
        dictionary = [{"name": c, "role": "ID" if c == "Id_number" else "Target" if c == TARGET else "Feature",
                       "source": SOURCE_URL} for c in ["Id_number"] + FEATURES + [TARGET]]
        pd.DataFrame(dictionary).to_csv(folder / "dictionary.csv", index=False)
        write_json(folder / "uci_metadata.json", {"source": SOURCE_URL, "data_url": url, "fallback": True})
    frame.to_csv(folder / "glass.csv", index=False)
    names = ["glass.csv", "dictionary.csv", "uci_metadata.json"]
    if failure:
        names.append("official_data.csv")
    record = {"source": SOURCE_URL, "citation": "German, B. (1987). Glass Identification. DOI:10.24432/C5WW2P",
              "licence": "CC BY 4.0", "download_method": method, "data_url": url,
              "downloaded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "primary_download_failure": failure,
              "files": {n: {"bytes": (folder/n).stat().st_size, "sha256": sha(folder/n)} for n in names}}
    write_json(manifest, record)
    return frame

def validate_data(frame):
    expected = ["Id_number"] + FEATURES + [TARGET]
    if list(frame.columns) != expected or frame.shape != (214, 11):
        raise ValueError("Unexpected UCI schema or row count")
    if frame.isna().any().any() or not np.isfinite(frame[FEATURES].to_numpy()).all():
        raise ValueError("Missing/nonfinite values: do not silently impute this dataset")
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in FEATURES):
        raise ValueError("Features must be numeric")
    if set(frame[TARGET]) != set(LABELS) or frame.Id_number.nunique() != len(frame):
        raise ValueError("Unexpected labels or repeated IDs")
    conflicts = frame.groupby(FEATURES, dropna=False)[TARGET].nunique().gt(1).sum()
    if conflicts:
        raise ValueError("Identical measurements have conflicting labels; protocol requires review")
    duplicates = frame.duplicated(FEATURES + [TARGET], keep="first")
    quality = {"raw_rows": len(frame), "n_features": len(FEATURES),
               "missing_cells": int(frame.isna().sum().sum()), "duplicate_full_rows": int(frame.duplicated().sum()),
               "duplicate_feature_rows": int(frame.duplicated(FEATURES).sum()),
               "duplicate_features_and_label": int(duplicates.sum()),
               "duplicate_removed_ids": frame.loc[duplicates, "Id_number"].astype(int).tolist(),
               "conflicting_feature_groups": int(conflicts),
               "feature_dtypes": {c: str(frame[c].dtype) for c in FEATURES},
               "modelling_rows": int((~duplicates).sum())}
    return frame.loc[~duplicates].reset_index(drop=True), quality

def split_data(frame):
    train, test = train_test_split(np.arange(len(frame)), test_size=.25,
                                   stratify=frame[TARGET], random_state=SEED)
    return train, test

def models():
    return {
        "Dummy majority": DummyClassifier(strategy="most_frequent", random_state=SEED),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(
            C=1., solver="lbfgs", max_iter=3000, random_state=SEED)),
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=5, weights="uniform", p=2)),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=None,
            min_samples_leaf=1, max_features="sqrt", class_weight=None, random_state=SEED, n_jobs=1),
    }

def metric_values(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred))}

def abstention(y, pred, confidence, threshold):
    y, pred, confidence = map(np.asarray, [y, pred, confidence])
    accepted = confidence >= threshold
    count = int(accepted.sum())
    correct = int(((y == pred) & accepted).sum())
    return {"threshold": threshold, "test_n": len(y), "accepted_n": count,
            "correct_accepted_n": correct, "coverage": count/len(y),
            "selective_accuracy": correct/count if count else np.nan,
            "review_required_count": len(y)-count,
            "correct_automated_fraction_all": correct/len(y)}

def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", metadata={"Software": "forensic-glass-ml"})
    plt.close(fig)

def comparison_plots(out):
    metrics = pd.read_csv(out / "metrics.csv")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(metrics))
    for offset, col, label, color in [(-.25, "cv_macro_f1_mean", "Training CV macro-F1", "#60718e"),
                                      (0, "accuracy", "Test accuracy", "#218c83"),
                                      (.25, "macro_f1", "Test macro-F1", "#ca6d36")]:
        ax.bar(x+offset, metrics[col], .25, label=label, color=color)
    ax.set_xticks(x, metrics.model)
    ax.set(ylim=(0, 1.08), ylabel="Score", title="Fixed candidates; winner selected using training CV only")
    ax.legend(fontsize=9)
    save_figure(fig, out / "model_comparison.png")
    matrix = pd.read_csv(out / "confusion_matrix.csv", index_col=0).to_numpy()
    selection = json.loads((out / "selection.json").read_text())
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(6), LABELS); ax.set_yticks(range(6), LABELS)
    ax.set(xlabel="Predicted category", ylabel="True category",
           title=f"{selection['selected_model']} — held-out test counts")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, int(matrix[i,j]), ha="center", va="center",
                    color="white" if matrix[i,j] > matrix.max()/2 else "black")
    save_figure(fig, out / "confusion_matrix.png")
    counts = pd.read_csv(out / "class_distribution.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    for label in ["train", "test"]:
        g = counts[counts.split == label].set_index("class_code").loc[LABELS]
        if label == "train":
            base = g["count"].to_numpy()
            ax.bar([str(c) for c in LABELS], base, label="Train", color="#218c83")
        else:
            ax.bar([str(c) for c in LABELS], g["count"], bottom=base, label="Test", color="#ca6d36")
    ax.set(xlabel="Glass category code", ylabel="Deduplicated samples", title="Class imbalance and fixed stratified split")
    ax.legend(); save_figure(fig, out / "class_distribution.png")

def run_models(out):
    if (out / "metrics.csv").exists():
        raise FileExistsError("Use --out with a fresh directory; existing experiments are preserved")
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    original = acquire()
    frame, quality = validate_data(original)
    train, test = split_data(frame)
    X, y = frame[FEATURES], frame[TARGET].astype(int)
    if y.iloc[train].value_counts().min() < 5:
        raise ValueError("Insufficient training class count for 5-fold stratification")
    cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X.iloc[train], y.iloc[train]))
    partition = frame[["Id_number", TARGET]].copy()
    partition["split"] = "test"; partition.loc[train, "split"] = "train"
    partition["cv_validation_fold"] = -1
    for fold, (_, validation) in enumerate(cv):
        partition.loc[train[validation], "cv_validation_fold"] = fold
    partition.to_csv(out / "split.csv", index=False)
    quality.update({"train_n": len(train), "test_n": len(test), "labels": LABELS})
    write_json(out / "data_quality.json", quality)
    counts = []
    for split, subset in [("raw", original), ("deduplicated", frame), ("train", frame.iloc[train]), ("test", frame.iloc[test])]:
        for label in LABELS:
            counts.append({"split": split, "class_code": label, "class_name": CLASS_NAMES[label],
                           "count": int((subset[TARGET] == label).sum())})
    pd.DataFrame(counts).to_csv(out / "class_distribution.csv", index=False)
    frame[FEATURES].describe().to_csv(out / "feature_summary.csv")
    folds, summaries = [], []
    candidates = models()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for name, model in candidates.items():
            scores = cross_validate(model, X.iloc[train], y.iloc[train], cv=cv,
                scoring={"macro_f1": "f1_macro", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"},
                n_jobs=1, error_score="raise")
            for fold in range(5):
                folds.append({"model": name, "fold": fold,
                              **{m: float(scores["test_"+m][fold]) for m in ["macro_f1", "accuracy", "balanced_accuracy"]}})
            summaries.append({"model": name, "cv_macro_f1_mean": float(scores["test_macro_f1"].mean()),
                              "cv_macro_f1_std": float(scores["test_macro_f1"].std(ddof=1))})
    cv_table = pd.DataFrame(summaries)
    best = cv_table.loc[cv_table.cv_macro_f1_mean.idxmax(), "model"]
    # Save winner before fitting or inspecting any held-out predictions.
    write_json(out / "selection.json", {"selected_model": best, "rule": "maximum mean training-only 5-fold macro-F1",
               "selected_before_test_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "model_order_for_ties": list(candidates), "hyperparameters": {n: str(m) for n,m in candidates.items()}})
    pd.DataFrame(folds).to_csv(out / "cv_folds.csv", index=False)
    predictions, recalls, metrics = [], [], []
    for entry in summaries:
        name = entry["model"]
        fitted = clone(candidates[name]).fit(X.iloc[train], y.iloc[train])
        pred = fitted.predict(X.iloc[test]); probabilities = fitted.predict_proba(X.iloc[test])
        assert list(fitted.classes_) == LABELS
        metrics.append({**entry, **metric_values(y.iloc[test], pred), "test_n": len(test), "selected_by_cv": name == best})
        record = pd.DataFrame({"model": name, "sample_id": frame.Id_number.iloc[test].to_numpy(),
                               "true_label": y.iloc[test].to_numpy(), "predicted_label": pred,
                               "max_probability": probabilities.max(axis=1)})
        for j, label in enumerate(LABELS):
            record[f"probability_{label}"] = probabilities[:,j]
        predictions.append(record)
        per_class = recall_score(y.iloc[test], pred, labels=LABELS, average=None, zero_division=0)
        for label, recall in zip(LABELS, per_class):
            recalls.append({"model": name, "class_code": label, "class_name": CLASS_NAMES[label],
                            "test_support": int((y.iloc[test] == label).sum()), "recall": float(recall)})
        if name == best:
            matrix = confusion_matrix(y.iloc[test], pred, labels=LABELS)
            pd.DataFrame(matrix, index=LABELS, columns=LABELS).rename_axis("true_label").to_csv(out / "confusion_matrix.csv")
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(recalls).to_csv(out / "per_class_recall.csv", index=False)
    comparison_plots(out)
    versions = {n: importlib.metadata.version(n) for n in ["ucimlrepo", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib"]}
    write_json(out / "run.json", {"started_utc": started, "completed_models_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "platform": platform.platform(), "python": platform.python_version(), "dependencies": versions,
               "random_state": SEED, "test_size": .25, "cv_folds": 5, "training_rows": len(train), "test_rows": len(test),
               "selected_model": best, "data_sha256": sha(ROOT/"data/glass.csv"),
               "source_sha256": sha(__file__), "protocol_sha256": sha(ROOT/"EXPERIMENT_PROTOCOL.md")})
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print("Selected on training CV:", best, flush=True)

def analyze(out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    selection = json.loads((out / "selection.json").read_text())
    all_predictions = pd.read_csv(out / "predictions.csv", float_precision="round_trip")
    p = all_predictions[all_predictions.model == selection["selected_model"]].copy()
    errors = p[p.true_label != p.predicted_label]
    errors.to_csv(out / "errors.csv", index=False)
    errors.groupby(["true_label", "predicted_label"]).size().rename("count").reset_index().sort_values(
        ["count", "true_label", "predicted_label"], ascending=[False, True, True]).to_csv(out / "error_pairs.csv", index=False)
    rows, classes = [], []
    for threshold in THRESHOLDS:
        row = abstention(p.true_label, p.predicted_label, p.max_probability, threshold)
        rows.append(row)
        accepted = p.max_probability >= threshold
        p[f"decision_{threshold:.2f}"] = np.where(accepted, p.predicted_label.astype(str), "REVIEW_REQUIRED")
        for label in LABELS:
            keep = p.true_label == label
            classes.append({"threshold": threshold, "class_code": label,
                            **abstention(p.loc[keep,"true_label"], p.loc[keep,"predicted_label"],
                                         p.loc[keep,"max_probability"], threshold)})
    tradeoff = pd.DataFrame(rows)
    tradeoff.to_csv(out / "confidence_metrics.csv", index=False)
    pd.DataFrame(classes).to_csv(out / "confidence_by_class.csv", index=False)
    p.to_csv(out / "confidence_predictions.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tradeoff.coverage, tradeoff.selective_accuracy, "o-", color="#218c83", label="Fixed thresholds")
    full_accuracy = (p.true_label == p.predicted_label).mean()
    ax.scatter([1], [full_accuracy], marker="s", color="#60718e", label="No abstention")
    for row in tradeoff.itertuples():
        ax.annotate(f"{row.threshold:.2f} ({row.accepted_n}/{row.test_n})",
                    (row.coverage, row.selective_accuracy), xytext=(5, -24 if row.threshold == .7 else 10), textcoords="offset points", fontsize=9)
    ax.set(xlim=(0, 1.1), ylim=(0, 1.1), xlabel="Coverage (accepted / all test samples)",
           ylabel="Selective accuracy (correct / accepted)", title="Exploratory abstention — uncalibrated model probabilities")
    ax.legend(loc="lower left"); save_figure(fig, out / "confidence_tradeoff.png")
    print(tradeoff.to_string(index=False), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results", help="Project-relative directory; use a new directory for repeated training")
    parser.add_argument("--stage", choices=["models", "analysis", "all"], default="all")
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(ROOT) or out == ROOT:
        raise ValueError("Output must remain inside this project")
    if args.stage in ["models", "all"]:
        run_models(out)
    if args.stage in ["analysis", "all"]:
        analyze(out)

if __name__ == "__main__":
    main()

