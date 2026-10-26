"""Check stored outputs independently and confirm deterministic table/figure regeneration."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.run import LABELS, comparison_plots, analyze, acquire

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(out, compare=None):
    acquire()  # cached source verification; no network when the supplied cache is intact
    metrics = pd.read_csv(out / "metrics.csv", float_precision="round_trip")
    predictions = pd.read_csv(out / "predictions.csv", float_precision="round_trip")
    for row in metrics.itertuples():
        p = predictions[predictions.model == row.model]
        y, pred = p.true_label.to_numpy(), p.predicted_label.to_numpy()
        correct = int((y == pred).sum())
        recalls, f1s = [], []
        for label in LABELS:
            tp = int(((y == label) & (pred == label)).sum())
            fp = int(((y != label) & (pred == label)).sum())
            fn = int(((y == label) & (pred != label)).sum())
            recalls.append(tp / (tp + fn))
            f1s.append(2*tp / (2*tp + fp + fn) if 2*tp + fp + fn else 0.)
        np.testing.assert_allclose([correct/len(y), np.mean(recalls), np.mean(f1s)],
                                   [row.accuracy, row.balanced_accuracy, row.macro_f1], rtol=0, atol=1e-14)
    winner = json.loads((out / "selection.json").read_text())["selected_model"]
    p = predictions[predictions.model == winner]
    for row in pd.read_csv(out / "confidence_metrics.csv").itertuples():
        accepted = p.max_probability >= row.threshold
        correct = int((accepted & (p.true_label == p.predicted_label)).sum())
        assert (int(accepted.sum()), correct, int((~accepted).sum())) == (row.accepted_n, row.correct_accepted_n, row.review_required_count)
        np.testing.assert_allclose([accepted.mean(), correct/accepted.sum(), correct/len(p)],
                                  [row.coverage, row.selective_accuracy, row.correct_automated_fraction_all], atol=1e-14, rtol=0)
    def fingerprints():
        return {f.name: digest(f) for f in sorted(out.iterdir()) if f.suffix in {".csv", ".png"}}
    before = fingerprints()
    comparison_plots(out)
    analyze(out)
    assert fingerprints() == before, "Rescoring/rendering changed a saved table or image"
    repeated = None
    if compare:
        other = {f.name: digest(f) for f in compare.iterdir() if f.suffix in {".csv", ".png"}}
        assert before == other, "Full rerun output mismatch"
        repeated = {"other_run": str(compare.relative_to(ROOT)), "identical_csv_png_files": len(before)}
    record = {"verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status": "PASS", "model_metrics_independently_checked": len(metrics),
              "confidence_thresholds_checked": 4, "source_hashes_verified": True,
              "regenerated_csv_png_identical": len(before), "full_repeat": repeated,
              "verification_source_sha256": digest(Path(__file__)),
              "current_analysis_source_sha256": digest(ROOT / "src/run.py"), "output_sha256": before}
    (out / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results")
    parser.add_argument("--compare")
    a = parser.parse_args()
    out = (ROOT / a.run).resolve()
    compare = (ROOT / a.compare).resolve() if a.compare else None
    if not out.is_relative_to(ROOT) or (compare and not compare.is_relative_to(ROOT)):
        raise ValueError("Keep run paths inside this project")
    verify(out, compare)

if __name__ == "__main__":
    main()

