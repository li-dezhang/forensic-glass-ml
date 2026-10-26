"""Real-data/output integration checks; small synthetic fixtures test metric edge cases only."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from src.run import FEATURES, LABELS, TARGET, abstention, metric_values, models, validate_data, split_data

class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = pd.read_csv(ROOT / 'data/glass.csv')
        cls.frame, cls.quality = validate_data(cls.original)
        cls.predictions = pd.read_csv(ROOT / 'results/predictions.csv', float_precision='round_trip')

    def test_real_data_dimensions_and_labels(self):
        self.assertEqual(self.original[FEATURES].shape, (214, 9))
        self.assertEqual(set(self.original[TARGET]), set(LABELS))
        self.assertFalse(self.original[FEATURES].isna().any().any())

    def test_id_excluded_and_numeric_features(self):
        self.assertNotIn('Id_number', FEATURES)
        self.assertNotIn(TARGET, FEATURES)
        self.assertTrue(all(pd.api.types.is_numeric_dtype(self.frame[c]) for c in FEATURES))

    def test_duplicate_removal_and_disjoint_split(self):
        self.assertEqual(len(self.frame), 213)
        self.assertEqual(self.quality['duplicate_removed_ids'], [40])
        train, test = split_data(self.frame)
        self.assertFalse(set(train) & set(test))
        a = set(map(tuple, self.frame.iloc[train][FEATURES].to_numpy()))
        b = set(map(tuple, self.frame.iloc[test][FEATURES].to_numpy()))
        self.assertFalse(a & b)

    def test_training_only_cv_folds(self):
        split = pd.read_csv(ROOT / 'results/split.csv')
        self.assertTrue((split.loc[split.split == 'test', 'cv_validation_fold'] == -1).all())
        self.assertEqual(set(split.loc[split.split == 'train', 'cv_validation_fold']), set(range(5)))
        for fold in range(5):
            self.assertEqual(set(split.loc[split.cv_validation_fold == fold, TARGET]), set(LABELS))

    def test_scalers_are_inside_pipelines(self):
        for name in ['Logistic Regression', 'kNN']:
            self.assertIsInstance(models()[name], Pipeline)
            self.assertEqual(models()[name].steps[0][0], 'standardscaler')

    def test_model_outputs_and_probability_mapping(self):
        for name, p in self.predictions.groupby('model'):
            self.assertEqual(len(p), 54)
            self.assertEqual(p.sample_id.nunique(), 54)
            self.assertTrue(set(p.predicted_label) <= set(LABELS))
            probabilities = p[[f'probability_{c}' for c in LABELS]].to_numpy()
            self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
            np.testing.assert_allclose(probabilities.max(axis=1), p.max_probability, atol=1e-12)
            np.testing.assert_array_equal(np.asarray(LABELS)[probabilities.argmax(axis=1)], p.predicted_label)

    def test_selection_uses_training_cv(self):
        folds = pd.read_csv(ROOT / 'results/cv_folds.csv')
        selected = json.loads((ROOT / 'results/selection.json').read_text())['selected_model']
        self.assertEqual(selected, folds.groupby('model').macro_f1.mean().idxmax())

    def test_saved_metrics_match_predictions(self):
        metrics = pd.read_csv(ROOT / 'results/metrics.csv')
        for row in metrics.itertuples():
            p = self.predictions[self.predictions.model == row.model]
            actual = metric_values(p.true_label, p.predicted_label)
            for name, value in actual.items():
                self.assertAlmostEqual(value, getattr(row, name), places=13)

    def test_confusion_matrix_matches_predictions(self):
        selected = json.loads((ROOT / 'results/selection.json').read_text())['selected_model']
        p = self.predictions[self.predictions.model == selected]
        actual = confusion_matrix(p.true_label, p.predicted_label, labels=LABELS)
        np.testing.assert_array_equal(actual, pd.read_csv(ROOT / 'results/confusion_matrix.csv', index_col=0))

    def test_abstention_boundary_and_empty_denominator(self):
        # Synthetic edge-case fixture, never included in empirical results.
        r = abstention([1, 2], [1, 1], [.5, .49], .5)
        self.assertEqual(r['accepted_n'], 1)
        self.assertEqual(r['selective_accuracy'], 1)
        self.assertEqual(r['coverage'], .5)
        r = abstention([1, 2], [1, 1], [.5, .49], .8)
        self.assertTrue(np.isnan(r['selective_accuracy']))
        self.assertEqual(r['review_required_count'], 2)

    def test_confidence_outputs_and_coverage(self):
        p = pd.read_csv(ROOT / 'results/confidence_predictions.csv', float_precision='round_trip')
        s = pd.read_csv(ROOT / 'results/confidence_metrics.csv')
        self.assertTrue((s.coverage.diff().dropna() <= 0).all())
        for row in s.itertuples():
            actual = abstention(p.true_label, p.predicted_label, p.max_probability, row.threshold)
            for name, value in actual.items():
                self.assertAlmostEqual(value, getattr(row, name), places=13)
            accepted = p.max_probability >= row.threshold
            self.assertEqual(int((p[f'decision_{row.threshold:.2f}'] == 'REVIEW_REQUIRED').sum()), int((~accepted).sum()))

    def test_required_outputs(self):
        for name in ['metrics.csv', 'predictions.csv', 'confusion_matrix.png', 'model_comparison.png',
                     'confidence_tradeoff.png', 'class_distribution.png', 'per_class_recall.csv', 'errors.csv']:
            p = ROOT / 'results' / name
            self.assertTrue(p.is_file(), name)
            self.assertGreater(p.stat().st_size, 20)
            if p.suffix == '.png':
                self.assertEqual(p.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')

if __name__ == '__main__':
    unittest.main()
