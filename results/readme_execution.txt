Verified cached official source hashes
              model  cv_macro_f1_mean  cv_macro_f1_std  accuracy  macro_f1  balanced_accuracy  test_n  selected_by_cv
     Dummy majority          0.087932         0.002841  0.351852  0.086758           0.166667      54           False
Logistic Regression          0.471704         0.042164  0.685185  0.670234           0.716374      54           False
                kNN          0.483893         0.116902  0.740741  0.608761           0.610746      54           False
      Random Forest          0.725478         0.053976  0.777778  0.758511           0.769128      54            True
Selected on training CV: Random Forest
 threshold  test_n  accepted_n  correct_accepted_n  coverage  selective_accuracy  review_required_count  correct_automated_fraction_all
       0.5      54          43                  38  0.796296            0.883721                     11                        0.703704
       0.6      54          33                  32  0.611111            0.969697                     21                        0.592593
       0.7      54          26                  25  0.481481            0.961538                     28                        0.462963
       0.8      54          18                  17  0.333333            0.944444                     36                        0.314815
Verified cached official source hashes
 threshold  test_n  accepted_n  correct_accepted_n  coverage  selective_accuracy  review_required_count  correct_automated_fraction_all
       0.5      54          43                  38  0.796296            0.883721                     11                        0.703704
       0.6      54          33                  32  0.611111            0.969697                     21                        0.592593
       0.7      54          26                  25  0.481481            0.961538                     28                        0.462963
       0.8      54          18                  17  0.333333            0.944444                     36                        0.314815
{
  "verified_at_utc": "2026-10-22T10:26:12.059402+00:00",
  "status": "PASS",
  "model_metrics_independently_checked": 4,
  "confidence_thresholds_checked": 4,
  "source_hashes_verified": true,
  "regenerated_csv_png_identical": 17,
  "full_repeat": {
    "other_run": "results",
    "identical_csv_png_files": 17
  },
  "verification_source_sha256": "c1888bc717df2d8776ffd2840c5e96844254a8b5eab982aa48912ab84792676d",
  "current_analysis_source_sha256": "3a0ac1e42efba7a4a2dd8305b25c707fc548535387a3a03b5601d05608d1305c",
  "output_sha256": {
    "class_distribution.csv": "475c4cee162a0759eef2634db35d93e82d05d725c37dfcda68f988de9420f458",
    "class_distribution.png": "a44fe7cc92c4a0a51376f0dc034631050c0fa9e99caa5b56b6cdebb29fae2ea8",
    "confidence_by_class.csv": "75b243c3c0eba3c92917afa6528b99e1b4b32b8496fa0c20daa7b0afe610067a",
    "confidence_metrics.csv": "f856123b58f407b1d331a354f2f0a52e0d8ccf83e33993b6117f4049423c7aa0",
    "confidence_predictions.csv": "6d31302556a3dff4c1eb354cff5f18d51444c24626088a64cd21056f25236cf2",
    "confidence_tradeoff.png": "6f237db1170fa17522220058fe101989ff8a1cb4b87c81dd5c4af6b59e06cd16",
    "confusion_matrix.csv": "7c00fb49972b97d85bf07e4558bd67d09d72e17e7507ddcca09134118ee742cc",
    "confusion_matrix.png": "04ba5b86877ca2f0ac39147851e124d77aeb3b85645aeae939838da0c112a73f",
    "cv_folds.csv": "3aac3e1822b9889801f5d3d54d9ad1ce94aa48bed754a5f2b592b2b07b0b89fb",
    "error_pairs.csv": "36d0614a13c0497d4233f4b9031f08ff433b1c8a0a9c901d8689b645ff1ed9e5",
    "errors.csv": "aebc53ec4538e7eaaab69607a8e4b9fea75ec20bdbbdaefcab1536b80450b6fe",
    "feature_summary.csv": "6c81023319ca9534ccc5605f880412e854d0326e32a51d7eebd6ebdb442d58fa",
    "metrics.csv": "34c48f440cb5c517458894c6f687ac2760f43bbc91cf2c71234d0067746dd384",
    "model_comparison.png": "6467332c3e079f2d4b620e44afe1dac8159b3d3aacb61087d9c060f35baca52a",
    "per_class_recall.csv": "8ef371ce85b95763868af8e6a5a28c0ae9d9330a1178a0a56c7af1895cb4de1f",
    "predictions.csv": "ac4a5919396b18d463040419258ad6286a237c3085171454d46b9ea2df391f3b",
    "split.csv": "306596241ec605b55d15326de6de66a63a6475c800bb63969a5d2c8298a9e7cb"
  }
}
