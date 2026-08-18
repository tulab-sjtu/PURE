import importlib.util
import os
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd


SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Script",
                      "PURE_CatBoost_SHAP_v3.py")
SPEC = importlib.util.spec_from_file_location("pure_v3", SCRIPT)
PURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PURE)


class ReviewRegressionTests(unittest.TestCase):
    def test_average_ranks_and_top_k_boundary_ties(self):
        ranked = PURE.stable_rank_frame(
            ["b", "a", "c", "d"], [10.0, 10.0, 10.0, 1.0], [0, 0, 0, 0])
        ranked = PURE.add_top_membership(ranked, [1, 2, 3])
        self.assertEqual(ranked["rank"].tolist(), [2.0, 2.0, 2.0, 4.0])
        self.assertEqual(ranked["top1"].tolist(), [1, 1, 1, 0])
        self.assertEqual(ranked["top2"].tolist(), [1, 1, 1, 0])
        self.assertEqual(ranked["top3"].tolist(), [1, 1, 1, 0])

    def test_summary_uses_empirical_interval_names_with_compatibility_aliases(self):
        run = pd.DataFrame({
            "entity_id": ["x", "x"], "mean_abs": [1.0, 3.0],
            "rank": [1.0, 2.0], "top1": [1, 0]})
        summary = PURE.summarize_rank_sources(run, pd.DataFrame(), [1])
        self.assertIn("run_mean_abs_interval_low", summary.columns)
        self.assertIn("run_mean_abs_interval_high", summary.columns)
        self.assertIn("run_mean_abs_ci_low", summary.columns)

    def test_split_defaults_alias_and_none_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            feature = os.path.join(directory, "features.csv")
            labels = os.path.join(directory, "labels.csv")
            pd.DataFrame({"f": [1]}, index=["g"]).to_csv(feature)
            pd.DataFrame({"label": ["a"]}, index=["g"]).to_csv(labels)
            common = ["--out_prefix", os.path.join(directory, "out"),
                      "--TF_features", feature, "--DEGs", labels]
            args = PURE.parse_args(common + ["--auto_class_weights", "None"])
            self.assertEqual(args.performance_splits, 5)
            self.assertEqual(args.stability_splits, 10)
            self.assertIsNone(args.n_splits)
            self.assertIsNone(args.auto_class_weights)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                alias = PURE.parse_args(common + ["--n_splits", "3"])
            self.assertEqual((alias.performance_splits, alias.stability_splits), (3, 3))
            self.assertTrue(any(issubclass(item.category, DeprecationWarning)
                                for item in caught))

    def test_duplicate_input_index_and_filename_sanitization(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = os.path.join(directory, "duplicate.csv")
            pd.DataFrame({"f": [1, 2]}, index=["g", "g"]).to_csv(duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                PURE.load_data(duplicate, "/regulons")
        self.assertEqual(PURE.sanitize_filename_component("safe-label.1"),
                         "safe-label.1")
        self.assertNotIn("/", PURE.sanitize_filename_component("case/one"))

    def test_pairwise_agreement_uses_tie_inclusive_membership(self):
        a = PURE.add_top_membership(PURE.stable_rank_frame(
            ["a", "b", "c"], [2, 2, 1], [0, 0, 0]), [1])
        b = PURE.add_top_membership(PURE.stable_rank_frame(
            ["a", "b", "c"], [2, 1, 2], [0, 0, 0]), [1])
        agreement = PURE.pairwise_agreement({1: a, 2: b}, [1], "TF", "raw")
        # Sets are {a,b} and {a,c}; tie-inclusive Jaccard is 1/3.
        self.assertAlmostEqual(agreement.loc[0, "top1_jaccard"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
