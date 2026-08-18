import importlib.util
import math
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "Script" / "PURE_Data_Process_v3.py"
SPEC = importlib.util.spec_from_file_location("pure_data_process_v3", MODULE_PATH)
PURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PURE)


class HomologyVotingTests(unittest.TestCase):
    def _files(self, alignment, chip):
        tempdir = tempfile.TemporaryDirectory()
        root = Path(tempdir.name)
        alignment_path = root / "alignment.tsv"
        chip_path = root / "chip.tsv"
        alignment_path.write_text(alignment)
        chip_path.write_text(chip)
        self.addCleanup(tempdir.cleanup)
        return str(alignment_path), str(chip_path)

    def test_votes_are_pooled_and_unique_across_reference_species(self):
        alignment, chip = self._files(
            "T1gene\tA1gene\t80\n"
            "T1gene\tB1gene\t70\n"
            "T1tf\tA1tf\t90\n"
            "T1tf\tB1tf\t85\n",
            # A1tf supports the target (duplicated); B1tf has ChIP data but does
            # not support B1gene. Thus the pooled vote is exactly 1/2.
            "A1tf\tA1gene\t10\n"
            "A1tf\tA1gene\t50\n"
            "B1tf\tB1other\t99\n",
        )
        groups = [["T1tf", "A1tf", "B1tf"]]

        rejected = PURE.process_homology_voting_chunk(
            ["T1gene"], groups, alignment, chip, "T", vote_threshold=0.75
        )
        self.assertEqual(rejected, [])

        accepted = PURE.process_homology_voting_chunk(
            ["T1gene"], groups, alignment, chip, "T", vote_threshold=0.5
        )
        self.assertEqual(accepted, [("T1tf", "T1gene", 50.0, None)])

    def test_weighted_mean_admits_only_complete_direct_alignment_paths(self):
        alignment, chip = self._files(
            "T1gene\tA1gene\t64\n"
            # T1tf-A1tf is deliberately absent: its OrthoGroup relation is transitive.
            "T1gene\tB1gene\t25\n"
            "B1gene\tT1gene\t20\n"
            "T1tf\tB1tf\t100\n"
            "B1tf\tT1tf\t90\n",
            "A1tf\tA1gene\t80\n"
            "B1tf\tB1gene\t40\n"
            "B1tf\tB1gene\t20\n",
        )
        groups = [["T1tf", "A1tf", "B1tf"]]
        observations = PURE.process_homology_voting_chunk(
            ["T1gene"], groups, alignment, chip, "T",
            vote_threshold=1.0, signal_aggregation="weighted_mean",
        )

        # Only the B path has both direct scores; sqrt(25 * 100) == 50.
        self.assertEqual(observations, [("T1tf", "T1gene", 40.0, 50.0)])
        self.assertEqual(PURE._aggregate_signal_observations(observations=[(40.0, 50.0)], method="weighted_mean"), 40.0)

    def test_weighted_aggregation_uses_geometric_path_weights(self):
        result = PURE._aggregate_signal_observations(
            [(10.0, math.sqrt(25.0 * 64.0)), (30.0, math.sqrt(100.0 * 36.0))],
            "weighted_mean",
        )
        self.assertAlmostEqual(result, 22.0)


if __name__ == "__main__":
    unittest.main()
