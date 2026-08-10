import unittest

import pandas as pd

from scripts.audit_homology import summarize_hits


class HomologyAuditTests(unittest.TestCase):
    def test_threshold_summary_counts_unique_queries(self) -> None:
        queries = pd.DataFrame(
            {"protein_id": ["q1", "q2", "q3"], "split": ["validation"] * 3}
        )
        hits = pd.DataFrame(
            {
                "query_id": ["q1", "q1", "q2"],
                "percent_identity": [95.0, 40.0, 55.0],
                "shorter_sequence_coverage": [0.90, 0.95, 0.70],
            }
        )
        summary = summarize_hits(hits, queries)
        self.assertEqual(summary["queries_with_any_blast_hit"], 2)
        self.assertEqual(
            summary["threshold_counts"]["identity_at_least_30"]["queries"], 1
        )
        self.assertEqual(
            summary["threshold_counts"]["identity_at_least_90"]["queries"], 1
        )


if __name__ == "__main__":
    unittest.main()
