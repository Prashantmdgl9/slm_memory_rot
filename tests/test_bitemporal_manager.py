"""Tests for FrameworkBitemporalManager retrieval fallback policy."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.memory.managers.framework_bitemporal import FrameworkBitemporalManager


class RetrievalFallbackTest(unittest.TestCase):

    def test_recovers_soft_deleted_fact_when_no_valid_entry_answers(self):
        m = FrameworkBitemporalManager("unused-model")
        eid = m.store.add("favorite_color: blue")
        m.store.delete(eid)  # false delete — only copy of this fact

        valid, recovered = m._retrieve("What is the user's favorite color?")

        self.assertEqual(valid, [])
        self.assertTrue(any("blue" in h["content"] for h in recovered),
                        "soft-deleted fact should be recovered as fallback")

    def test_does_not_recover_when_valid_entry_answers(self):
        m = FrameworkBitemporalManager("unused-model")
        m.store.add("favorite_color: green")      # valid, answers the query
        eid = m.store.add("pet: oldcat")
        m.store.delete(eid)                         # unrelated soft-deleted fact

        valid, recovered = m._retrieve("What is the user's favorite color?")

        self.assertTrue(any("green" in h["content"] for h in valid))
        self.assertEqual(recovered, [],
                         "must not resurrect unrelated deleted facts")


if __name__ == "__main__":
    unittest.main()
