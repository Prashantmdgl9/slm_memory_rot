"""Tests for BitemporalStore — non-destructive, recoverable deletes."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.memory.store import BitemporalStore


class SoftDeleteTest(unittest.TestCase):

    def test_soft_delete_excludes_entry_from_effective_store(self):
        store = BitemporalStore()
        eid = store.add("city: Austin", created_at="2025-01-01")
        self.assertEqual(len(store), 1)

        store.delete(eid)

        # the soft-deleted fact is gone from the *effective* (valid) store
        self.assertEqual(len(store), 0)
        self.assertEqual(store.all_entries(), [])
        self.assertEqual(store.search("Austin"), [])

    def test_soft_deleted_entry_is_recoverable(self):
        store = BitemporalStore()
        eid = store.add("city: Austin", created_at="2025-01-01")
        store.delete(eid)

        recoverable = store.recoverable()
        self.assertEqual(len(recoverable), 1)
        self.assertIn("Austin", recoverable[0][1]["content"])

        hits = store.search("Austin", include_invalid=True)
        self.assertTrue(any("Austin" in h["content"] for h in hits))

    def test_search_ranks_valid_strictly_before_invalid(self):
        store = BitemporalStore()
        store.add("city: Boston")
        eid = store.add("city: Austin")
        store.delete(eid)  # Austin invalidated, Boston still valid

        hits = store.search("city", k=10, include_invalid=True)
        contents = [h["content"] for h in hits]
        self.assertIn("city: Boston", contents)
        self.assertIn("city: Austin", contents)
        self.assertLess(contents.index("city: Boston"),
                        contents.index("city: Austin"))


if __name__ == "__main__":
    unittest.main()
