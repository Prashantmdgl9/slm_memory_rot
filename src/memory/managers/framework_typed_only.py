"""
Framework ablation: typed slots only, deletion signals disabled.

This isolates the contribution of typed slots alone, without rule-based
deletion signal detection. Compared against framework.py (full), the delta
shows how much the deletion-signal mechanism contributes independently.
"""
from src.memory.managers.framework import FrameworkManager


class FrameworkTypedOnlyManager(FrameworkManager):

    def _apply_fact(self, key: str, value: str, date: str):
        key   = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if not key or not value:
            return

        # Typed slots only — no deletion signal detection
        if key in self._slots:
            self.store.update(self._slots[key]["eid"], f"{key}: {value}")
            self._slots[key]["value"] = value
        else:
            eid = self.store.add(f"{key}: {value}", created_at=date)
            self._slots[key] = {"eid": eid, "value": value}
