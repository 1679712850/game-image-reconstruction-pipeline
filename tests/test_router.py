"""QA retries cannot continue beyond their budget."""
import unittest

from agent.routers import route_after_qa


class RouterTests(unittest.TestCase):
    def test_pass_continues(self) -> None:
        self.assertEqual(route_after_qa({"failed_objects": []}), "continue")

    def test_failures_retry_only_with_budget(self) -> None:
        self.assertEqual(route_after_qa({"failed_objects": ["tree_001"], "retry_count": 0, "max_retry": 1}), "retry")
        self.assertEqual(route_after_qa({"failed_objects": ["tree_001"], "retry_count": 1, "max_retry": 1}), "continue")

    def test_zero_budget_never_retries(self) -> None:
        self.assertEqual(route_after_qa({"failed_objects": ["x"], "max_retry": 0}), "continue")

    def test_persistent_failures_terminate(self) -> None:
        state = {"failed_objects": ["x"], "retry_count": 0, "max_retry": 3}
        visits = 0
        while route_after_qa(state) == "retry":
            state["retry_count"] += 1
            visits += 1
            self.assertLessEqual(visits, 3)
        self.assertEqual(visits, 3)
