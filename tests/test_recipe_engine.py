"""Tests for YAML-controlled execution order and structured control flow."""

import unittest

from oakforge.core.context import GenerationContext
from oakforge.core.recipe_engine import RecipeEngine
from oakforge.tools.registry import ToolRegistry
import numpy as np


class RecipeEngineTests(unittest.TestCase):
    """Verify that recipes, rather than Python growth code, control tool execution."""

    def setUp(self):
        self.values = {"numbers": [1, 2, 3], "collected": [], "counter": 0}
        self.context = GenerationContext(self.values, np.random.default_rng(4))
        self.registry = ToolRegistry()
        self.registry.register("test.record", lambda context, arguments: arguments["value"])
        self.engine = RecipeEngine(self.context, self.registry)

    def test_for_if_and_tool_order_are_recipe_driven(self):
        self.engine.execute([
            {"if": {
                "condition": {"eq": ["$counter", 0]},
                "then": [{"set": {"selected": "then branch"}}],
                "else": [{"set": {"selected": "else branch"}}],
            }},
            {"for": {
                "each": "$numbers",
                "as": "number_item",
                "do": [{"call": {
                    "tool": "test.record",
                    "args": {"value": "$number_item"},
                    "append_to": "collected",
                }}],
            }},
        ])
        self.assertEqual(self.values["selected"], "then branch")
        self.assertEqual(self.values["collected"], [1, 2, 3])

        self.engine.execute([{"if": {
            "condition": False,
            "then": [{"set": {"selected": "unexpected then"}}],
            "else": [{"set": {"selected": "else branch"}}],
        }}])
        self.assertEqual(self.values["selected"], "else branch")

    def test_while_and_do_while_have_expected_iteration_semantics(self):
        self.engine.execute([
            {"while": {
                "condition": {"lt": ["$counter", 3]},
                "do": [{"set": {"counter": {"add": ["$counter", 1]}}}],
            }},
            {"set": {"do_count": 0}},
            {"do_while": {
                "condition": {"lt": ["$do_count", 1]},
                "do": [{"set": {"do_count": {"add": ["$do_count", 1]}}}],
            }},
        ])
        self.assertEqual(self.values["counter"], 3)
        self.assertEqual(self.values["do_count"], 1)

    def test_repeat_and_safe_range_expression(self):
        self.engine.execute([
            {"repeat": {
                "count": 3,
                "as": "step",
                "do": [{"call": {
                    "tool": "test.record",
                    "args": {"value": "$step"},
                    "append_to": "collected",
                }}],
            }},
            {"for": {
                "each": {"range": [3, 5]},
                "as": "step",
                "do": [{"call": {
                    "tool": "test.record",
                    "args": {"value": "$step"},
                    "append_to": "collected",
                }}],
            }},
        ])
        self.assertEqual(self.values["collected"], [0, 1, 2, 3, 4])

    def test_loop_guard_stops_an_unbounded_recipe(self):
        with self.assertRaisesRegex(Exception, "max_iterations"):
            self.engine.execute([{"while": {
                "condition": True,
                "max_iterations": 2,
                "do": [{"set": {"counter": {"add": ["$counter", 1]}}}],
            }}])


if __name__ == "__main__":
    unittest.main()
