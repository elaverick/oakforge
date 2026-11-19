"""Interpreter for the YAML growth graph and its control-flow actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from oakforge.core.context import GenerationContext
from oakforge.tools.registry import ToolRegistry


class RecipeError(RuntimeError):
    """Raised when a recipe contains an invalid action or expression."""


class RecipeEngine:
    """Execute tool calls and control flow described entirely by a recipe."""

    def __init__(self, context: GenerationContext, registry: ToolRegistry) -> None:
        self.context = context
        self.registry = registry

    def _get_path(self, path: str) -> Any:
        """Resolve a dotted variable path such as '$tree.parameters.height'."""
        path_parts = path.split(".")
        current_value: Any = self.context.values
        for path_part in path_parts:
            if isinstance(current_value, Mapping):
                if path_part not in current_value:
                    raise RecipeError(f"Variable reference '${path}' has no member '{path_part}'")
                current_value = current_value[path_part]
            elif isinstance(current_value, (list, tuple)) and path_part.isdigit():
                current_value = current_value[int(path_part)]
            elif hasattr(current_value, path_part):
                current_value = getattr(current_value, path_part)
            else:
                raise RecipeError(f"Variable reference '${path}' cannot resolve '{path_part}'")
        return current_value

    def _set_path(self, path: str, value: Any) -> None:
        """Assign a recipe variable, creating intermediate dictionaries as needed."""
        path_parts = path.split(".")
        current_value = self.context.values
        for path_part in path_parts[:-1]:
            if path_part not in current_value or not isinstance(current_value[path_part], dict):
                current_value[path_part] = {}
            current_value = current_value[path_part]
        current_value[path_parts[-1]] = value

    def evaluate(self, expression: Any) -> Any:
        """Evaluate recipe references and a small safe expression vocabulary."""
        if isinstance(expression, str) and expression.startswith("$"):
            return self._get_path(expression[1:])
        if isinstance(expression, list):
            return [self.evaluate(item) for item in expression]
        if isinstance(expression, tuple):
            return tuple(self.evaluate(item) for item in expression)
        if isinstance(expression, Mapping):
            if len(expression) == 1:
                operator, operands = next(iter(expression.items()))
                if operator in {"eq", "ne", "gt", "gte", "lt", "lte", "and", "or", "not",
                                "add", "subtract", "multiply", "divide", "modulo", "len", "range"}:
                    return self._evaluate_operator(operator, operands)
            return {key: self.evaluate(value) for key, value in expression.items()}
        return expression

    def _evaluate_operator(self, operator: str, operands: Any) -> Any:
        """Evaluate a supported expression without using Python eval or exec."""
        if operator == "len":
            return len(self.evaluate(operands))
        if operator == "range":
            values = self.evaluate(operands)
            return list(range(*[int(value) for value in values]))
        if operator == "not":
            return not bool(self.evaluate(operands))

        values = self.evaluate(operands)
        if not isinstance(values, (list, tuple)):
            values = [values]
        if operator == "and":
            return all(bool(value) for value in values)
        if operator == "or":
            return any(bool(value) for value in values)
        if len(values) < 2 and operator not in {"add", "multiply"}:
            raise RecipeError(f"Expression '{operator}' requires at least two operands")
        if operator == "eq":
            return values[0] == values[1]
        if operator == "ne":
            return values[0] != values[1]
        if operator == "gt":
            return values[0] > values[1]
        if operator == "gte":
            return values[0] >= values[1]
        if operator == "lt":
            return values[0] < values[1]
        if operator == "lte":
            return values[0] <= values[1]
        if operator == "add":
            return sum(values)
        if operator == "multiply":
            result = 1
            for value in values:
                result *= value
            return result
        if operator == "subtract":
            return values[0] - values[1]
        if operator == "divide":
            if values[1] == 0:
                raise RecipeError("Division by zero in recipe expression")
            return values[0] / values[1]
        if operator == "modulo":
            return values[0] % values[1]
        raise RecipeError(f"Unsupported expression operator '{operator}'")

    def execute(self, actions: Sequence[Mapping[str, Any]]) -> None:
        """Run actions in the order declared by the recipe."""
        for action_index, action in enumerate(actions):
            if not isinstance(action, Mapping) or len(action) != 1:
                raise RecipeError(f"Action {action_index} must contain exactly one action key")
            action_name, action_arguments = next(iter(action.items()))
            try:
                self._execute_action(action_name, action_arguments)
            except Exception as error:
                if isinstance(error, RecipeError):
                    raise RecipeError(f"Pipeline action {action_index} ({action_name}): {error}") from error
                raise RecipeError(f"Pipeline action {action_index} ({action_name}) failed: {error}") from error

    def _execute_action(self, action_name: str, arguments: Any) -> None:
        """Dispatch one tool call or structured control-flow action."""
        if action_name == "call":
            self._execute_tool_call(arguments)
        elif action_name == "set":
            if not isinstance(arguments, Mapping):
                raise RecipeError("'set' expects a mapping of variable names to values")
            for variable_name, expression in arguments.items():
                self._set_path(str(variable_name), self.evaluate(expression))
        elif action_name == "if":
            condition = bool(self.evaluate(arguments.get("condition", False)))
            branch_actions = arguments.get("then", []) if condition else arguments.get("else", [])
            self.execute(branch_actions)
        elif action_name == "for":
            self._execute_for(arguments)
        elif action_name == "while":
            self._execute_while(arguments, check_before_body=True)
        elif action_name == "do_while":
            self._execute_while(arguments, check_before_body=False)
        elif action_name == "repeat":
            self._execute_repeat(arguments)
        else:
            raise RecipeError(f"Unknown action '{action_name}'")

    def _execute_tool_call(self, arguments: Mapping[str, Any]) -> None:
        """Resolve a named tool, evaluate its inputs and store its result."""
        if not isinstance(arguments, Mapping) or "tool" not in arguments:
            raise RecipeError("'call' expects a tool name and optional args/save_as/append_to")
        tool_name = str(arguments["tool"])
        tool_arguments = self.evaluate(arguments.get("args", {}))
        tool_result = self.registry.get(tool_name)(self.context, tool_arguments)
        if "save_as" in arguments:
            self._set_path(str(arguments["save_as"]), tool_result)
        if "append_to" in arguments:
            destination = str(arguments["append_to"])
            existing_values = self._get_path(destination)
            if not isinstance(existing_values, list):
                raise RecipeError(f"append_to target '${destination}' must be a list")
            if isinstance(tool_result, list):
                existing_values.extend(tool_result)
            else:
                existing_values.append(tool_result)

    def _execute_for(self, arguments: Mapping[str, Any]) -> None:
        """Run a body once for every item in a referenced or constructed sequence."""
        items = self.evaluate(arguments.get("each", []))
        variable_name = str(arguments.get("as", "item"))
        if not isinstance(items, (list, tuple, range)):
            raise RecipeError("for.each must evaluate to a list, tuple or range")
        had_previous_value = variable_name in self.context.values
        previous_value = self.context.values.get(variable_name)
        for item in items:
            self.context.values[variable_name] = item
            self.execute(arguments.get("do", []))
        if had_previous_value:
            self.context.values[variable_name] = previous_value
        else:
            self.context.values.pop(variable_name, None)

    def _execute_while(self, arguments: Mapping[str, Any], check_before_body: bool) -> None:
        """Run while a condition holds, with a guard against accidental infinite loops."""
        condition = arguments.get("condition", False)
        maximum_iterations = int(arguments.get("max_iterations", 10000))
        body_actions = arguments.get("do", [])
        iteration_count = 0

        if not check_before_body:
            self.execute(body_actions)
            iteration_count += 1

        while bool(self.evaluate(condition)):
            if iteration_count >= maximum_iterations:
                raise RecipeError(
                    f"Loop exceeded max_iterations={maximum_iterations}; check its exit condition"
                )
            self.execute(body_actions)
            iteration_count += 1

    def _execute_repeat(self, arguments: Mapping[str, Any]) -> None:
        """Repeat a body a fixed number of times, exposing its zero-based index."""
        iteration_count = int(self.evaluate(arguments.get("count", 0)))
        if iteration_count < 0 or iteration_count > int(arguments.get("max_iterations", 10000)):
            raise RecipeError("repeat.count must be non-negative and within max_iterations")
        variable_name = str(arguments.get("as", "iteration"))
        had_previous_value = variable_name in self.context.values
        previous_value = self.context.values.get(variable_name)
        for iteration_index in range(iteration_count):
            self.context.values[variable_name] = iteration_index
            self.execute(arguments.get("do", []))
        if had_previous_value:
            self.context.values[variable_name] = previous_value
        else:
            self.context.values.pop(variable_name, None)
