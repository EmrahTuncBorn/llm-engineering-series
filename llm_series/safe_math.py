"""
Safe arithmetic evaluator
=========================

Replaces ``eval()`` in the calculator tool.

Why not ``eval(expression, {"__builtins__": {}})``?
    * Emptying ``__builtins__`` is not a sandbox: Python object introspection
      (``().__class__.__mro__ ...``) can still reach dangerous objects.
    * Even "pure arithmetic" can freeze the process: ``9**9**9`` has hundreds
      of millions of digits (a denial-of-service, DoS).

How this works
    The expression is parsed into an abstract syntax tree (AST) and walked
    node by node. Only a whitelist of node types is accepted: numbers, the
    arithmetic operators, unary +/-, a few named constants and a few math
    functions. Anything else (attribute access, names, calls to unknown
    functions, comprehensions, strings...) is rejected before evaluation.
"""

from __future__ import annotations

import ast
import math
import operator

MAX_EXPRESSION_LENGTH = 300
# Upper bound on the number of digits an integer power may produce.
MAX_POWER_DIGITS = 10_000

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "exp": math.exp,
    "log": math.log,      # log(x) natural, log(x, base) any base
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "degrees": math.degrees,
    "radians": math.radians,
    "floor": math.floor,
    "ceil": math.ceil,
}


class UnsafeExpressionError(ValueError):
    """Raised when an expression contains something outside the whitelist."""


def _check_power(base: float, exponent: float) -> None:
    """Reject integer powers whose result would be astronomically large."""
    if isinstance(base, int) and isinstance(exponent, int) and exponent > 0 and abs(base) > 1:
        estimated_digits = exponent * math.log10(abs(base))
        if estimated_digits > MAX_POWER_DIGITS:
            raise UnsafeExpressionError(
                f"Result would have about {estimated_digits:,.0f} digits; "
                f"the limit is {MAX_POWER_DIGITS:,}."
            )


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        # bool is a subclass of int; reject it explicitly.
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise UnsafeExpressionError(f"Unsupported literal: {node.value!r}")

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            _check_power(left, right)
        return _BINARY_OPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise UnsafeExpressionError(f"Unknown name: '{node.id}'")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise UnsafeExpressionError("Only these functions are allowed: " + ", ".join(sorted(_FUNCTIONS)))
        if node.keywords:
            raise UnsafeExpressionError("Keyword arguments are not supported.")
        args = [_eval_node(arg) for arg in node.args]
        return _FUNCTIONS[node.func.id](*args)

    raise UnsafeExpressionError(f"Unsupported syntax: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    """Evaluate an arithmetic expression safely and return the number.

    ``^`` is treated as exponentiation, because that is what users and LLMs
    almost always mean (in Python ``^`` is bitwise XOR, so ``2^3`` would
    silently give 1 instead of 8).
    """
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise UnsafeExpressionError(f"Expression is longer than {MAX_EXPRESSION_LENGTH} characters.")
    normalized = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Invalid expression: {exc.msg}") from exc
    return _eval_node(tree)


def calculate(expression: str) -> str:
    """Tool-friendly wrapper: always returns a string, never raises.

    Errors are returned as text on purpose: the agent reads the message and
    can correct its expression and try again (self-correction).
    """
    try:
        result = safe_eval(expression)
    except ZeroDivisionError:
        return "Calculation error: division by zero."
    except (UnsafeExpressionError, ValueError, OverflowError, TypeError) as exc:
        return f"Calculation error: {exc}"
    if isinstance(result, float):
        # Hide binary floating-point noise: sqrt(2)**2 is 2.0000000000000004.
        result = float(f"{result:.12g}")
        if result.is_integer() and abs(result) < 1e15:
            result = int(result)
    return str(result)
