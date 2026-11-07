import pytest

from llm_series.safe_math import UnsafeExpressionError, calculate, safe_eval


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("2 + 2 * 5", "12"),
        ("(2 + 2) * 5", "20"),
        ("10 / 4", "2.5"),
        ("7 // 2", "3"),
        ("7 % 3", "1"),
        ("-3 + +5", "2"),
        ("2 ** 10", "1024"),
        ("2 ^ 3", "8"),          # ^ means power, not XOR (XOR would give 1)
        ("sqrt(16)", "4"),
        ("sqrt(2) ^ 2", "2"),
        ("log(e)", "1"),
        ("log(8, 2)", "3"),
        ("round(pi, 3)", "3.142"),
        ("12 × 3 ÷ 4", "9"),
    ],
)
def test_valid_expressions(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hacked')",
        "().__class__.__mro__",
        "open('secret.txt')",
        "[x for x in range(10)]",
        "'a' * 10",
        "lambda: 1",
        "x + 1",
        "True + 1",
        "sqrt(x=4)",
    ],
)
def test_unsafe_expressions_are_rejected(expression):
    with pytest.raises(UnsafeExpressionError):
        safe_eval(expression)
    assert calculate(expression).startswith("Calculation error")


def test_huge_power_is_rejected_quickly():
    # 9**9**9 would have ~370 million digits and freeze the process.
    assert "digits" in calculate("9 ** 9 ** 9")


def test_errors_are_returned_as_text_not_raised():
    assert calculate("1 / 0") == "Calculation error: division by zero."
    assert calculate("2 +").startswith("Calculation error")
    assert calculate("sqrt(-1)").startswith("Calculation error")
    assert calculate("1" * 400).startswith("Calculation error")
