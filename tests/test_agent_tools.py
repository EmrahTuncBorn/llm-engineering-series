import pandas as pd
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from agent_tools import MAX_OUTPUT_ROWS, build_tools
from llm_series.documents import build_vectorstore

DF = pd.DataFrame(
    {
        "city": ["Izmir", "Manisa", "Izmir", "Aydin"],
        "temp": [21.5, 19.0, 23.5, 22.0],
        "rainy": [False, True, False, True],
    }
)


def tool(name, vectorstore=None, dataframe=DF):
    return {t.name: t for t in build_tools(vectorstore, dataframe)}[name]


def run(**kwargs):
    return tool("analyze_data").invoke(kwargs)


def test_tool_set_without_tavily_key():
    names = [t.name for t in build_tools(None, None)]
    assert names == ["calculator", "wikipedia", "document_search", "analyze_data"]


def test_tool_set_with_tavily_key(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake")
    names = [t.name for t in build_tools(None, None)]
    assert "tavily_search" in names and len(names) == 5


def test_tools_report_missing_uploads():
    assert "No CSV" in tool("analyze_data", dataframe=None).invoke({"operation": "shape"})
    assert "No PDF" in tool("document_search").invoke({"query": "anything"})


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"operation": "shape"}, "4 rows, 3 columns"),
        ({"operation": "mean", "column": "temp"}, "21.5"),
        ({"operation": "max", "column": "temp"}, "23.5"),
        ({"operation": "nunique", "column": "city"}, "3"),
        ({"operation": "mean", "column": "temp", "filter_column": "city",
          "filter_operator": "==", "filter_value": "Izmir"}, "22.5"),
        ({"operation": "count", "column": "city", "filter_column": "temp",
          "filter_operator": ">=", "filter_value": "22"}, "2"),
        ({"operation": "count", "column": "city", "filter_column": "rainy",
          "filter_operator": "==", "filter_value": "true"}, "2"),
        ({"operation": "count", "column": "city", "filter_column": "city",
          "filter_operator": "contains", "filter_value": "izm"}, "2"),
    ],
)
def test_operations(kwargs, expected):
    assert run(**kwargs) == expected


def test_groupby_and_value_counts():
    assert "Izmir" in run(operation="groupby_mean", column="temp", group_by="city")
    out = run(operation="value_counts", column="city")
    assert "Izmir" in out and "3 distinct values" in out


def test_helpful_errors_let_the_model_self_correct():
    assert "Available columns" in run(operation="mean", column="temperature")
    assert "not numeric" in run(operation="mean", column="city")
    assert "needs a numeric column" in run(
        operation="count", column="city", filter_column="city", filter_operator=">", filter_value="A"
    )
    assert "not a number" in run(
        operation="count", column="city", filter_column="temp", filter_operator=">", filter_value="hot"
    )
    assert "all three" in run(operation="shape", filter_column="city")
    assert "no rows" in run(
        operation="shape", filter_column="city", filter_operator="==", filter_value="Paris"
    )


def test_invalid_operation_is_rejected_by_the_schema():
    with pytest.raises(Exception):
        run(operation="__class__")


def test_large_outputs_are_truncated():
    big = pd.DataFrame({"id": [f"id-{i}" for i in range(500)]})
    out = tool("analyze_data", dataframe=big).invoke({"operation": "value_counts", "column": "id"})
    assert f"first {MAX_OUTPUT_ROWS} of 500" in out
    head = tool("analyze_data", dataframe=big).invoke({"operation": "head", "n": 10_000})
    assert len(head.splitlines()) == MAX_OUTPUT_ROWS + 1  # header + capped rows


def test_document_search_returns_passages_with_pages(make_pdf):
    store, _ = build_vectorstore(make_pdf(["Darboux frames on surfaces", "Geodesic curvature"]), DeterministicFakeEmbedding(size=64))
    out = tool("document_search", vectorstore=store).invoke({"query": "geodesic"})
    assert "[page" in out


def test_calculator_tool_is_safe():
    calc = tool("calculator")
    assert calc.invoke({"expression": "2^3"}) == "8"
    assert calc.invoke({"expression": "__import__('os')"}).startswith("Calculation error")
